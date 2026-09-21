from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Self-contained MLP matching the APREB A/R tensor architecture.

    Matching parameter names allow optional initialization from the legacy v3 checkpoint,
    but CCMRT does not import any APREB Python module. After optional
    initialization, the selected A/R parameters remain trainable.
    """
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.1, final_norm=False):
        super().__init__()
        layers = [
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(out_dim)),
        ]
        if final_norm:
            layers.append(nn.LayerNorm(int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(int(dims[i + 1])), nn.GELU()]
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class AnatomyPathologyBottleneck(nn.Module):
    """Trainable anatomy/residual bottleneck for CCMRT.

    A and R are encoded online from frozen upstream X/E/C. They are not treated as fixed
    bank variables during joint intervention training.
    """
    def __init__(
        self,
        segment_dim: int,
        task_dim: int,
        coordinate_dim: int,
        anatomy_dim: int = 96,
        residual_dim: int = 48,
        task_embedding_dim: int = 24,
        coordinate_latent_dim: int = 48,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.segment_dim = int(segment_dim)
        self.task_dim = int(task_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.anatomy_dim = int(anatomy_dim)
        self.residual_dim = int(residual_dim)
        self.task_embedding_dim = int(task_embedding_dim)
        self.coordinate_latent_dim = int(coordinate_latent_dim)

        self.coordinate_encoder = ResidualMLP(coordinate_dim, coordinate_latent_dim, hidden_dim, dropout, True)
        self.anatomy_encoder = ResidualMLP(segment_dim + coordinate_latent_dim, anatomy_dim, hidden_dim, dropout, True)
        self.task_embedding = nn.Embedding(3, task_embedding_dim)
        self.anatomy_evidence_decoder = ResidualMLP(
            anatomy_dim + coordinate_latent_dim + task_embedding_dim,
            task_dim,
            hidden_dim,
            dropout,
            False,
        )
        self.residual_encoder = ResidualMLP(
            task_dim * 2 + anatomy_dim + coordinate_latent_dim + task_embedding_dim,
            residual_dim,
            hidden_dim,
            dropout,
            True,
        )
        self.residual_evidence_decoder = ResidualMLP(
            residual_dim + task_embedding_dim,
            task_dim,
            hidden_dim,
            dropout,
            False,
        )
        self.pathology_head = ResidualMLP(
            residual_dim + task_embedding_dim,
            1,
            max(64, hidden_dim // 2),
            dropout,
            False,
        )
        self.segment_decoder = ResidualMLP(anatomy_dim, segment_dim, hidden_dim, dropout, False)
        self.coordinate_decoder = ResidualMLP(anatomy_dim, coordinate_dim, hidden_dim, dropout, False)

    def encode_anatomy(self, segment_base, coordinate_state):
        s = torch.as_tensor(segment_base).float()
        c = torch.as_tensor(coordinate_state).float()
        ce = self.coordinate_encoder(c)
        a = self.anatomy_encoder(torch.cat([s, ce], -1))
        return a, ce

    def _task_grid(self, b, l, device):
        idx = torch.arange(3, device=device).view(1, 1, 3).expand(b, l, 3)
        return idx, self.task_embedding(idx)

    def anatomy_component(self, anatomy, coord_latent):
        b, l = anatomy.shape[:2]
        idx, te = self._task_grid(b, l, anatomy.device)
        aa = anatomy[:, :, None, :].expand(-1, -1, 3, -1)
        cc = coord_latent[:, :, None, :].expand(-1, -1, 3, -1)
        base = self.anatomy_evidence_decoder(torch.cat([aa, cc, te], -1))
        return base, idx, te

    def encode_residual(self, task_features, anatomy, coord_latent, base, task_emb):
        x = torch.as_tensor(task_features).float()
        aa = anatomy[:, :, None, :].expand(-1, -1, 3, -1)
        cc = coord_latent[:, :, None, :].expand(-1, -1, 3, -1)
        return self.residual_encoder(torch.cat([x, x - base, aa, cc, task_emb], -1))

    def decode_from_latents(self, anatomy, residual, coordinate_state):
        ce = self.coordinate_encoder(coordinate_state)
        base, _, te = self.anatomy_component(anatomy, ce)
        res = self.residual_evidence_decoder(torch.cat([residual, te], -1))
        return base + res, base, res

    def forward(self, segment_base, task_features, coordinate_state) -> Dict[str, torch.Tensor]:
        a, ce = self.encode_anatomy(segment_base, coordinate_state)
        base, _, te = self.anatomy_component(a, ce)
        r = self.encode_residual(task_features, a, ce, base, te)
        res = self.residual_evidence_decoder(torch.cat([r, te], -1))
        ehat = base + res
        logits = self.pathology_head(torch.cat([r, te], -1)).squeeze(-1)
        return {
            "anatomy": a,
            "residual": r,
            "coordinate_latent": ce,
            "anatomy_component": base,
            "residual_component": res,
            "reconstructed_task_features": ehat,
            "pathology_logits": logits,
            "pathology_probabilities": torch.sigmoid(logits),
            "reconstructed_segment_base": self.segment_decoder(a),
            "reconstructed_coordinate": self.coordinate_decoder(a),
        }


class CoordinateConditionedResidualTransporter(nn.Module):
    """Coordinate-conditioned residual transport with structural reciprocity.

    Default ``reversible_cayley`` mode separates transport into:
      1) a scalar evidence-strength change, and
      2) an orthogonal Cayley rotation in residual direction.

    The conditioning latent is explicitly antisymmetrized under source/destination swap.
    Therefore swapping B->A to A->B negates the transport parameters. For the default
    mode this yields, up to floating-point error:

        T(T(R_B, C_B, C_A), C_A, C_B) ~= R_B

    Same-coordinate identity is also hard by construction because every learned change is
    multiplied by a coordinate identity gate that is exactly zero when C_A == C_B.

    ``film_lowrank`` is retained only as an architectural ablation and is not expected to
    have the same structural cycle property.
    """
    def __init__(
        self,
        r_dim=48,
        c_dim=47,
        task_emb_dim=12,
        hidden=96,
        rank=6,
        dropout=0.05,
        mode="reversible_cayley",
        use_coordinate_conditioning=True,
        identity_tau=2.0,
        rotation_scale=0.12,
        strength_scale=0.15,
        film_scale=0.12,
    ):
        super().__init__()
        self.r_dim = int(r_dim)
        self.c_dim = int(c_dim)
        self.rank = int(rank)
        self.mode = str(mode)
        self.use_coordinate_conditioning = bool(use_coordinate_conditioning)
        self.identity_tau = float(identity_tau)
        self.rotation_scale = float(rotation_scale)
        self.strength_scale = float(strength_scale)
        self.film_scale = float(film_scale)

        self.task_emb = nn.Embedding(3, task_emb_dim)

        # Symmetric/antisymmetric pair parameterization:
        # midpoint, signed delta, |delta|, quality mean/delta, task embedding.
        cond_dim = c_dim * 3 + 2 + task_emb_dim
        self.cond = MLP([cond_dim, hidden, hidden], 0.0 if self.mode == "reversible_cayley" else dropout)

        # Bias-free heads preserve odd symmetry of h_odd.
        self.rotation_coeff = nn.Linear(hidden, self.rank, bias=False)
        self.log_strength = nn.Linear(hidden, 1, bias=False)
        self.film_gamma = nn.Linear(hidden, r_dim, bias=False)
        self.film_beta = nn.Linear(hidden, r_dim, bias=False)

        # Task-specific learned skew bases. Nonzero basis initialization prevents the
        # zero-head/zero-basis product from killing initial gradients to rotation_coeff.
        self.skew_raw = nn.Parameter(torch.empty(3, self.rank, r_dim, r_dim))
        nn.init.normal_(self.skew_raw, mean=0.0, std=0.015)

        nn.init.zeros_(self.rotation_coeff.weight)
        nn.init.zeros_(self.log_strength.weight)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.weight)

    def _identity_gate(self, ca, cb):
        delta = ca - cb
        if self.use_coordinate_conditioning:
            norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            return torch.tanh(norm / max(self.identity_tau, 1e-6))
        # Ablation is allowed to know only whether this is the exact identity call; it does
        # not receive coordinate magnitudes/directions as learned features.
        return (delta.abs().sum(-1, keepdim=True) > 1e-12).to(delta.dtype)

    def _pair_features(self, ca, cb, qa, qb, task_index, reverse=False):
        # ca = destination coordinate, cb = source coordinate.
        delta = ca - cb
        midpoint = 0.5 * (ca + cb)
        abs_delta = delta.abs()
        qmean = 0.5 * (qa + qb)
        qdelta = qa - qb

        if reverse:
            delta = -delta
            qdelta = -qdelta

        if not self.use_coordinate_conditioning:
            midpoint = torch.zeros_like(midpoint)
            delta = torch.zeros_like(delta)
            abs_delta = torch.zeros_like(abs_delta)

        te = self.task_emb(task_index.long())
        return torch.cat(
            [midpoint, delta, abs_delta, qmean[:, None], qdelta[:, None], te],
            dim=-1,
        )

    def _odd_condition(self, ca, cb, qa, qb, task_index):
        # Shared network + explicit subtraction guarantees h_odd(A,B) = -h_odd(B,A)
        # when qualities are swapped together with coordinates.
        hp = self.cond(self._pair_features(ca, cb, qa, qb, task_index, reverse=False))
        hm = self.cond(self._pair_features(ca, cb, qa, qb, task_index, reverse=True))
        return 0.5 * (hp - hm)

    def _skew_matrix(self, coeff, task_index):
        # raw - raw^T is exactly skew-symmetric.
        basis = self.skew_raw - self.skew_raw.transpose(-1, -2)
        selected = basis[task_index.long()]  # [B,K,D,D]
        return torch.einsum("bk,bkij->bij", coeff, selected) / max(1.0, self.r_dim ** 0.5)

    def _cayley(self, k):
        # Q(K)=(I-K)^-1(I+K), Q(-K)=Q(K)^-1 for skew K.
        b, d, _ = k.shape
        eye = torch.eye(d, device=k.device, dtype=k.dtype).expand(b, d, d)
        return torch.linalg.solve(eye - k, eye + k)

    def forward(self, rb, cb, ca, task_index, qb=None, qa=None):
        if qa is None:
            qa = torch.zeros(rb.shape[0], device=rb.device, dtype=rb.dtype)
        if qb is None:
            qb = torch.zeros(rb.shape[0], device=rb.device, dtype=rb.dtype)

        h = self._odd_condition(ca, cb, qa, qb, task_index)
        g = self._identity_gate(ca, cb)

        if self.mode == "reversible_cayley":
            coeff = g * self.rotation_scale * torch.tanh(self.rotation_coeff(h))
            log_s = g * self.strength_scale * torch.tanh(self.log_strength(h))
            k = self._skew_matrix(coeff, task_index)
            q = self._cayley(k)
            rotated = torch.bmm(q, rb.unsqueeze(-1)).squeeze(-1)
            return torch.exp(log_s) * rotated

        if self.mode == "film_lowrank":
            gamma = torch.exp(g * self.film_scale * torch.tanh(self.film_gamma(h)))
            beta = g * self.film_scale * torch.tanh(self.film_beta(h))
            return gamma * rb + beta

        raise ValueError(f"Unknown transport mode: {self.mode}")


class CCMRTModel(nn.Module):
    def __init__(
        self,
        x_dim=256,
        c_dim=47,
        task_dim=128,
        a_dim=96,
        r_dim=48,
        task_embedding_dim=24,
        coord_latent_dim=48,
        ar_hidden=256,
        hidden=256,
        dropout=0.1,
        transport_hidden=96,
        transport_rank=6,
        transport_mode="reversible_cayley",
        use_coordinate_conditioning=True,
        identity_tau=2.0,
        transport_rotation_scale=0.12,
        transport_strength_scale=0.15,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.c_dim = int(c_dim)
        self.task_dim = int(task_dim)
        self.a_dim = int(a_dim)
        self.r_dim = int(r_dim)

        self.ar = AnatomyPathologyBottleneck(
            x_dim,
            task_dim,
            c_dim,
            a_dim,
            r_dim,
            task_embedding_dim,
            coord_latent_dim,
            ar_hidden,
            dropout,
        )

        # Warmed factual guard is frozen only during Phase B and acts as a fixed audit ruler.
        self.state_coord_encoder = MLP([c_dim, 96, 48], dropout)
        self.evidence_composer = MLP([3 * task_dim + 48, hidden, hidden, x_dim], dropout)
        self.state_task_heads = nn.ModuleList([MLP([x_dim, 96, 1], dropout) for _ in range(3)])
        self.anatomy_probe = MLP([x_dim, 160, a_dim], dropout)
        self.coord_probe = MLP([x_dim, 160, c_dim], dropout)

        self.transporter = CoordinateConditionedResidualTransporter(
            r_dim,
            c_dim,
            hidden=transport_hidden,
            rank=transport_rank,
            dropout=dropout * 0.5,
            mode=transport_mode,
            use_coordinate_conditioning=use_coordinate_conditioning,
            identity_tau=identity_tau,
            rotation_scale=transport_rotation_scale,
            strength_scale=transport_strength_scale,
        )

    def _state_from_evidence(self, ehat, c):
        ce = self.state_coord_encoder(c)
        xhat = self.evidence_composer(torch.cat([ehat.flatten(-2), ce], -1))
        logits = torch.cat([h(xhat) for h in self.state_task_heads], -1)
        return {
            "xhat": xhat,
            "state_logits": logits,
            "ahat_probe": self.anatomy_probe(xhat),
            "chat_probe": self.coord_probe(xhat),
        }

    def forward(self, x, e, c):
        ar = self.ar(x, e, c)
        state = self._state_from_evidence(ar["reconstructed_task_features"], c)
        return {**ar, **state}

    def encode_level(self, x, e, c):
        out = self.forward(x[:, None, :], e[:, None, :, :], c[:, None, :])
        squeezed = {}
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == 1:
                squeezed[k] = v[:, 0]
            else:
                squeezed[k] = v
        return squeezed

    @staticmethod
    def gather_task(r_all, task_index):
        return r_all[torch.arange(r_all.shape[0], device=r_all.device), task_index.long()]

    @staticmethod
    def scatter_task(r_all, task_index, value):
        out = r_all.clone()
        out[torch.arange(out.shape[0], device=out.device), task_index.long()] = value
        return out

    def state_from_level_latents(self, a, r_all, c):
        ehat, base, res = self.ar.decode_from_latents(a[:, None, :], r_all[:, None, :, :], c[:, None, :])
        ehat = ehat[:, 0]
        state = self._state_from_evidence(ehat[:, None, :, :], c[:, None, :])
        return {
            "reconstructed_task_features": ehat,
            "anatomy_component": base[:, 0],
            "residual_component": res[:, 0],
            "xhat": state["xhat"][:, 0],
            "state_logits": state["state_logits"][:, 0],
            "ahat_probe": state["ahat_probe"][:, 0],
            "chat_probe": state["chat_probe"][:, 0],
        }

    def counterfactual_from_encoded(self, fa, fb, task_index, alpha, qa=None, qb=None, transport=True):
        ra = fa["residual"]
        rb = fb["residual"]
        ca = fa["coordinate_input"]
        cb = fb["coordinate_input"]
        aa = fa["anatomy"]

        ra_t = self.gather_task(ra, task_index)
        rb_t = self.gather_task(rb, task_index)

        if qa is not None and qa.ndim == 2:
            qa_t = qa[torch.arange(qa.shape[0], device=qa.device), task_index.long()]
        else:
            qa_t = qa
        if qb is not None and qb.ndim == 2:
            qb_t = qb[torch.arange(qb.shape[0], device=qb.device), task_index.long()]
        else:
            qb_t = qb

        rt = self.transporter(rb_t, cb, ca, task_index, qb=qb_t, qa=qa_t) if transport else rb_t
        if alpha.ndim == 1:
            alpha = alpha[:, None]

        # Direct and transported methods use the same linear alpha path; only endpoint differs.
        rcf = ra_t + alpha * (rt - ra_t)
        rall = self.scatter_task(ra, task_index, rcf)
        out = self.state_from_level_latents(aa, rall, ca)
        out.update({
            "r_target": rt,
            "r_cf": rcf,
            "rall_cf": rall,
            "ra_target": ra_t,
            "rb_target": rb_t,
        })
        return out

    def encode_pair_level(self, x, e, c):
        out = self.encode_level(x, e, c)
        out["coordinate_input"] = c
        return out

    def load_apreb_checkpoint(self, path, strict=True):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if "state_dict" not in obj:
            raise RuntimeError(f"invalid legacy A/R checkpoint: {path}")
        self.ar.load_state_dict(obj["state_dict"], strict=strict)
        return obj

    def freeze_transporter(self):
        for p in self.transporter.parameters():
            p.requires_grad_(False)

    def set_factual_warmup_trainable(self):
        for p in self.parameters():
            p.requires_grad_(True)
        self.freeze_transporter()

    def set_joint_trainable(self):
        # Core requirement of v2.1: A/R + Transporter stay trainable.
        # Only the warmed audit/composer ruler is frozen in Phase B.
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.ar.parameters():
            p.requires_grad_(True)
        for p in self.transporter.parameters():
            p.requires_grad_(True)

    def anatomy_parameters(self):
        mods = [
            self.ar.coordinate_encoder,
            self.ar.anatomy_encoder,
            self.ar.anatomy_evidence_decoder,
            self.ar.segment_decoder,
            self.ar.coordinate_decoder,
        ]
        return [p for m in mods for p in m.parameters()]

    def residual_parameters(self):
        mods = [
            self.ar.task_embedding,
            self.ar.residual_encoder,
            self.ar.residual_evidence_decoder,
            self.ar.pathology_head,
        ]
        return [p for m in mods for p in m.parameters()]

