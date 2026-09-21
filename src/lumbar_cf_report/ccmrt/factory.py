from __future__ import annotations
from .model import CCMRTModel


def make_model(bank, cfg):
    m = cfg["model"]
    return CCMRTModel(
        x_dim=bank.X.shape[-1],
        c_dim=bank.C.shape[-1],
        task_dim=bank.E.shape[-1],
        a_dim=int(m.get("anatomy_dim", 96)),
        r_dim=int(m.get("residual_dim", 48)),
        task_embedding_dim=int(m.get("task_embedding_dim", 24)),
        coord_latent_dim=int(m.get("coordinate_latent_dim", 48)),
        ar_hidden=int(m.get("ar_hidden", 256)),
        hidden=int(m.get("hidden", 256)),
        dropout=float(m.get("dropout", 0.1)),
        transport_hidden=int(m.get("transport_hidden", 96)),
        transport_rank=int(m.get("transport_rank", 6)),
        transport_mode=str(m.get("transport_mode", "reversible_cayley")),
        use_coordinate_conditioning=bool(m.get("use_coordinate_conditioning", True)),
        identity_tau=float(m.get("identity_tau", 2.0)),
        transport_rotation_scale=float(m.get("transport_rotation_scale", 0.12)),
        transport_strength_scale=float(m.get("transport_strength_scale", 0.15)),
    )

