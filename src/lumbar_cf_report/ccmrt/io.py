from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
import yaml


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(obj, path):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False)+"\n")


def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

