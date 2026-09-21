"""Small, cohort-independent runtime helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def json_load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_dump(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalized_serial(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    match = re.search(r"(\d+)", str(value).strip())
    if match is None:
        raise ValueError(f"cannot normalize serial from {value!r}")
    return int(match.group(1))


def resolve_existing_directory(candidates: Iterable[str | Path], label: str) -> Path:
    checked = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        checked.append(str(path))
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(f"No {label} directory found; checked {checked}")


def clean_report(text: str) -> str:
    value = str(text).strip()
    start = value.find("<所见>")
    if start >= 0:
        value = value[start:]
    end = value.find("</结论>")
    return (value[: end + len("</结论>")] if end >= 0 else value).strip()


def device_from_arg(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)

