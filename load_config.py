"""Resolve a data-spec config and its FRED API key.

The spec config (e.g. configs/us.yaml) is meant to live in version control
and contains only public information (series codes, transformations, sample
windows). The API key is sourced separately, in order:

  1. `api_key` field in the spec file (only honored if it's not the example
     placeholder), for ad-hoc use.
  2. `FRED_API_KEY` environment variable.
  3. A sibling secrets file (`config.yaml` next to the spec, or in CWD)
     that contains at least `api_key:`.

`load_config` returns a dict identical to the spec with `api_key` merged in.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


PLACEHOLDER_PREFIXES = ("<", "your", "YOUR")


def _looks_like_placeholder(val: str) -> bool:
    return any(val.startswith(p) for p in PLACEHOLDER_PREFIXES)


def load_config(spec_path: str, secrets_path: Optional[str] = None) -> dict[str, Any]:
    spec = Path(spec_path)
    cfg = yaml.safe_load(spec.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Spec config at {spec_path} did not parse to a dict.")

    inline = cfg.get("api_key")
    if isinstance(inline, str) and inline and not _looks_like_placeholder(inline):
        return cfg

    env_key = os.environ.get("FRED_API_KEY")
    if env_key:
        cfg["api_key"] = env_key
        return cfg

    candidates: list[Path] = []
    if secrets_path:
        candidates.append(Path(secrets_path))
    candidates.append(spec.parent / "config.yaml")
    candidates.append(Path.cwd() / "config.yaml")

    for p in candidates:
        if p.exists() and p.resolve() != spec.resolve():
            secrets = yaml.safe_load(p.read_text()) or {}
            if "api_key" in secrets:
                cfg["api_key"] = secrets["api_key"]
                return cfg

    raise RuntimeError(
        "No FRED api_key found. Tried: spec file, FRED_API_KEY env var, "
        f"and secrets candidates {[str(p) for p in candidates]}."
    )
