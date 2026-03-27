"""
Config loader.  Supports YAML and JSON.

Config files are merged in order:
  1. defaults (built-in)
  2. user config file
  3. CLI overrides

Usage:
    cfg = load_config("configs/my_experiment.yaml")
    llm_cfg = cfg["llm"]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_DEFAULTS: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "model_name": "llama3.1:8b-instruct-q4_K_M",
        "temperature": 0.3,
        "max_tokens": 512,
    },
    "extractor": {
        "provider": "ollama",
        "model_name": "llama3.1:8b-instruct-q4_K_M",
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    "verifier": {
        "backend": "transformers",          # "transformers" | "llm"
        "model_name": "facebook/bart-large-mnli",
        "device": "cpu",
    },
    "prompts": {
        "summarize": "prompts/summarize.txt",
        "extract_claims": "prompts/extract_claims.txt",
        "extract_facts": "prompts/extract_facts.txt",
    },
    "evaluation": {
        "n_runs": 1,
        "target_words": 150,
    },
    "output": {
        "dir": "outputs",
        "save_source": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = _DEFAULTS.copy()

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                file_cfg = yaml.safe_load(raw)
            except ImportError as exc:
                raise ImportError("pip install pyyaml") from exc
        elif path.suffix == ".json":
            file_cfg = json.loads(raw)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        if file_cfg:
            cfg = _deep_merge(cfg, file_cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return cfg
