"""Experiment and model configuration (YAML files in configs/ and models/)."""

from __future__ import annotations

from pathlib import Path

import yaml

from .search import DEFAULT_SEARCH
from .web import DEFAULT_WEB

DEFAULT_EXPERIMENT = {
    "name": "pilot",
    "category": "freight_trucking",
    "n_scenarios": 20,
    "seed": 7,
    "corpus_seed": 1,
    "pool_size": 40,
    "market_size": 12,
    "min_full_fit": 2,
    "shortlist_size": 3,
}
DEFAULT_FEATURES = {
    "markdown": 0.5, "llms_txt": 0.5, "schema_jsonld": 0.5,
    "js_rendered": 0.3, "faq_page": 0.5, "directory_listing": 0.5,
    "descriptive_heading": 0.5,
}
DEFAULT_AGENT = {"max_searches": 3, "max_fetches": 5, "max_steps": 14, "save_transcripts": True}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict:
    raw = load_yaml(path)
    base = Path(path).resolve().parent.parent
    cfg = {
        "experiment": {**DEFAULT_EXPERIMENT, **raw.get("experiment", {})},
        "features": {**DEFAULT_FEATURES, **raw.get("features", {})},
        "web": {**DEFAULT_WEB, **raw.get("web", {})},
        "search": {**DEFAULT_SEARCH, **raw.get("search", {})},
        "agent": {**DEFAULT_AGENT, **raw.get("agent", {})},
        "models": raw.get("models", ["models/baseline.yaml"]),
        "output_dir": raw.get("output_dir", "runs"),
        "max_workers": raw.get("max_workers", 4),
        "seller": raw.get("seller", {}),
        "_base": str(base),
    }
    return cfg


def load_model(path: str, base: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = Path(base) / p
    m = load_yaml(p)
    m.setdefault("display_name", m.get("model", p.stem))
    m.setdefault("temperature", 1.0)
    return m
