"""Run an experiment: every model in the config on every scenario.

Output goes to <output_dir>/<experiment name>/<commit>/episodes.jsonl, one
line per (model, scenario), where <commit> is the Git commit of the code
(see provenance.py). Re-running the same command on the same commit skips
finished episodes, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agents.base import Episode
from .agents.baseline import run_baseline
from .agents.llm import run_llm
from .config import load_model
from .market import Firm, generate_pool
from .provenance import run_dir_for
from .scenario import build_scenario


def run_episode(cfg: dict, model: dict, pool: list[Firm], idx: int,
                firm_overrides: dict | None = None, feature_overrides: dict | None = None, tag: str | None = None,
                code_version: str | None = None) -> dict:
    exp = cfg["experiment"]
    scenario = build_scenario(exp, cfg["features"], pool, idx, firm_overrides, feature_overrides)
    ep = Episode(scenario, cfg["web"], cfg["search"], cfg["agent"], exp["shortlist_size"])
    try:
        if model.get("provider") == "baseline":
            usage, steps, error, transcript = run_baseline(ep)
        else:
            usage, steps, error, transcript = run_llm(ep, model)
    except Exception as e:  # keep the run going; the failure is logged on the episode
        usage, steps, error, transcript = {}, 0, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}", None
    rec = ep.record(model, usage, steps, error, transcript)
    rec["experiment"] = exp["name"]
    rec["arm"] = tag
    rec["code_version"] = code_version
    return rec


def _done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("completed") or not r.get("error"):
                    keys.add((r["model"], r["scenario_id"], r.get("arm")))
    return keys


def run_experiment(cfg: dict, limit: int | None = None, out_name: str = "episodes.jsonl", jobs: list | None = None) -> Path:
    exp = cfg["experiment"]
    out_dir, info = run_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    (out_dir / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2))
    info_path = out_dir / "run_info.json"
    if not info_path.exists():
        info_path.write_text(json.dumps(info, indent=2))
    pool = generate_pool(exp["category"], exp["pool_size"], exp["corpus_seed"])
    (out_dir / "firms.json").write_text(json.dumps([f.to_dict() for f in pool], indent=2))

    models = [load_model(m, cfg["_base"]) for m in cfg["models"]]
    n = min(exp["n_scenarios"], limit) if limit else exp["n_scenarios"]
    if jobs is None:
        jobs = [(m, i, None, None, None) for m in models for i in range(n)]
    done = _done_keys(out)
    todo = [j for j in jobs if (j[0]["display_name"], j[1], j[4]) not in done]
    print(f"{len(jobs) - len(todo)} episodes already done, {len(todo)} to run -> {out}")

    lock = threading.Lock()
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool_ex:
        futs = [pool_ex.submit(run_episode, cfg, m, pool, i, fo, feo, tag, out_dir.name) for (m, i, fo, feo, tag) in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if k % 25 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} done")
    return out
