"""Run an experiment: every model, under every platform profile, on every scenario.

Output goes to <output_dir>/<experiment name>/<commit>/episodes.jsonl, one
line per (model, profile, scenario), where <commit> is the Git commit of the
code (see provenance.py). Re-running the same command on the same commit
skips finished episodes, so an interrupted run picks up where it stopped.

A `budget` block in the config stops a run before it starts new episodes once
total tokens pass a cap (tokens already in the output file count too):
  budget:
    max_input_tokens: 20000000
    max_output_tokens: 2000000
"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .agents.base import Episode
from .agents.baseline import run_baseline
from .agents.llm import run_llm
from .config import apply_profile, load_model
from .market import Firm, generate_pool
from .provenance import run_dir_for
from .scenario import build_scenario


def run_episode(cfg: dict, model: dict, pool: list[Firm], idx: int,
                firm_overrides: dict | None = None, feature_overrides: dict | None = None, tag: str | None = None,
                code_version: str | None = None) -> dict:
    """Run one episode. `cfg` must already have its profile applied."""
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
    rec["profile"] = cfg.get("profile")
    rec["arm"] = tag
    rec["code_version"] = code_version
    return rec


def job(model: dict, idx: int, profile: str | None, firm_overrides=None, feature_overrides=None, tag=None) -> dict:
    return {"model": model, "idx": idx, "profile": profile, "fo": firm_overrides, "feo": feature_overrides, "tag": tag}


def _key(model_name: str, profile, idx: int, tag) -> tuple:
    return (model_name, profile, idx, tag)


def _read_log(path: Path) -> tuple[set, int, int]:
    keys, tin, tout = set(), 0, 0
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                u = r.get("usage") or {}
                tin += u.get("input_tokens", 0)
                tout += u.get("output_tokens", 0)
                if r.get("completed") or not r.get("error"):
                    keys.add(_key(r["model"], r.get("profile"), r["scenario_id"], r.get("arm")))
    return keys, tin, tout


def _over_budget(budget: dict, tin: int, tout: int) -> str | None:
    if budget.get("max_input_tokens") and tin >= budget["max_input_tokens"]:
        return f"input tokens {tin:,} reached the cap of {budget['max_input_tokens']:,}"
    if budget.get("max_output_tokens") and tout >= budget["max_output_tokens"]:
        return f"output tokens {tout:,} reached the cap of {budget['max_output_tokens']:,}"
    return None


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
        jobs = [job(m, i, p) for p in cfg["profiles"] for m in models for i in range(n)]
    profile_cfgs = {p: apply_profile(cfg, p) for p in {j["profile"] for j in jobs}}
    done, tin, tout = _read_log(out)
    todo = [j for j in jobs if _key(j["model"]["display_name"], j["profile"], j["idx"], j["tag"]) not in done]
    print(f"{len(jobs) - len(todo)} episodes already done, {len(todo)} to run -> {out}")

    budget, stopped, finished = cfg.get("budget", {}), None, 0
    queue = list(todo)
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
        running = set()
        while queue or running:
            stopped = stopped or _over_budget(budget, tin, tout)
            while queue and not stopped and len(running) < cfg["max_workers"]:
                j = queue.pop(0)
                running.add(ex.submit(run_episode, profile_cfgs[j["profile"]], j["model"], pool, j["idx"],
                                      j["fo"], j["feo"], j["tag"], out_dir.name))
            if not running:
                break
            done_now, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in done_now:
                rec = fut.result()
                u = rec.get("usage") or {}
                tin += u.get("input_tokens", 0)
                tout += u.get("output_tokens", 0)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                finished += 1
                if finished % 25 == 0:
                    print(f"  {finished}/{len(todo)} done; tokens in/out so far {tin:,}/{tout:,}")
    if stopped:
        print(f"Budget stop: {stopped}. {len(queue)} episodes left unstarted; rerun with a higher cap to continue.")
    print(f"  {finished}/{len(todo)} done; tokens in/out {tin:,}/{tout:,}")
    return out
