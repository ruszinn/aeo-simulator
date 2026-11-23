"""Seller-side experiment: one firm changes its site; everything else stays fixed.

For each scenario a focal firm is drawn (seeded). The same scenarios run
twice: arm "before" as generated, arm "after" with the focal firm's change.
Because both arms share seeds, any difference in the focal firm's
recommendation rate comes from the change.

Strategies:
  aeo_checklist   switch on markdown, llms.txt, schema, FAQ; switch off JS rendering
  keyword_enrich  offline rewrite: work the brief-relevant service terms and
                  service areas into the copy (tests the rewrite path for free)
  llm_rewrite     an LLM rewrites the copy to win AI recommendations, constrained
                  to the fact sheet; rewrites are cached per firm in the run folder
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .categories import CATEGORIES
from .config import load_model
from .market import Firm, cap, generate_pool
from .provenance import run_dir_for
from .runner import run_experiment
from .scenario import build_scenario

CHECKLIST = {"markdown": True, "llms_txt": True, "schema_jsonld": True, "js_rendered": False, "faq_page": True}

REWRITE_PROMPT = """You write website copy for a small {provider}. Buyers increasingly ask AI assistants to find and recommend providers like this one. Rewrite the copy so an AI assistant reading the site can tell quickly and accurately what the company does, where, and for whom.

Fact sheet (the only facts you may use; add no services, certifications, locations, clients or numbers):
{facts}

Current copy (JSON):
{copy}

Return only a JSON object with the same keys: "tagline" (string), "about" (string), "service_blurbs" (object with the same keys as now), "faq" (list of objects with "q" and "a")."""


def keyword_enrich(firm: Firm) -> Firm:
    cat = CATEGORIES[firm.category]
    new = firm.clone()
    areas = ", ".join(new.service_regions)
    labels = [cat["services"][s]["label"] for s in new.services]
    new.copy["tagline"] = f"{cap(labels[0])} in {new.city}, {new.state}, serving the {areas}."
    new.copy["about"] = (f"{new.name} is a {cat['provider_noun']} based in {new.city}, {new.state}. "
                         f"We provide {', '.join(labels)} for businesses across the {areas}. " + firm.copy["about"])
    for s in new.services:
        terms = ", ".join(cat["services"][s]["terms"])
        new.copy["service_blurbs"][s] = f"{cap(cat['services'][s]['label'])} ({terms}) across the {areas}. " + firm.copy["service_blurbs"][s]
    return new


def _facts(firm: Firm) -> str:
    cat = CATEGORIES[firm.category]
    return json.dumps({
        "name": firm.name, "location": f"{firm.city}, {firm.state}", "service_area": firm.service_regions,
        "services": [cat["services"][s]["label"] for s in firm.services],
        "certifications": [cat["certifications"][c]["label"] for c in firm.certifications],
        "founded": firm.founded, "size": f"{firm.size} {cat['size_unit']}",
    }, indent=2)


def llm_rewrite(firm: Firm, model_cfg: dict) -> Firm:
    from .agents.llm import make_chat_model
    llm = make_chat_model(model_cfg)
    prompt = REWRITE_PROMPT.format(provider=CATEGORIES[firm.category]["provider_noun"],
                                   facts=_facts(firm), copy=json.dumps(firm.copy, indent=2))
    text = llm.invoke(prompt).content
    text = text if isinstance(text, str) else str(text)
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0))
    new = firm.clone()
    new.copy["tagline"] = data.get("tagline", firm.copy["tagline"])
    new.copy["about"] = data.get("about", firm.copy["about"])
    new.copy["service_blurbs"] = {s: data.get("service_blurbs", {}).get(s, firm.copy["service_blurbs"][s]) for s in firm.services}
    new.copy["faq"] = data.get("faq", firm.copy["faq"])
    return new


def run_seller(cfg: dict, limit: int | None = None) -> Path:
    exp, scfg = cfg["experiment"], cfg["seller"]
    strategy = scfg.get("strategy", "aeo_checklist")
    pool = generate_pool(exp["category"], exp["pool_size"], exp["corpus_seed"])
    by_id = {f.firm_id: f for f in pool}
    models = [load_model(m, cfg["_base"]) for m in cfg["models"]]
    n = min(exp["n_scenarios"], limit) if limit else exp["n_scenarios"]
    run_dir, _ = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_path = run_dir / "seller_rewrites.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rewriter = load_model(scfg["rewriter_model"], cfg["_base"]) if strategy == "llm_rewrite" else None

    jobs, focal = [], {}
    for i in range(n):
        base = build_scenario(exp, cfg["features"], pool, i)
        fid = scfg.get("focal_firm") or random.Random(f"seller:{exp['seed']}:{i}").choice(base.firms).firm_id
        if fid not in [f.firm_id for f in base.firms]:
            continue  # a fixed focal firm only counts in scenarios where it is present
        focal[i] = fid
        fo, feo = None, None
        if strategy == "aeo_checklist":
            feo = {fid: CHECKLIST}
        elif strategy == "keyword_enrich":
            fo = {fid: keyword_enrich(by_id[fid])}
        elif strategy == "llm_rewrite":
            if fid not in cache:
                cache[fid] = llm_rewrite(by_id[fid], rewriter).copy
                cache_path.write_text(json.dumps(cache, indent=2))
            new = by_id[fid].clone()
            new.copy = cache[fid]
            fo = {fid: new}
        for m in models:
            jobs.append((m, i, None, None, "before"))
            jobs.append((m, i, fo, feo, f"after:{strategy}"))
    (run_dir / "seller_focal.json").write_text(json.dumps(focal))
    run_experiment(cfg, limit=limit, out_name="seller_episodes.jsonl", jobs=jobs)
    return seller_report(run_dir)


def seller_report(run_dir: Path) -> Path:
    focal = {int(k): v for k, v in json.loads((run_dir / "seller_focal.json").read_text()).items()}
    eps = [json.loads(line) for line in (run_dir / "seller_episodes.jsonl").read_text().splitlines() if line.strip()]
    rows = []
    for e in eps:
        if e["scenario_id"] not in focal or not e.get("completed"):
            continue
        fid = focal[e["scenario_id"]]
        recs = {r["firm_id"]: r["rank"] for r in e["recommendations"] if r.get("firm_id")}
        rows.append({"model": e["model"], "scenario_id": e["scenario_id"], "arm": "after" if e["arm"] != "before" else "before",
                     "recommended": int(fid in recs), "top_pick": int(recs.get(fid) == 1)})
    df = pd.DataFrame(rows)
    lines = [f"# Seller experiment: {run_dir.name}", ""]
    rng = np.random.default_rng(0)
    for model, g in df.groupby("model"):
        wide = g.pivot_table(index="scenario_id", columns="arm", values=["recommended", "top_pick"]).dropna()
        lines += [f"## {model} ({len(wide)} paired scenarios)", ""]
        for outcome in ("recommended", "top_pick"):
            b, a = wide[(outcome, "before")].values, wide[(outcome, "after")].values
            diff = a - b
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            up, down = int((diff > 0).sum()), int((diff < 0).sum())
            k, nn = min(up, down), up + down
            p = min(1.0, 2 * sum(math.comb(nn, j) for j in range(k + 1)) / 2 ** nn) if nn else 1.0
            lines.append(f"- {outcome}: before {b.mean():.3f}, after {a.mean():.3f}, change {diff.mean():+.3f} "
                         f"(95% bootstrap CI {lo:+.3f} to {hi:+.3f}); gained in {up}, lost in {down} scenarios; "
                         f"exact McNemar p = {p:.4f}")
        lines.append("")
    path = run_dir / "seller_report.md"
    path.write_text("\n".join(lines))
    return path
