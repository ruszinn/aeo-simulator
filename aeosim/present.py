"""Plain-language views of a run, for demos and for readers who skip the statistics.

  replay   one episode as a story: the request, the market, each step the
           agent took, the final shortlist, and which good firms it missed
  summary  each model and profile's results in plain English: how often firms
           were recommended with each website feature on versus off
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis import build_panel, load_episodes
from .site import FEATURES

LABELS = {
    "markdown": "markdown copy",
    "llms_txt": "llms.txt file",
    "schema_jsonld": "structured data",
    "js_rendered": "JavaScript-only site",
    "faq_page": "FAQ page",
    "directory_listing": "directory listing",
    "descriptive_heading": "descriptive heading",
}
RULE = "-" * 72


def resolve_run(path: str | Path, file: str = "episodes.jsonl") -> Path:
    """Accept a run folder (runs/pilot/<commit>) or an experiment folder
    (runs/pilot), in which case the most recent run inside it is used."""
    p = Path(path)
    if (p / file).exists():
        return p
    runs = sorted((d for d in p.glob("*") if (d / file).exists()), key=lambda d: (d / file).stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No {file} found in {p} or its subfolders. Run an experiment first.")
    return runs[-1]


def _label(ep: dict) -> str:
    return ep["model"] + (f" · {ep['profile']}" if ep.get("profile") else "")


def pick_episode(eps: list[dict], scenario: int | None, model: str | None, profile: str | None) -> dict:
    def ok(e):
        return (e.get("completed") and (scenario is None or e["scenario_id"] == scenario)
                and (model is None or model.lower() in e["model"].lower())
                and (profile is None or e.get("profile") == profile))
    found = [e for e in eps if ok(e)]
    if not found:
        raise ValueError("No completed episode matches those filters.")
    real = [e for e in found if e.get("provider") not in ("baseline", "scripted")]
    return (real or found)[0]


def _steps(ep: dict) -> list[tuple[str, dict]]:
    """Searches and page reads in the order the agent made them."""
    searches, fetches = list(ep["searches"]), list(ep["fetches"])
    steps = []
    if ep.get("mode") == "forced_exposure" and searches:
        steps.append(("shown", searches.pop(0)))
    if ep.get("transcript"):
        for m in ep["transcript"]:
            if m["role"] == "ai" and m.get("content", "").strip():
                steps.append(("thought", {"text": m["content"].strip()}))
            for tc in m.get("tool_calls", []):
                if tc["name"] == "web_search" and searches:
                    steps.append(("search", searches.pop(0)))
                elif tc["name"] == "fetch_page" and fetches:
                    steps.append(("fetch", fetches.pop(0)))
    steps += [("search", s) for s in searches] + [("fetch", f) for f in fetches]
    return steps


def replay(ep: dict, width: int = 72) -> str:
    firms = {m["firm_id"]: m for m in ep["market"]}
    fits = {fid for fid, m in firms.items() if m["fit"]["full_fit"]}
    out = [RULE, f"EPISODE  scenario {ep['scenario_id']}  ·  {_label(ep)}", RULE, "",
           "THE BUYER ASKED", f'  "{ep["brief"]["text"]}"', "",
           f"THE MARKET  {len(firms)} firms; [fits] marks the {len(fits)} that truly match the request"]
    for m in ep["market"]:
        on = [LABELS[k] for k, v in m["features"].items() if v]
        tag = "[fits]" if m["firm_id"] in fits else "      "
        out.append(f"  {tag} {m['name']:<28} {', '.join(on)}")
    out.append("")

    seen = set()
    for n, (kind, s) in enumerate(_steps(ep), 1):
        if kind == "thought":
            out += [f"  (the agent noted: {s['text'][:300]})", ""]
            continue
        if kind in ("search", "shown"):
            head = ("WAS SHOWN a list of every firm, in random order" if kind == "shown"
                    else f'SEARCHED "{s["query"]}"' + (f'  (sent as "{s["query_sent"]}")'
                                                          if s.get("query_sent") not in (None, s["query"]) else ""))
            out.append(f"STEP {n}  {head}")
            for r in s["results"][:10]:
                who = firms.get(r.get("firm_id") or "", {}).get("name") or "third-party page"
                seen.add(r.get("firm_id"))
                snip = r.get("snippet", "")[:44] or "(empty: nothing readable before scripts run)"
                out.append(f"    {r['rank']:>2}. {who:<28} {snip}")
        else:
            who = firms.get(s.get("firm_id") or "", {}).get("name") or "third-party page"
            how = {"markdown_alternate": "the site's own markdown copy", "llms_txt": "its llms.txt file"}.get(
                s.get("served"), "the page, converted to text")
            out.append(f"STEP {n}  OPENED {s['url'].split('//')[-1]}  ({who}; read {how})")
            seen.update(s.get("listed_firms", []))
        out.append("")

    out.append("FINAL SHORTLIST")
    for r in ep["recommendations"]:
        fid = r.get("firm_id")
        verdict = "fits the request" if fid in fits else "does NOT fully fit the request"
        out.append(f"  {r['rank']}. {r['name']}  ({verdict})")
        if r.get("reason"):
            out.append(f"     reason given: {r['reason'][:200]}")
    picked = {r.get("firm_id") for r in ep["recommendations"]}
    missed = [firms[f]["name"] for f in fits if f not in picked]
    if missed:
        out += ["", "GOOD FIRMS LEFT OUT"]
        for f in fits - picked:
            why = ("never appeared in anything the agent saw" if f not in seen
                   else "seen in results or on a page the agent read, then passed over")
            on = [LABELS[k] for k, v in firms[f]["features"].items() if v]
            out.append(f"  {firms[f]['name']}: {why}  (features: {', '.join(on) or 'none'})")
    u = ep.get("usage") or {}
    out += ["", f"Tokens used: {u.get('input_tokens', 0):,} in, {u.get('output_tokens', 0):,} out"
            + (f"  ·  model version {u['model_version']}" if u.get("model_version") else ""), RULE]
    return "\n".join(out)


def summary(run_dir: Path, file: str = "episodes.jsonl") -> str:
    eps = load_episodes(run_dir / file)
    panel = build_panel(eps)
    out = [RULE, f"RESULTS  {run_dir.parent.name}  ·  {len(eps)} episodes  ·  code version {run_dir.name}", RULE]
    for label, g in panel.groupby("model"):
        eps_m = [e for e in eps if _label(e) == label]
        recs = g[g.recommended == 1]
        out += ["", f"{label}  ({g.episode.nunique()} episodes)",
                f"  Shortlisted firms that truly fit the request: {recs.full_fit.mean():.0%}",
                "  How often a firm made the shortlist, with each feature on vs off:"]
        rows = []
        for f in FEATURES:
            on, off = g[g[f] == 1].recommended.mean(), g[g[f] == 0].recommended.mean()
            rows.append((on - off, f, on, off))
        for diff, f, on, off in sorted(rows, key=lambda r: r[0]):
            arrow = "up" if diff > 0 else "down"
            pts = round(abs(diff) * 100)
            out.append(f"    {LABELS[f]:<22} {on:>4.0%} vs {off:>4.0%}   ({arrow} {pts} point{'s' if pts != 1 else ''})")
        n = max(len(eps_m), 1)
        tin = sum((e.get("usage") or {}).get("input_tokens", 0) for e in eps_m)
        out.append(f"  Per episode: {sum(len(e['searches']) for e in eps_m) / n:.1f} searches, "
                   f"{sum(len(e['fetches']) for e in eps_m) / n:.1f} pages read, {tin / n:,.0f} input tokens")
    out += ["", "These are raw comparisons. The statistical analysis adds controls for",
            "brief fit and each firm's baseline (python -m aeosim analyze).", RULE]
    return "\n".join(out)
