"""Turn episode logs into a firm-level panel, a funnel table and choice models.

Panel: one row per (episode, firm in that episode's market) with
  surfaced          any of the firm's pages appeared in a search result
  best_rank         its best result position (11 if it never appeared)
  seen              surfaced, or named on a third-party page the agent read
  fetched           the agent read one of the firm's pages
  recommended       the firm made the shortlist
  top_pick          the firm was ranked first
plus the six feature switches and the brief-fit controls.

Models: conditional logit, grouped by episode, so each firm is compared
only with the other firms in the same scenario. Two stages:
  retrieval  outcome = surfaced, all firms
  selection  outcome = recommended, firms the agent saw
Firm fixed effects (one dummy per firm) are on by default.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .site import FEATURES

FIT_VARS = ["service_match", "region_match", "cert_match"]


def load_episodes(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_panel(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for ep in episodes:
        if not ep.get("completed"):
            continue
        ranks: dict[str, int] = {}
        for s in ep["searches"]:
            for r in s["results"]:
                if r["firm_id"]:
                    ranks[r["firm_id"]] = min(ranks.get(r["firm_id"], 99), r["rank"])
        fetched = {f["firm_id"] for f in ep["fetches"] if f.get("firm_id")}
        via_third_party = {fid for f in ep["fetches"] for fid in f.get("listed_firms", [])}
        recs = {r["firm_id"]: r["rank"] for r in ep["recommendations"] if r.get("firm_id")}
        key = f"{ep['model']}|{ep.get('arm')}|{ep['scenario_id']}"
        for m in ep["market"]:
            fid = m["firm_id"]
            row = {
                "episode": key, "model": ep["model"], "arm": ep.get("arm"), "scenario_id": ep["scenario_id"],
                "firm_id": fid, "style": m["style"],
                "surfaced": int(fid in ranks), "best_rank": ranks.get(fid, 11),
                "seen": int(fid in ranks or fid in via_third_party), "fetched": int(fid in fetched),
                "recommended": int(fid in recs), "rec_rank": recs.get(fid), "top_pick": int(recs.get(fid) == 1),
            }
            row.update({k: int(v) for k, v in m["features"].items()})
            row.update({k: m["fit"][k] for k in FIT_VARS + ["full_fit"]})
            rows.append(row)
    return pd.DataFrame(rows)


def funnel(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model, g in panel.groupby("model"):
        for feat in FEATURES:
            for val in (0, 1):
                s = g[g[feat] == val]
                seen = s[s.seen == 1]
                out.append({
                    "model": model, "feature": feat, "value": val, "n_firm_episodes": len(s),
                    "surfaced": s.surfaced.mean(), "seen": s.seen.mean(), "fetched": s.fetched.mean(),
                    "recommended": s.recommended.mean(),
                    "recommended_given_seen": seen.recommended.mean() if len(seen) else np.nan,
                })
    return pd.DataFrame(out)


def _design(df: pd.DataFrame, firm_fe: bool, extra: list[str]) -> pd.DataFrame:
    X = df[FEATURES + FIT_VARS + extra].astype(float)
    if firm_fe:
        X = pd.concat([X, pd.get_dummies(df["firm_id"], prefix="firm", drop_first=True, dtype=float)], axis=1)
    else:
        X = pd.concat([X, pd.get_dummies(df["style"], prefix="style", drop_first=True, dtype=float)], axis=1)
    return X.loc[:, X.std() > 0]


def conditional_logit(df: pd.DataFrame, outcome: str, firm_fe: bool = True, extra: list[str] | None = None) -> pd.DataFrame:
    from statsmodels.discrete.conditional_models import ConditionalLogit

    extra = extra or []
    # Groups where every firm has the same outcome carry no information.
    var = df.groupby("episode")[outcome].transform(lambda s: s.nunique() > 1)
    d = df[var].reset_index(drop=True)
    if d.empty:
        raise ValueError(f"No episode has variation in {outcome}.")
    X = _design(d, firm_fe, extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ConditionalLogit(d[outcome].astype(float), X, groups=d["episode"]).fit(disp=0, maxiter=200)
    tab = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues})
    tab["odds_ratio"] = np.exp(tab["coef"])
    tab.attrs.update(n_obs=len(d), n_groups=d["episode"].nunique())
    return tab.loc[[c for c in tab.index if not c.startswith("firm_")]]


def analyze(run_dir: str | Path, episodes_file: str = "episodes.jsonl", firm_fe: bool = True) -> Path:
    run_dir = Path(run_dir)
    eps = load_episodes(run_dir / episodes_file)
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    shuffled = cfg.get("search", {}).get("mode") == "shuffled"
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    panel = build_panel(eps)
    panel.to_csv(out / "panel.csv", index=False)
    fun = funnel(panel)
    fun.to_csv(out / "funnel.csv", index=False)

    lines = [f"# Analysis of {run_dir.name}", "",
             f"Episodes: {len(eps)} logged, {sum(e.get('completed', False) for e in eps)} completed.", ""]
    for model, g in panel.groupby("model"):
        eps_m = [e for e in eps if e["model"] == model]
        tok_in = sum(e.get("usage", {}).get("input_tokens", 0) for e in eps_m)
        tok_out = sum(e.get("usage", {}).get("output_tokens", 0) for e in eps_m)
        fit_share = g[g.recommended == 1].full_fit.mean()
        lines += [f"## {model}", "",
                  f"- Episodes: {g.episode.nunique()}; tokens in/out: {tok_in:,}/{tok_out:,}",
                  f"- Share of recommendations that fully fit the brief: {fit_share:.2f}",
                  f"- Mean searches per episode: {np.mean([len(e['searches']) for e in eps_m]):.2f}; "
                  f"mean page reads: {np.mean([len(e['fetches']) for e in eps_m]):.2f}", ""]
        f = fun[fun.model == model].pivot(index="feature", columns="value",
                                          values=["surfaced", "recommended_given_seen", "recommended"])
        lines += ["Funnel by feature (off -> on):", "", "```", f.round(3).to_string(), "```", ""]
        stages = [("Retrieval (outcome: surfaced)", g, "surfaced", []),
                  ("Selection (outcome: recommended, among firms seen)", g[g.seen == 1], "recommended",
                   ["best_rank"] if shuffled else [])]
        for title, data, outcome, extra in stages:
            try:
                tab = conditional_logit(data, outcome, firm_fe=firm_fe, extra=extra)
                tab.to_csv(out / f"clogit_{outcome}_{model.replace(' ', '_')}.csv")
                lines += [f"{title}; n={tab.attrs['n_obs']}, groups={tab.attrs['n_groups']}, "
                          f"firm fixed effects={'on' if firm_fe else 'off'}:", "", "```",
                          tab.round(3).to_string(), "```", ""]
            except Exception as e:  # small pilots often lack variation; report and continue
                lines += [f"{title}: model could not be fit ({type(e).__name__}: {e}).", ""]
    (out / "report.md").write_text("\n".join(lines))
    return out / "report.md"
