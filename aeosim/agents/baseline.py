"""Offline keyword agent: a free, deterministic stand-in for an LLM.

It searches with the brief's service and location, reads the top results
and ranks firms by how many of the brief's key terms appear on the pages
it read. Use it to test the pipeline, size experiments and sanity-check
the analysis before spending money on API calls. It is also a useful
"keyword-matching" reference point to compare LLM behavior against.
"""

from __future__ import annotations

from ..categories import CATEGORIES
from .base import Episode, candidate_domains


def run_baseline(ep: Episode) -> tuple[dict, int, None, None]:
    b = ep.scenario.brief
    cat = CATEGORIES[b.category]
    svc = cat["services"][b.service]
    keywords = [t.lower() for t in svc["terms"]] + [b.state.lower(), b.region.lower()]
    queries = [f"{svc['label']} {b.city} {b.state}"]
    if b.required_cert:
        cert = cat["certifications"][b.required_cert]
        keywords += [t.lower() for t in cert["terms"]]
        queries.append(f"{cat['provider_noun']} {cert['label']} {b.state}")

    if ep.exposure is not None:  # forced exposure: the list arrives with the question
        results = list(ep.exposure)
        queries = []
    else:
        results = []
        for q in queries[: ep.limits["max_searches"]]:
            ep.search(q)
            results += ep.searches[-1]["results"]

    scores: dict[str, float] = {}
    order: list[str] = []
    # First impression from titles and snippets (all the agent has in snippet-only mode).
    for r in results:
        if r["firm_id"] and r["kind"] == "firm_page":
            s = 0.5 * sum((r["title"] + " " + r["snippet"]).lower().count(k) > 0 for k in keywords)
            if r["firm_id"] not in scores:
                order.append(r["firm_id"])
            scores[r["firm_id"]] = max(scores.get(r["firm_id"], 0), s)
    for url in candidate_domains(results)[: ep.limits["max_fetches"]]:
        text = ep.fetch(url).lower()
        fid = ep.web.firm_for_url(url)
        if fid:
            s = sum(text.count(k) > 0 for k in keywords)
            if fid not in scores:
                order.append(fid)
            scores[fid] = max(scores.get(fid, 0), s)
        else:  # directory page: score each listed firm by its own line
            for line in text.splitlines():
                for f in ep.scenario.firms:
                    if f.name.lower() in line:
                        s = 0.5 * sum(line.count(k) > 0 for k in keywords)
                        if f.firm_id not in scores:
                            order.append(f.firm_id)
                        scores[f.firm_id] = max(scores.get(f.firm_id, 0), s)

    ranked = sorted((fid for fid in order if scores[fid] > 0), key=lambda fid: (-scores[fid], order.index(fid)))
    firms = {f.firm_id: f for f in ep.scenario.firms}
    ep.recommend([{"company_name": firms[fid].name, "website": f"https://{firms[fid].domain}/",
                   "reason": f"matched {scores[fid]:g} brief term{'' if scores[fid] == 1 else 's'}"} for fid in ranked[: ep.k]])
    return {"input_tokens": 0, "output_tokens": 0}, len(queries) + len(order), None, None
