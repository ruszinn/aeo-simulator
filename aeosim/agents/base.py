"""The episode: the three tools a buyer agent can call, with limits and logging.

Every agent (LLM or offline baseline) acts only through Episode.search,
Episode.fetch and Episode.recommend, so every run produces the same log
format no matter which agent made the choices.
"""

from __future__ import annotations

import random
import re
import time

from ..scenario import Scenario
from ..search import SearchEngine
from ..web import domain_of, normalize

SYSTEM_PROMPT = """You are a research assistant that helps business buyers find service providers.

You have three tools:
- web_search: search the web. Returns titles, URLs and snippets.
- fetch_page: read a web page by URL.
- recommend: submit your final answer.

Search for candidates, read the most promising company websites, then call recommend with up to {k} companies ranked best first. Give each company's website URL and a one-sentence reason grounded in what you read.

Limits: at most {max_searches} searches and {max_fetches} page reads. Call recommend exactly once, at the end."""


class Episode:
    def __init__(self, scenario: Scenario, web_cfg: dict, search_cfg: dict, agent_cfg: dict, shortlist_size: int):
        self.scenario = scenario
        self.web = scenario.build_web(web_cfg)
        self.engine = SearchEngine(self.web.index_docs(), search_cfg, random.Random(scenario.search_seed))
        self.limits = agent_cfg
        self.k = shortlist_size
        self.searches: list[dict] = []
        self.fetches: list[dict] = []
        self.recommendations: list[dict] | None = None
        self.started = time.time()
        self.names = {f.firm_id: f.name for f in scenario.firms}

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(k=self.k, max_searches=self.limits["max_searches"],
                                    max_fetches=self.limits["max_fetches"])

    # ---- tools -------------------------------------------------------------
    def search(self, query: str) -> str:
        if len(self.searches) >= self.limits["max_searches"]:
            return "Search limit reached. Read pages from earlier results or call recommend."
        results = self.engine.search(query)
        self.searches.append({"query": query, "results": [
            {"rank": r["rank"], "url": r["url"], "firm_id": r["firm_id"], "kind": r["kind"]} for r in results]})
        if not results:
            return "No results."
        return "\n".join(f"{r['rank']}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for r in results)

    def fetch(self, url: str) -> str:
        if len(self.fetches) >= self.limits["max_fetches"]:
            return "Page-read limit reached. Call recommend with what you have."
        out = self.web.fetch(url)
        rec = {k: out.get(k) for k in ("url", "status", "firm_id", "served", "chars", "truncated")}
        if out["status"] == 200 and out["firm_id"] is None:  # third-party page: which firms does it name?
            rec["listed_firms"] = [f.firm_id for f in self.scenario.firms if f.name.lower() in out["text"].lower()]
        self.fetches.append(rec)
        return out["text"]

    def recommend(self, recommendations: list[dict]) -> str:
        resolved = []
        for i, rec in enumerate(recommendations[: self.k]):
            url = rec.get("website") or rec.get("url") or ""
            name = rec.get("company_name") or rec.get("name") or ""
            fid = self._resolve(url, name)
            resolved.append({"rank": i + 1, "firm_id": fid, "name": name, "url": url,
                             "reason": rec.get("reason", ""), "resolved": fid is not None})
        self.recommendations = resolved
        return "Recommendation recorded."

    def _resolve(self, url: str, name: str) -> str | None:
        if url:
            fid = self.web.firm_for_url(url)
            if fid:
                return fid
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key:
            for fid, n in self.names.items():
                nk = re.sub(r"[^a-z0-9]", "", n.lower())
                if key == nk or (len(key) > 5 and (key in nk or nk in key)):
                    return fid
        return None

    # ---- record ------------------------------------------------------------
    def record(self, model: dict, usage: dict, steps: int, error: str | None, transcript: list | None) -> dict:
        s = self.scenario
        return {
            "scenario_id": s.scenario_id, "seed": s.seed, "model": model.get("display_name"),
            "provider": model.get("provider"), "brief": s.brief.to_dict(), "market": s.market_record(),
            "overrides": s.overrides, "searches": self.searches, "fetches": self.fetches,
            "recommendations": self.recommendations or [], "completed": self.recommendations is not None,
            "usage": usage, "steps": steps, "error": error, "elapsed_s": round(time.time() - self.started, 2),
            "transcript": transcript,
        }


def candidate_domains(results: list[dict]) -> list[str]:
    seen, out = set(), []
    for r in results:
        d = domain_of(r["url"])
        if d not in seen:
            seen.add(d)
            out.append(normalize(r["url"]))
    return out
