"""The episode: the tools a buyer agent can call, with limits and logging.

Every agent (LLM or offline baseline) acts only through Episode.search,
Episode.fetch and Episode.recommend, so every run produces the same log
format no matter which agent made the choices.

Agent settings that shape an episode:
  mode          "search": the agent searches the mock web itself (default)
                "forced_exposure": the agent gets the same list of every firm in
                the market, in random order, and searches nothing. Every firm is
                seen, so differences in recommendation come from the model's
                reading of the firms alone.
  max_searches  search cap (default 5)
  max_fetches   page-read cap; 0 is snippet-only mode, where the agent judges
                firms from titles and snippets, like ChatGPT's instant answers
"""

from __future__ import annotations

import random
import re
import time

from ..scenario import Scenario
from ..search import SearchEngine
from ..web import domain_of, normalize

TOOL_LINES = {
    "web_search": "- web_search: search the web. Returns titles, URLs and snippets.",
    "fetch_page": "- fetch_page: read a web page by URL.",
    "recommend": "- recommend: submit your final answer.",
}


def build_system_prompt(tools: list[str], k: int, max_searches: int, max_fetches: int, forced: bool) -> str:
    lines = ["You are a research assistant that helps business buyers find service providers.", "",
             f"You have {len(tools)} tool{'s' if len(tools) > 1 else ''}:"] + [TOOL_LINES[t] for t in tools] + [""]
    find = ("The user's message includes web search results for their request." if forced
            else "Search for candidates.")
    read = (" Read the most promising company websites." if "fetch_page" in tools
            else " Judge the candidates from the titles and snippets in the search results.")
    lines.append(f"{find}{read} Then call recommend with up to {k} companies ranked best first. "
                 "Give each company's website URL and a one-sentence reason grounded in the information you found.")
    limits = []
    if "web_search" in tools:
        limits.append(f"at most {max_searches} searches")
    if "fetch_page" in tools:
        limits.append(f"at most {max_fetches} page reads")
    lines += ["", ("Limits: " + " and ".join(limits) + ". " if limits else "") + "Call recommend exactly once, at the end."]
    return "\n".join(lines)


def format_results(results: list[dict]) -> str:
    return "\n".join(f"{r['rank']}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for r in results)


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
        self.mode = agent_cfg.get("mode", "search")
        self.exposure: list[dict] | None = None
        if self.mode == "forced_exposure":
            self.exposure = self._exposure_list(random.Random(f"exposure:{scenario.seed}"))
            self.searches.append({"query": "(forced exposure)", "results": [self._log_result(r) for r in self.exposure]})

    # ---- setup -------------------------------------------------------------
    def _exposure_list(self, rng: random.Random) -> list[dict]:
        """Every firm's homepage as a search result, in random order."""
        homes = {d["firm_id"]: d for d in self.web.index_docs() if d["kind"] == "firm_page" and d["url"].endswith("/")}
        firms = [f.firm_id for f in self.scenario.firms]
        rng.shuffle(firms)
        return [{"rank": i + 1, "title": homes[fid]["title"], "url": homes[fid]["url"], "snippet": homes[fid]["snippet"],
                 "firm_id": fid, "kind": "firm_page"} for i, fid in enumerate(firms)]

    @staticmethod
    def _log_result(r: dict) -> dict:
        return {k: r[k] for k in ("rank", "url", "firm_id", "kind", "title", "snippet")}

    @property
    def tool_names(self) -> list[str]:
        tools = []
        if self.mode == "search" and self.limits["max_searches"] > 0:
            tools.append("web_search")
        if self.limits["max_fetches"] > 0:
            tools.append("fetch_page")
        return tools + ["recommend"]

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(self.tool_names, self.k, self.limits["max_searches"],
                                   self.limits["max_fetches"], self.mode == "forced_exposure")

    @property
    def first_message(self) -> str:
        """The buyer's message, plus the result list in forced-exposure mode."""
        brief = self.scenario.brief.text
        if self.exposure is None:
            return brief
        return f"{brief}\n\nSearch results for this request:\n\n{format_results(self.exposure)}"

    # ---- tools -------------------------------------------------------------
    def search(self, query: str) -> str:
        if len(self.searches) >= self.limits["max_searches"]:
            return "Search limit reached. Read pages from earlier results or call recommend."
        n = self.engine.cfg.get("max_query_words")
        sent = " ".join(query.split()[:n]) if n else query
        results = self.engine.search(sent)
        self.searches.append({"query": query, "query_sent": sent, "results": [self._log_result(r) for r in results]})
        if not results:
            return "No results."
        return format_results(results)

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
            "provider": model.get("provider"), "model_id": model.get("model"),
            "model_version": usage.get("model_version"), "temperature": model.get("temperature"),
            "brief": s.brief.to_dict(), "market": s.market_record(),
            "overrides": s.overrides, "mode": self.mode, "tools": self.tool_names, "searches": self.searches, "fetches": self.fetches,
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
