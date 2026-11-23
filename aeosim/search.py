"""Mock search engine over the mock web's crawler view.

method:
  bm25    keyword ranking: scores pages by matches on the query's words,
          giving rare words more weight
  hybrid  bm25 plus embedding (meaning-based) ranking, merged with
          reciprocal rank fusion; needs `pip install sentence-transformers`
mode:
  ranked    results in score order (use for retrieval experiments)
  shuffled  the same top results in random order (use for selection
            experiments, so position effects separate from content effects)
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

from .web import domain_of

STOP = set("""a an and are as at be by for from has have i in is it its of on or our that the their them they this to
we were what which who with you your can need needs looking near about who should recommend few""".split())

DEFAULT_SEARCH = {"method": "bm25", "mode": "ranked", "results_per_query": 10, "max_per_domain": 2,
                  "k1": 1.5, "b": 0.75, "title_weight": 2, "embedding_model": "all-MiniLM-L6-v2"}


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if t in STOP or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float, b: float):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = [len(d) for d in docs_tokens]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for d in docs_tokens for t in set(d))
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


_EMBEDDERS: dict = {}


def _embedder(name: str):
    if name not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer  # optional dependency
        _EMBEDDERS[name] = SentenceTransformer(name)
    return _EMBEDDERS[name]


class SearchEngine:
    def __init__(self, docs: list[dict], cfg: dict | None, rng: random.Random):
        self.cfg = {**DEFAULT_SEARCH, **(cfg or {})}
        self.docs = docs
        self.rng = rng
        w = self.cfg["title_weight"]
        toks = [tokenize(d["title"]) * w + tokenize(d["text"]) for d in docs]
        self.bm25 = BM25(toks, self.cfg["k1"], self.cfg["b"])
        self._emb = None
        if self.cfg["method"] == "hybrid":
            model = _embedder(self.cfg["embedding_model"])
            self._emb = model.encode([f"{d['title']}\n{d['text'][:2000]}" for d in docs], normalize_embeddings=True)

    def _ranking(self, query: str) -> list[int]:
        s = self.bm25.scores(tokenize(query))
        bm_order = [i for i in sorted(range(len(s)), key=lambda i: -s[i]) if s[i] > 0]
        if self._emb is None:
            return bm_order
        qv = _embedder(self.cfg["embedding_model"]).encode([query], normalize_embeddings=True)[0]
        sims = self._emb @ qv
        emb_order = sorted(range(len(sims)), key=lambda i: -sims[i])
        rrf: dict[int, float] = {}
        for order in (bm_order, emb_order):
            for r, i in enumerate(order):
                rrf[i] = rrf.get(i, 0) + 1 / (60 + r)
        return sorted(rrf, key=lambda i: -rrf[i])

    def search(self, query: str) -> list[dict]:
        picked, per_domain = [], Counter()
        for i in self._ranking(query):
            dom = domain_of(self.docs[i]["url"])
            if per_domain[dom] >= self.cfg["max_per_domain"]:
                continue
            per_domain[dom] += 1
            picked.append(i)
            if len(picked) >= self.cfg["results_per_query"]:
                break
        if self.cfg["mode"] == "shuffled":
            self.rng.shuffle(picked)
        return [{"rank": r + 1, "title": self.docs[i]["title"], "url": self.docs[i]["url"],
                 "snippet": self.docs[i]["snippet"], "firm_id": self.docs[i]["firm_id"], "kind": self.docs[i]["kind"]}
                for r, i in enumerate(picked)]
