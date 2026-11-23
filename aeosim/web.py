"""The mock web: an in-memory set of pages plus two ways to read them.

The crawler view decides what the search engine indexes. The fetcher view
decides what the buyer agent reads when it opens a page. Both are set by the
`web` block of the experiment config, because real AI products differ here:

  crawler_renders_js     does the indexer run JavaScript before indexing?
  crawler_reads_llms_txt does the indexer pick up /llms.txt files?
  crawler_reads_jsonld   does the indexer read schema.org structured data?
  fetcher                "html_text": read the page's HTML as text
                         "prefer_markdown": use the site's /index.md when it exists
  fetcher_renders_js     does the agent's fetch tool run JavaScript?
  fetcher_keeps_jsonld   does the fetch tool keep structured data in its output?
  max_fetch_chars        pages longer than this are cut off
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .site import Resource

DEFAULT_WEB = {
    "crawler_renders_js": False,
    "crawler_reads_llms_txt": True,
    "crawler_reads_jsonld": True,
    "fetcher": "prefer_markdown",
    "fetcher_renders_js": False,
    "fetcher_keeps_jsonld": False,
    "max_fetch_chars": 6000,
}


def normalize(url: str) -> str:
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    if path in ("/index.html", "/home"):
        path = "/"
    return f"https://{host}{path}"


def domain_of(url: str) -> str:
    return urlparse(normalize(url)).netloc


def html_to_text(raw: str, keep_jsonld: bool) -> tuple[str, str]:
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    structured = []
    for tag in soup.find_all("script"):
        if keep_jsonld and tag.get("type") == "application/ld+json":
            try:
                structured.append(json.dumps(json.loads(tag.string or "{}"), separators=(", ", ": ")))
            except json.JSONDecodeError:
                pass
        tag.decompose()
    for tag in soup.find_all("style"):
        tag.decompose()
    body = soup.body.get_text("\n", strip=True) if soup.body else soup.get_text("\n", strip=True)
    text = body
    if structured:
        text += "\n\nStructured data (JSON-LD):\n" + "\n".join(structured)
    return title, text


class MockWeb:
    def __init__(self, resources: list[Resource], web_cfg: dict | None = None):
        self.cfg = {**DEFAULT_WEB, **(web_cfg or {})}
        self.pages = {normalize(r.url): r for r in resources}
        self.markdown_by_domain = {domain_of(r.url): r for r in resources if r.kind == "markdown"}
        self.domain_to_firm = {domain_of(r.url): r.firm_id for r in resources if r.firm_id}

    def get(self, url: str) -> Resource | None:
        return self.pages.get(normalize(url))

    def firm_for_url(self, url: str) -> str | None:
        return self.domain_to_firm.get(domain_of(url))

    def index_docs(self) -> list[dict]:
        docs = []
        for r in self.pages.values():
            if r.kind == "markdown":
                continue
            if r.kind == "llms_txt" and not self.cfg["crawler_reads_llms_txt"]:
                continue
            if r.content_type == "html":
                raw = r.rendered if self.cfg["crawler_renders_js"] else r.initial
                title, text = html_to_text(raw, keep_jsonld=self.cfg["crawler_reads_jsonld"])
                snippet_src = BeautifulSoup(raw, "html.parser").find("meta", attrs={"name": "description"})
                snippet = snippet_src["content"] if snippet_src else text[:160]
            else:
                title, text = r.title, r.initial
                snippet = text[:160]
            docs.append({"url": r.url, "title": title or r.title, "text": text, "snippet": snippet,
                         "firm_id": r.firm_id, "kind": r.kind})
        return docs

    def fetch(self, url: str) -> dict:
        """What the agent's fetch tool returns. Always a dict with text and metadata."""
        r = self.get(url)
        if r is None:
            return {"url": url, "status": 404, "text": f"Error 404: {url} was not found.", "served": None, "firm_id": None}
        served = r.kind
        if (self.cfg["fetcher"] == "prefer_markdown" and r.kind == "firm_page"
                and domain_of(r.url) in self.markdown_by_domain):
            md = self.markdown_by_domain[domain_of(r.url)]
            text, served = md.initial, "markdown_alternate"
        elif r.content_type == "html":
            raw = r.rendered if self.cfg["fetcher_renders_js"] else r.initial
            title, body = html_to_text(raw, keep_jsonld=self.cfg["fetcher_keeps_jsonld"])
            text = f"Title: {title}\n\n{body}"
        else:
            text = r.initial
        limit = self.cfg["max_fetch_chars"]
        truncated = len(text) > limit
        if truncated:
            text = text[:limit] + "\n[... page truncated ...]"
        return {"url": r.url, "status": 200, "text": text, "served": served, "firm_id": r.firm_id,
                "chars": len(text), "truncated": truncated}
