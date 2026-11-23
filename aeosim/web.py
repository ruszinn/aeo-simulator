"""The mock web: an in-memory set of pages plus two ways to read them.

The crawler view decides what the search engine indexes. The fetcher view
decides what the buyer agent reads when it opens a page. Both are set by the
`web` block of the experiment config, because real AI products differ here:

  crawler_renders_js     does the indexer run JavaScript before indexing?
  crawler_reads_llms_txt does the indexer pick up /llms.txt files?
  crawler_reads_jsonld   does the indexer read schema.org structured data?
  snippet_source         "h1": the result snippet starts at the page's main heading
                         (measured for ChatGPT); "meta": the meta description
  fetcher                "markdown_convert": convert the page's HTML to Markdown,
                         dropping scripts, frames and structured data (measured
                         for ChatGPT's cache; the default)
                         "html_text": the page's visible text, one line per element
                         "prefer_markdown": the site's own /index.md when it exists
  fetcher_main_content   also drop menus, headers, footers and cookie banners,
                         the way a browser's reader view does (unmeasured; a
                         robustness setting)
  fetcher_renders_js     does the agent's fetch tool run JavaScript?
  fetcher_keeps_jsonld   does the fetch tool keep structured data in its output?
  max_fetch_chars        pages longer than this are cut off

The defaults follow the published measurements of ChatGPT's pipeline.
Platform profiles (profiles/*.yaml) override them for other products.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify

from .site import Resource

DEFAULT_WEB = {
    "crawler_renders_js": False,
    "crawler_reads_llms_txt": False,
    "crawler_reads_jsonld": True,
    "snippet_source": "h1",
    "snippet_chars": 200,
    "fetcher": "markdown_convert",
    "fetcher_main_content": False,
    "fetcher_renders_js": False,
    "fetcher_keeps_jsonld": False,
    "max_fetch_chars": 6000,
}

BOILERPLATE_TAGS = ["nav", "header", "footer", "aside"]


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


def _structured(soup: BeautifulSoup) -> list[str]:
    out = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            out.append(json.dumps(json.loads(tag.string or "{}"), separators=(", ", ": ")))
        except json.JSONDecodeError:
            pass
    return out


def html_to_markdown(raw: str, keep_jsonld: bool, main_content: bool) -> tuple[str, str]:
    """Convert HTML to Markdown the way an AI fetch pipeline does: parse the
    tag tree, prune scripts, styles and frames, then write each remaining tag
    in its Markdown form (h1 -> #, h2 -> ##, links, image alt text)."""
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    structured = _structured(soup) if keep_jsonld else []
    for tag in soup(["script", "style", "iframe", "svg", "template"]):
        tag.decompose()
    if main_content:
        for tag in soup(BOILERPLATE_TAGS):
            tag.decompose()
        for tag in soup.find_all(class_=re.compile("cookie|banner|consent", re.I)):
            tag.decompose()
    body = soup.body if soup.body else soup
    md = markdownify(str(body), heading_style="ATX", bullets="-")
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if structured:
        md += "\n\nStructured data (JSON-LD):\n" + "\n".join(structured)
    return title, md


def make_snippet(raw: str, source: str, chars: int) -> str:
    """The short text a results list shows for a page. 'h1' starts at the main
    heading and runs on through the page text; 'meta' uses the meta description."""
    soup = BeautifulSoup(raw, "html.parser")
    if source == "meta":
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"][:chars]
    for tag in soup(["script", "style", "iframe", "noscript", "template"]):
        tag.decompose()
    h1 = soup.find("h1")
    if source == "h1" and h1:
        parts = [h1.get_text(" ", strip=True)] + [t.strip() for t in h1.find_all_next(string=True) if t.strip()]
        text = " ".join(dict.fromkeys(parts))
    else:
        body = soup.body if soup.body else soup
        text = body.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:chars].rstrip() + ("..." if len(text) > chars else "")


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
                snippet = make_snippet(raw, self.cfg["snippet_source"], self.cfg["snippet_chars"])
            else:
                title, text = r.title, r.initial
                snippet = re.sub(r"\s+", " ", text)[: self.cfg["snippet_chars"]]
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
            if self.cfg["fetcher"] == "html_text":
                title, body = html_to_text(raw, keep_jsonld=self.cfg["fetcher_keeps_jsonld"])
            else:
                title, body = html_to_markdown(raw, keep_jsonld=self.cfg["fetcher_keeps_jsonld"],
                                               main_content=self.cfg["fetcher_main_content"])
            text = f"Title: {title}\n\n{body}"
        else:
            text = r.initial
        limit = self.cfg["max_fetch_chars"]
        truncated = len(text) > limit
        if truncated:
            text = text[:limit] + "\n[... page truncated ...]"
        return {"url": r.url, "status": 200, "text": text, "served": served, "firm_id": r.firm_id,
                "chars": len(text), "truncated": truncated}
