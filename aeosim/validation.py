"""Validation kit: compare the simulator with real AI search.

1. `export_briefs` writes buyer briefs to paste into ChatGPT search,
   Perplexity, Gemini or Claude, plus a sheet for recording what each
   platform recommended.
2. `audit_sites` visits real company websites and detects the same
   kinds of features the simulator toggles (descriptive headings and directory
   listings need a manual check), so you can test whether the
   firms real AI search recommends share the features the simulator says
   should help. The audit records each homepage's H1 text for the heading check.
"""

from __future__ import annotations

import csv
import random
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .market import generate_brief

UA = "Mozilla/5.0 (compatible; aeo-simulator-audit/0.1; research)"


def export_briefs(category: str, n: int, seed: int, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"validation:{seed}")
    with open(out_dir / "validation_briefs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["brief_id", "category", "service", "city", "state", "required_cert", "brief_text"])
        for i in range(n):
            b = generate_brief(category, rng)
            w.writerow([i, category, b.service, b.city, b.state, b.required_cert or "", b.text])
    with open(out_dir / "validation_results_template.csv", "w", newline="") as f:
        csv.writer(f).writerow(["brief_id", "platform", "date", "rank", "company_name", "website"])
    return out_dir / "validation_briefs.csv"


def _get(url: str, timeout: int = 15) -> tuple[int, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read(2_000_000).decode("utf-8", "replace")
    except Exception as e:
        return getattr(e, "code", 0) or 0, "", ""


def detect_features(home_html: str, llms_status: int, llms_body: str, md_status: int, md_body: str) -> dict:
    soup = BeautifulSoup(home_html, "html.parser")
    h1 = soup.find("h1")
    scripts = soup.find_all("script")
    jsonld = any(s.get("type") == "application/ld+json" for s in scripts)
    for s in scripts:
        s.decompose()
    visible = soup.get_text(" ", strip=True) if soup.body else ""
    md_link = soup.find("link", attrs={"type": "text/markdown"}) is not None
    faq = any(re.search(r"faq|frequently", (a.get("href", "") + " " + a.get_text()).lower()) for a in soup.find_all("a"))
    looks_html = lambda body: "<html" in body[:500].lower()  # many sites answer 200 with their HTML 404 page
    return {
        "markdown": md_link or (md_status == 200 and not looks_html(md_body) and len(md_body) > 50),
        "llms_txt": llms_status == 200 and not looks_html(llms_body) and len(llms_body) > 20,
        "schema_jsonld": jsonld,
        "js_rendered": len(visible) < 300 and len(scripts) > 0,
        "faq_page": faq,
        "visible_chars": len(visible),
        "h1_text": h1.get_text(" ", strip=True)[:120] if h1 else "",
    }


def audit_sites(urls: list[str], out_csv: str | Path) -> Path:
    rows = []
    for url in urls:
        url = url if url.startswith("http") else "https://" + url
        status, _, home = _get(url)
        ls, _, lb = _get(urljoin(url, "/llms.txt"))
        ms, _, mb = _get(urljoin(url, "/index.md"))
        row = {"url": url, "home_status": status}
        row.update(detect_features(home, ls, lb, ms, mb) if home else {})
        rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("url", "home_status"), k))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return Path(out_csv)
