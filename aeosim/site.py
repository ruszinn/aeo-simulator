"""Render a firm's website, with each AEO feature as an on/off switch.

Features (all booleans):
  markdown           serve a clean markdown copy of the site at /index.md
  llms_txt           serve /llms.txt, a plain-text summary written for AI crawlers
  schema_jsonld      embed schema.org JSON-LD structured data in each page
  js_rendered        initial HTML is an empty shell; a script fills in the content
  faq_page           add /faq.html restating services, areas and certifications
  directory_listing  list the firm on the third-party directory site
  descriptive_heading  the main heading (H1) names the services and city;
                     off, it carries the firm's name. AI search snippets are
                     anchored on the H1, so this is what a results list shows.

Every page is stored twice: `initial` is the HTML a simple fetcher receives,
`rendered` is the HTML after JavaScript runs. They differ only for js_rendered.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass

from .categories import CATEGORIES
from .market import Firm, cap

FEATURES = ["markdown", "llms_txt", "schema_jsonld", "js_rendered", "faq_page", "directory_listing",
            "descriptive_heading"]

NAV = ["Home", "About Us", "Services", "Industries", "Careers", "Blog", "Contact", "Request a Quote"]
COOKIE = (
    "We use cookies to improve your experience, analyze site traffic and personalize content. "
    "By clicking Accept All you consent to our use of cookies. Manage preferences."
)


@dataclass
class Resource:
    url: str
    content_type: str  # "html", "markdown" or "text"
    title: str
    initial: str
    rendered: str
    firm_id: str | None
    kind: str  # "firm_page", "markdown", "llms_txt", "directory"


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def jsonld(firm: Firm, cat: dict) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "LocalBusiness",
        "name": firm.name,
        "url": f"https://{firm.domain}/",
        "description": firm.copy["tagline"],
        "foundingDate": str(firm.founded),
        "address": {"@type": "PostalAddress", "addressLocality": firm.city, "addressRegion": firm.state, "addressCountry": "US"},
        "areaServed": firm.service_regions,
        "hasOfferCatalog": {
            "@type": "OfferCatalog",
            "name": "Services",
            "itemListElement": [
                {"@type": "Offer", "itemOffered": {"@type": "Service", "name": cat["services"][s]["label"],
                                                   "alternateName": cat["services"][s]["terms"]}}
                for s in firm.services
            ],
        },
        "hasCredential": [cat["certifications"][c]["label"] for c in firm.certifications],
    }
    return json.dumps(data)


def _page(firm: Firm, title: str, body: str, features: dict, cat: dict) -> tuple[str, str]:
    head = f"<title>{_esc(title)}</title><meta name='description' content='{_esc(firm.copy['tagline'])}'>"
    if features.get("schema_jsonld"):
        head += f"<script type='application/ld+json'>{jsonld(firm, cat)}</script>"
    if features.get("markdown"):
        head += "<link rel='alternate' type='text/markdown' href='/index.md'>"
    nav = "<nav>" + " | ".join(f"<a href='#'>{n}</a>" for n in NAV) + "</nav>"
    cookie = f"<div class='cookie'>{COOKIE}</div>"
    footer = (
        f"<footer>&copy; {firm.founded}-2026 {_esc(firm.name)}. All rights reserved. "
        "Privacy Policy | Terms of Use | Accessibility | Sitemap</footer>"
    )
    rendered = f"<html><head>{head}</head><body>{cookie}{nav}<main>{body}</main>{footer}</body></html>"
    if features.get("js_rendered"):
        initial = (
            f"<html><head>{head}</head><body><div id='root'></div>"
            "<noscript>You need to enable JavaScript to run this app.</noscript>"
            "<script src='/static/js/main.4f2a9c.js'></script></body></html>"
        )
    else:
        initial = rendered
    return initial, rendered


def _facts_lines(firm: Firm, cat: dict) -> dict:
    svc, certs = cat["services"], cat["certifications"]
    return {
        "services": [svc[s]["label"] for s in firm.services],
        "certs": [certs[c]["label"] for c in firm.certifications],
        "areas": ", ".join(firm.service_regions),
    }


def descriptive_heading(firm: Firm, cat: dict) -> str:
    """Main heading that says what the firm does and where, e.g.
    'Port drayage and refrigerated trucking in Chicago, Illinois'."""
    labels = [cat["services"][s]["label"] for s in firm.services[:2]]
    return f"{cap(' and '.join(labels))} in {firm.city}, {firm.state}"


def render_site(firm: Firm, features: dict) -> list[Resource]:
    cat = CATEGORIES[firm.category]
    f = _facts_lines(firm, cat)
    c = firm.copy
    base = f"https://{firm.domain}"
    out: list[Resource] = []
    if features.get("descriptive_heading"):
        h1 = descriptive_heading(firm, cat)
        brand = f"<p class='brand'>{_esc(firm.name)}</p>"
        services_h1 = f"{cap(', '.join(f['services']))} services in {firm.city}, {firm.state}"
    else:
        h1, brand, services_h1 = firm.name, "", "Our services"

    cert_html = (
        "<section><h2>Certifications</h2><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in f["certs"]) + "</ul></section>"
        if f["certs"] else ""
    )
    home_body = (
        f"<h1>{_esc(h1)}</h1>{brand}<p class='tagline'>{_esc(c['tagline'])}</p>"
        f"<section><h2>About us</h2><p>{_esc(c['about'])}</p></section>"
        "<section><h2>What we do</h2><ul>" + "".join(f"<li>{_esc(s)}</li>" for s in f["services"]) + "</ul></section>"
        + cert_html
        + f"<p>Located in {_esc(firm.city)}, {_esc(firm.state)}. <a href='/contact.html'>Contact us</a> for a quote.</p>"
    )
    services_body = f"<h1>{_esc(services_h1)}</h1>" + "".join(
        f"<section><h2>{_esc(cap(cat['services'][s]['label']))}</h2><p>{_esc(c['service_blurbs'].get(s, ''))}</p></section>"
        for s in firm.services
    )
    contact_body = (
        f"<h1>Contact {_esc(firm.name)}</h1><p>{_esc(firm.city)}, {_esc(firm.state)}</p>"
        f"<p>Service area: {_esc(f['areas'])}</p><p>Phone: (555) 555-{int(firm.firm_id.split('_')[1]):04d}</p>"
        f"<p>Email: quotes@{firm.domain}</p>"
    )
    pages = [("/", f"{firm.name} | {c['tagline']}", home_body),
             ("/services.html", f"Services | {firm.name}", services_body),
             ("/contact.html", f"Contact | {firm.name}", contact_body)]
    if features.get("faq_page"):
        faq_body = "<h1>Frequently asked questions</h1>" + "".join(
            f"<h3>{_esc(item['q'])}</h3><p>{_esc(item['a'])}</p>" for item in c["faq"]
        )
        pages.append(("/faq.html", f"FAQ | {firm.name}", faq_body))

    for path, title, body in pages:
        initial, rendered = _page(firm, title, body, features, cat)
        out.append(Resource(base + path, "html", title, initial, rendered, firm.firm_id, "firm_page"))

    if features.get("markdown"):
        md = [f"# {h1}"] + ([firm.name] if brand else []) + ["", f"> {c['tagline']}", "", "## About", c["about"], "", "## Services"]
        for s in firm.services:
            md.append(f"### {cap(cat['services'][s]['label'])}")
            md.append(c["service_blurbs"].get(s, ""))
        if f["certs"]:
            md += ["", "## Certifications"] + [f"- {x}" for x in f["certs"]]
        md += ["", "## Contact", f"{firm.city}, {firm.state}. Service area: {f['areas']}.", f"Email: quotes@{firm.domain}"]
        if features.get("faq_page"):
            md += ["", "## FAQ"] + [f"**{i['q']}** {i['a']}" for i in c["faq"]]
        text = "\n".join(md)
        out.append(Resource(base + "/index.md", "markdown", firm.name, text, text, firm.firm_id, "markdown"))

    if features.get("llms_txt"):
        lines = [
            f"# {firm.name}", f"> {cap(cat['provider_noun'])} in {firm.city}, {firm.state}. {c['tagline']}", "",
            f"Services: {', '.join(f['services'])}",
            f"Service area: {f['areas']}",
            f"Certifications: {', '.join(f['certs']) if f['certs'] else 'none listed'}",
            f"Founded: {firm.founded}. Size: {firm.size} {cat['size_unit']}.",
            f"Contact: quotes@{firm.domain}", "", "## Pages",
            f"- [Home]({base}/)", f"- [Services]({base}/services.html)", f"- [Contact]({base}/contact.html)",
        ]
        if features.get("markdown"):
            lines.append(f"- [Full site in markdown]({base}/index.md)")
        text = "\n".join(lines)
        out.append(Resource(base + "/llms.txt", "text", f"{firm.name} llms.txt", text, text, firm.firm_id, "llms_txt"))
    return out


def plural(noun_phrase: str) -> str:
    """'freight trucking company' -> 'freight trucking companies'."""
    if noun_phrase.endswith("y") and noun_phrase[-2:-1] not in "aeiou":
        return noun_phrase[:-1] + "ies"
    return noun_phrase + "s"


def directory_domain(category: str) -> str:
    return {"freight_trucking": "freightfinderdirectory.example",
            "contract_packaging": "packagersdirectory.example"}.get(category, f"{category}directory.example")


def render_directory(category: str, listed: list[Firm]) -> list[Resource]:
    """Third-party directory: one page per region listing firms that serve it."""
    cat = CATEGORIES[category]
    dom = directory_domain(category)
    out = []
    regions = sorted({r for firm in listed for r in firm.service_regions})
    for region in regions:
        firms = [x for x in listed if region in x.service_regions]
        rows = "".join(
            f"<li><a href='https://{x.domain}/'>{_esc(x.name)}</a>, {_esc(x.city)}, {_esc(x.state)}. "
            f"Services: {_esc(', '.join(cat['services'][s]['label'] for s in x.services))}.</li>"
            for x in firms
        )
        title = f"Top {plural(cat['label'])} in the {region} | Directory"
        desc = f"Compare {len(firms)} {plural(cat['label'])} serving the {region}: services, locations and contact links."
        page = (f"<html><head><title>{_esc(title)}</title><meta name='description' content='{_esc(desc)}'></head><body><nav>Home | Browse | Add your business</nav>"
                f"<main><h1>{_esc(title)}</h1><ul>{rows}</ul></main></body></html>")
        url = f"https://{dom}/{region.lower()}"
        out.append(Resource(url, "html", title, page, page, None, "directory"))
    return out
