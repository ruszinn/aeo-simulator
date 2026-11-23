"""Fictional firms and buyer briefs.

A Firm is a fact sheet: what the company does, where, and with which
certifications. Its website copy (tagline, about text, service blurbs, FAQ)
lives in Firm.copy so a seller agent can rewrite it while the facts stay fixed.

The firm pool for a category is generated once from a fixed corpus seed, so
the same firms recur across scenarios. That recurrence is what lets the
analysis estimate firm fixed effects.
"""

from __future__ import annotations

import copy as _copy
import random
import re
from dataclasses import asdict, dataclass, field

from .categories import CATEGORIES, NEIGHBORS, REGIONS

SURNAMES = [
    "Kessler", "Marlowe", "Hadley", "Brandt", "Okafor", "Lindqvist", "Delgado", "Prescott",
    "Vance", "Whitaker", "Novak", "Castellano", "Bauer", "Ashford", "Mendel", "Rowan",
    "Sterling", "Calloway", "Duquesne", "Halvorsen", "Iverson", "Moreau", "Pruitt", "Quill",
]
PLACE_WORDS = [
    "Blue Ridge", "Great Lakes", "Keystone", "Lone Star", "Heartland", "Riverbend", "Ironwood",
    "Summit", "Harbor Point", "Prairie", "Cardinal", "Northstar", "Granite", "Bluegrass", "Tri-State",
]
STYLES = ["plain", "sales", "technical"]


def cap(text: str) -> str:
    """Uppercase the first letter only (keeps acronyms like LTL intact)."""
    return text[:1].upper() + text[1:]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@dataclass
class Firm:
    firm_id: str
    category: str
    name: str
    domain: str
    city: str
    state: str
    region: str
    service_regions: list[str]
    services: list[str]
    certifications: list[str]
    founded: int
    size: int
    style: str
    copy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def clone(self) -> "Firm":
        return _copy.deepcopy(self)


@dataclass
class Brief:
    category: str
    buyer_type: str
    buyer_company: str
    service: str
    city: str
    state: str
    region: str
    volume: str
    required_cert: str | None
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _all_cities():
    for region, states in REGIONS.items():
        for state, cities in states.items():
            for city in cities:
                yield region, state, city


def default_copy(firm: Firm, cat: dict, rng: random.Random) -> dict:
    """Website copy written from the fact sheet in one of three house styles."""
    svc = cat["services"]
    certs = cat["certifications"]
    svc_labels = [svc[s]["label"] for s in firm.services]
    joined = ", ".join(svc_labels[:-1]) + (" and " if len(svc_labels) > 1 else "") + svc_labels[-1]
    areas = ", ".join(firm.service_regions)
    years = 2026 - firm.founded

    if firm.style == "sales":
        tagline = rng.choice([
            "Moving your business forward.",
            "Reliability you can count on, every time.",
            "Your partner for what comes next.",
        ])
        about = (
            f"For over {years} years, {firm.name} has delivered world-class service to customers "
            f"who demand the best. Our passionate team goes the extra mile so you can focus on growth. "
            f"Based in {firm.city}, we are proud to be a trusted partner to businesses of every size."
        )
    elif firm.style == "technical":
        tagline = f"{cap(svc_labels[0])} from {firm.city}, {firm.state}."
        about = (
            f"{firm.name} provides {joined} from our {firm.city}, {firm.state} facility. "
            f"We serve the {areas} region(s) with {firm.size} {cat['size_unit']}. "
            f"Founded {firm.founded}."
        )
    else:
        tagline = f"{cap(cat['provider_noun'])} serving {firm.service_regions[0]} businesses."
        about = (
            f"{firm.name} is a family-owned {cat['provider_noun']} in {firm.city}, {firm.state}. "
            f"We started in {firm.founded} and today offer {joined}. "
            f"Our customers are mostly in the {areas}."
        )

    blurbs = {}
    for s in firm.services:
        terms = svc[s]["terms"]
        if firm.style == "sales":
            blurbs[s] = f"Our {svc[s]['label']} solutions are built around you. Ask us how we can help."
        elif firm.style == "technical":
            blurbs[s] = f"{cap(svc[s]['label'])}: {', '.join(terms[1:3])}. Capacity scales with demand."
        else:
            blurbs[s] = f"We handle {svc[s]['label']} ({terms[1] if len(terms) > 1 else terms[0]}) for customers across the {areas}."

    faq = [
        {"q": f"What areas does {firm.name} serve?", "a": f"We serve customers in the {areas}, from our base in {firm.city}, {firm.state}."},
        {"q": "What services do you offer?", "a": f"We offer {joined}."},
    ]
    if firm.certifications:
        cert_labels = ", ".join(certs[c]["label"] for c in firm.certifications)
        faq.append({"q": "What certifications do you hold?", "a": f"We are {cert_labels}."})
    return {"tagline": tagline, "about": about, "service_blurbs": blurbs, "faq": faq}


def generate_pool(category: str, n: int, corpus_seed: int) -> list[Firm]:
    cat = CATEGORIES[category]
    rng = random.Random(f"{category}:{corpus_seed}")
    cities = list(_all_cities())
    services = list(cat["services"])
    cert_keys = list(cat["certifications"])
    firms, used_names = [], set()

    for i in range(n):
        while True:
            pattern = rng.random()
            if pattern < 0.5:
                name = f"{rng.choice(SURNAMES)} {rng.choice(cat['name_nouns'])}"
            else:
                name = f"{rng.choice(PLACE_WORDS)} {rng.choice(cat['name_nouns'])}"
            if name not in used_names:
                used_names.add(name)
                break
        region, state, city = cities[i % len(cities)] if i < len(cities) else rng.choice(cities)
        service_regions = [region] + [r for r in NEIGHBORS[region] if rng.random() < 0.35]
        # Round-robin the first service so every service has providers.
        first = services[i % len(services)]
        others = rng.sample([s for s in services if s != first], k=rng.randint(1, 3))
        firm_services = [first] + others
        certs = set()
        for s in firm_services:
            link = cat["service_cert_links"].get(s)
            if link and rng.random() < 0.6:
                certs.add(link)
        for c in rng.sample(cert_keys, k=rng.randint(0, 2)):
            if c not in cat["service_cert_links"].values():
                certs.add(c)
        lo, hi = cat["size_range"]
        firm = Firm(
            firm_id=f"{category[:2]}_{i:03d}",
            category=category,
            name=name,
            domain=f"{slugify(name)}.example",
            city=city,
            state=state,
            region=region,
            service_regions=service_regions,
            services=firm_services,
            certifications=sorted(certs),
            founded=rng.randint(1958, 2021),
            size=int(round(rng.uniform(lo ** 0.5, hi ** 0.5) ** 2)),
            style=rng.choice(STYLES),
        )
        firm.copy = default_copy(firm, cat, rng)
        firms.append(firm)
    return firms


BUYER_COMPANY_WORDS = ["Acme", "Brightline", "Copperleaf", "Driftwood", "Evergreen", "Foxglove", "Juniper", "Meridian", "Oakhurst", "Westfield"]

BRIEF_TEMPLATES = [
    "We're {article} {buyer_type} in {city}, {state}. We need a {provider} that can handle {need}, {volume}. {cert_sentence}Who should we talk to?",
    "I run procurement at {company}, {article} {buyer_type} based in {city}, {state}. We're looking for a {provider} for {need}, {volume}. {cert_sentence}Can you recommend a few companies?",
    "Looking for a {provider} near {city}, {state}. We're {article} {buyer_type} and need {need}, {volume}. {cert_sentence}Which providers would you shortlist?",
]


def generate_brief(category: str, rng: random.Random) -> Brief:
    cat = CATEGORIES[category]
    buyer_type, service = rng.choice(cat["buyers"])
    region, state, city = rng.choice(list(_all_cities()))
    volume = rng.choice(cat["volume_phrases"]).format(n=rng.choice([5, 10, 20, 40]))
    linked = cat["service_cert_links"].get(service)
    roll = rng.random()
    if linked and roll < 0.6:
        required = linked
    elif roll < 0.25:
        required = rng.choice(list(cat["certifications"]))
    else:
        required = None
    cert_sentence = f"They have to be {cat['certifications'][required]['label']}. " if required else ""
    company = f"{rng.choice(BUYER_COMPANY_WORDS)} {cap(buyer_type.split()[-1])}s"
    article = "an" if buyer_type[0] in "aeiou" else "a"
    text = rng.choice(BRIEF_TEMPLATES).format(
        article=article, buyer_type=buyer_type, city=city, state=state, provider=cat["provider_noun"],
        need=cat["services"][service]["buyer_need"], volume=volume, cert_sentence=cert_sentence,
        company=company,
    )
    return Brief(category, buyer_type, company, service, city, state, region, volume, required, text)


def fit(firm: Firm, brief: Brief) -> dict:
    """Ground-truth match between a firm and a brief, used as analysis controls."""
    service_match = int(brief.service in firm.services)
    region_match = int(brief.region in firm.service_regions)
    cert_match = int(brief.required_cert is None or brief.required_cert in firm.certifications)
    return {
        "service_match": service_match,
        "region_match": region_match,
        "cert_match": cert_match,
        "full_fit": int(service_match and region_match and cert_match),
    }
