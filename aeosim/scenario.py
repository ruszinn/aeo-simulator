"""Seeded scenario generator.

A scenario is one buying task: a buyer brief, the market of firms the
buyer could find, and each firm's AEO feature switches. Every random draw
comes from a generator seeded with (experiment seed, scenario index), so
scenario 17 is identical every time it is built. The seller experiment
relies on this: it rebuilds the same scenarios and changes one firm.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .market import Brief, Firm, fit, generate_brief
from .site import FEATURES, render_directory, render_site
from .web import MockWeb


@dataclass
class Scenario:
    scenario_id: int
    seed: str
    brief: Brief
    firms: list[Firm]
    features: dict[str, dict]
    fits: dict[str, dict]
    search_seed: int
    overrides: dict = field(default_factory=dict)

    def build_web(self, web_cfg: dict) -> MockWeb:
        resources = []
        for f in self.firms:
            resources += render_site(f, self.features[f.firm_id])
        listed = [f for f in self.firms if self.features[f.firm_id]["directory_listing"]]
        if listed:
            resources += render_directory(self.brief.category, listed)
        return MockWeb(resources, web_cfg)

    def market_record(self) -> list[dict]:
        return [{"firm_id": f.firm_id, "name": f.name, "domain": f.domain, "style": f.style,
                 "features": self.features[f.firm_id], "fit": self.fits[f.firm_id]} for f in self.firms]


def build_scenario(exp: dict, feature_probs: dict, pool: list[Firm], idx: int,
                   firm_overrides: dict[str, Firm] | None = None,
                   feature_overrides: dict[str, dict] | None = None) -> Scenario:
    """Build scenario `idx`. Overrides replace a firm's copy or features after
    all random draws are made, so the rest of the scenario stays identical."""
    seed = f"{exp['seed']}:{idx}"
    rng = random.Random(seed)
    category = exp["category"]

    for _ in range(200):
        brief = generate_brief(category, rng)
        full = [f for f in pool if fit(f, brief)["full_fit"]]
        if len(full) >= exp["min_full_fit"]:
            break
    else:
        raise RuntimeError("Could not draw a brief with enough well-matched firms; enlarge pool_size.")

    n_fit = min(len(full), max(exp["min_full_fit"], rng.randint(exp["min_full_fit"], exp["min_full_fit"] + 2)))
    chosen = rng.sample(full, n_fit)
    rest = [f for f in pool if f not in chosen]
    chosen += rng.sample(rest, exp["market_size"] - n_fit)
    rng.shuffle(chosen)

    features = {f.firm_id: {k: rng.random() < feature_probs.get(k, 0.0) for k in FEATURES} for f in chosen}
    search_seed = rng.randrange(2**31)

    firms = []
    for f in chosen:
        if firm_overrides and f.firm_id in firm_overrides:
            firms.append(firm_overrides[f.firm_id])
        else:
            firms.append(f)
    if feature_overrides:
        for fid, patch in feature_overrides.items():
            if fid in features:
                features[fid] = {**features[fid], **patch}

    fits = {f.firm_id: fit(f, brief) for f in firms}
    return Scenario(idx, seed, brief, firms, features, fits, search_seed,
                    overrides={"firms": sorted(firm_overrides or {}), "features": feature_overrides or {}})
