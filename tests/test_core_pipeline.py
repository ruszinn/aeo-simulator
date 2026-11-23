import json
import random
from pathlib import Path

import pytest

from aeosim.agents.base import Episode
from aeosim.agents.llm import run_llm
from aeosim.agents.scripted import ScriptedChatModel
from aeosim.analysis import build_panel, load_episodes
from aeosim.categories import CATEGORIES
from aeosim.config import apply_profile, load_config
from aeosim.market import generate_brief, generate_pool
from aeosim.runner import run_experiment
from aeosim.scenario import build_scenario
from aeosim.seller import fact_violations
from aeosim.site import FEATURES, render_site
from aeosim.web import MockWeb, html_to_markdown, make_snippet

ROOT = Path(__file__).resolve().parent.parent
OFF = {k: False for k in FEATURES}


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "configs/pilot.yaml")
    c["_base"] = str(ROOT)
    c["output_dir"] = str(tmp_path)
    c["experiment"]["n_scenarios"] = 4
    c["_allow_dirty"] = True
    return c


def firm(i=14):
    return generate_pool("freight_trucking", 40, 1)[i]


def test_markdown_converter_keeps_headings_and_drops_structured_data():
    f = firm()
    html = MockWeb(render_site(f, {**OFF, "schema_jsonld": True})).get(f"https://{f.domain}/services.html").initial
    _, md = html_to_markdown(html, keep_jsonld=False, main_content=False)
    assert "\n## Port drayage" in md and "LocalBusiness" not in md and "cookies" in md
    _, clean = html_to_markdown(html, keep_jsonld=False, main_content=True)
    assert "cookies" not in clean and "[Home]" not in clean and "## Port drayage" in clean


def test_snippets_start_at_heading_or_use_meta():
    f = firm()
    html = MockWeb(render_site(f, {**OFF, "descriptive_heading": True})).get(f"https://{f.domain}/").initial
    h1 = make_snippet(html, "h1", 200)
    assert h1.startswith("Warehousing and cross-docking") and len(h1) <= 203
    assert make_snippet(html, "meta", 200) == f.copy["tagline"]
    js = MockWeb(render_site(f, {**OFF, "js_rendered": True})).get(f"https://{f.domain}/").initial
    assert make_snippet(js, "h1", 200) == ""


def test_briefs_only_require_matching_or_general_certifications():
    rng = random.Random(2)
    for cat, spec in CATEGORIES.items():
        tied = set(spec["service_cert_links"].values())
        for _ in range(500):
            b = generate_brief(cat, rng)
            if b.required_cert in tied:
                assert spec["service_cert_links"].get(b.service) == b.required_cert


def test_profiles_override_config_blocks(cfg):
    inst = apply_profile(cfg, "chatgpt_instant")
    gem = apply_profile(cfg, "gemini")
    assert inst["agent"]["max_fetches"] == 0 and inst["web"]["snippet_source"] == "h1"
    assert gem["web"]["crawler_renders_js"] is True and gem["profile"] == "gemini"
    assert apply_profile(cfg, None)["web"] == cfg["web"]


def test_snippet_only_mode_hides_the_fetch_tool(cfg):
    c = apply_profile(cfg, "chatgpt_instant")
    sc = build_scenario(c["experiment"], c["features"], generate_pool("freight_trucking", 40, 1), 1)
    ep = Episode(sc, c["web"], c["search"], c["agent"], 3)
    assert "fetch_page" not in ep.tool_names and "titles and snippets" in ep.system_prompt
    usage, _, error, _ = run_llm(ep, {"provider": "scripted"}, chat_model=ScriptedChatModel())
    assert error is None and ep.fetches == [] and ep.recommendations
    assert usage["model_version"] == "scripted-test-1"


def test_forced_exposure_shows_every_firm_once(cfg):
    c = apply_profile(cfg, "chatgpt_thinking")
    c["agent"]["mode"] = "forced_exposure"
    sc = build_scenario(c["experiment"], c["features"], generate_pool("freight_trucking", 40, 1), 2)
    ep = Episode(sc, c["web"], c["search"], c["agent"], 3)
    shown = [r["firm_id"] for r in ep.exposure]
    assert sorted(shown) == sorted(f.firm_id for f in sc.firms) and "web_search" not in ep.tool_names
    assert ep.first_message.startswith(sc.brief.text) and sc.firms[0].domain in ep.first_message
    run_llm(ep, {"provider": "scripted"}, chat_model=ScriptedChatModel())
    assert ep.recommendations and len(ep.searches) == 1


def test_copilot_profile_shortens_queries(cfg):
    c = apply_profile(cfg, "copilot")
    sc = build_scenario(c["experiment"], c["features"], generate_pool("freight_trucking", 40, 1), 0)
    ep = Episode(sc, c["web"], c["search"], c["agent"], 3)
    ep.search("one two three four five six seven eight")
    assert ep.searches[0]["query_sent"] == "one two three four five six"


def test_grid_run_records_profiles_and_resumes(cfg):
    out = run_experiment(cfg)
    eps = load_episodes(out)
    assert len(eps) == 2 * 2 * 4 and {e["profile"] for e in eps} == {"chatgpt_thinking", "chatgpt_instant"}
    assert all(e["code_version"] for e in eps)
    run_experiment(cfg)
    assert len(load_episodes(out)) == 16
    panel = build_panel(eps)
    assert panel.model.str.contains("chatgpt_instant").any()


def test_budget_stops_new_episodes(cfg):
    cfg["models"] = ["models/scripted.yaml"]
    cfg["profiles"] = ["chatgpt_thinking"]
    cfg["max_workers"] = 1
    cfg["budget"] = {"max_input_tokens": 500}
    out = run_experiment(cfg)
    assert len(load_episodes(out)) < 4


def test_fact_guard_catches_invented_claims():
    f = firm(0)
    bad = {**f.copy, "about": f.copy["about"] + " We are ISO 9001 certified and offer flatbed hauling."}
    found = fact_violations(f, bad)
    assert any("ISO 9001" in x for x in found) == ("iso9001" not in f.certifications)
    assert any("flatbed" in x for x in found) == ("flatbed" not in f.services)
    assert fact_violations(f, f.copy) == []
