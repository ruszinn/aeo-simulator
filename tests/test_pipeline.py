import json
import random
from pathlib import Path

import pytest

from aeosim.agents.base import Episode
from aeosim.agents.llm import run_llm
from aeosim.agents.scripted import ScriptedChatModel
from aeosim.analysis import analyze
from aeosim.config import load_config
from aeosim.market import generate_pool
from aeosim.runner import run_experiment
from aeosim.scenario import build_scenario
from aeosim.seller import keyword_enrich, run_seller
from aeosim.site import render_site
from aeosim.validation import detect_features
from aeosim.web import MockWeb

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "configs/pilot.yaml")
    c["_base"] = str(ROOT)
    c["output_dir"] = str(tmp_path)
    c["experiment"]["n_scenarios"] = 6
    c["_allow_dirty"] = True
    return c


def pool(c):
    e = c["experiment"]
    return generate_pool(e["category"], e["pool_size"], e["corpus_seed"])


def test_scenarios_are_reproducible(cfg):
    p = pool(cfg)
    a = build_scenario(cfg["experiment"], cfg["features"], p, 3)
    b = build_scenario(cfg["experiment"], cfg["features"], p, 3)
    assert a.brief == b.brief and a.features == b.features
    assert [f.firm_id for f in a.firms] == [f.firm_id for f in b.firms]
    assert sum(fit["full_fit"] for fit in a.fits.values()) >= cfg["experiment"]["min_full_fit"]


def test_overrides_change_only_the_focal_firm(cfg):
    p = pool(cfg)
    base = build_scenario(cfg["experiment"], cfg["features"], p, 2)
    fid = base.firms[0].firm_id
    after = build_scenario(cfg["experiment"], cfg["features"], p, 2, feature_overrides={fid: {"llms_txt": True}})
    assert after.brief == base.brief and after.search_seed == base.search_seed
    for f in base.firms[1:]:
        assert after.features[f.firm_id] == base.features[f.firm_id]
    assert after.features[fid]["llms_txt"] is True


def test_js_rendered_site_is_blank_to_plain_fetcher():
    firm = generate_pool("freight_trucking", 5, 1)[0]
    feats = {"markdown": False, "llms_txt": False, "schema_jsonld": False, "js_rendered": True,
             "faq_page": False, "directory_listing": False}
    web = MockWeb(render_site(firm, feats), {"fetcher": "html_text"})
    text = web.fetch(firm.domain)["text"]
    assert "enable JavaScript" in text and firm.copy["about"] not in text
    web_md = MockWeb(render_site(firm, {**feats, "markdown": True}), {"fetcher": "prefer_markdown"})
    out = web_md.fetch(firm.domain)
    assert out["served"] == "markdown_alternate" and firm.copy["about"] in out["text"]


def test_jsonld_reaches_crawler_but_not_plain_fetcher():
    firm = generate_pool("freight_trucking", 5, 1)[1]
    feats = {k: False for k in ["markdown", "llms_txt", "js_rendered", "faq_page", "directory_listing"]}
    feats["schema_jsonld"] = True
    web = MockWeb(render_site(firm, feats), {"fetcher": "html_text", "crawler_reads_jsonld": True})
    home = [d for d in web.index_docs() if d["url"].endswith("/")][0]
    assert "Structured data" in home["text"]
    assert "Structured data" not in web.fetch(firm.domain)["text"]


def test_llm_loop_with_scripted_model(cfg):
    p = pool(cfg)
    sc = build_scenario(cfg["experiment"], cfg["features"], p, 1)
    ep = Episode(sc, cfg["web"], cfg["search"], cfg["agent"], 3)
    usage, steps, error, transcript = run_llm(ep, {"provider": "scripted"}, chat_model=ScriptedChatModel())
    assert error is None and ep.recommendations and all(r["resolved"] for r in ep.recommendations)
    assert len(ep.searches) == 1 and len(ep.fetches) == 3 and usage["input_tokens"] > 0


def test_run_resume_and_analyze(cfg):
    out = run_experiment(cfg)
    n1 = len(out.read_text().splitlines())
    run_experiment(cfg)  # second call skips finished episodes
    assert len(out.read_text().splitlines()) == n1 == 24  # 2 models x 2 profiles x 6 scenarios
    report = analyze(out.parent).read_text()
    assert "Funnel by feature" in report


def test_seller_pairs(cfg):
    cfg["experiment"]["name"] = "seller_test"
    cfg["seller"] = {"strategy": "keyword_enrich"}
    cfg["models"] = ["models/baseline.yaml"]
    report_path = run_seller(cfg)
    eps = [json.loads(l) for l in (report_path.parent / "seller_episodes.jsonl").read_text().splitlines()]
    report = report_path.read_text()
    assert {e["arm"] for e in eps} == {"before", "after:keyword_enrich"}
    assert "paired scenarios" in report


def test_keyword_enrich_keeps_facts():
    firm = generate_pool("freight_trucking", 5, 1)[2]
    new = keyword_enrich(firm)
    assert new.services == firm.services and new.certifications == firm.certifications
    assert new.copy["about"] != firm.copy["about"] and firm.copy["about"] == firm.clone().copy["about"]


def test_feature_audit_detection():
    html = ("<html><head><script type='application/ld+json'>{}</script><link rel='alternate' type='text/markdown' href='/index.md'>"
            "</head><body><a href='/faq'>FAQ</a>" + "word " * 100 + "</body></html>")
    f = detect_features(html, 200, "# Firm\n> summary of services", 404, "")
    assert f["schema_jsonld"] and f["markdown"] and f["llms_txt"] and f["faq_page"] and not f["js_rendered"]
    shell = "<html><body><div id='root'></div><script src='/app.js'></script></body></html>"
    assert detect_features(shell, 404, "", 404, "")["js_rendered"]


def test_dirty_repo_blocks_runs(cfg, tmp_path):
    from aeosim.provenance import DirtyRepoError, run_dir_for
    cfg["_base"] = str(tmp_path)  # a folder outside any Git repo
    cfg["_allow_dirty"] = False
    with pytest.raises(DirtyRepoError):
        run_dir_for(cfg)
    cfg["_allow_dirty"] = True
    path, info = run_dir_for(cfg)
    assert path.name == "nogit" and info["commit"] is None


def test_descriptive_heading_changes_h1_and_keeps_name():
    firm = generate_pool("freight_trucking", 5, 1)[0]
    base = {k: False for k in ["markdown", "llms_txt", "schema_jsonld", "js_rendered", "faq_page", "directory_listing"]}
    off = MockWeb(render_site(firm, {**base, "descriptive_heading": False}), {"fetcher": "html_text"})
    on = MockWeb(render_site(firm, {**base, "descriptive_heading": True}), {"fetcher": "html_text"})
    off_text, on_text = off.fetch(firm.domain)["text"], on.fetch(firm.domain)["text"]
    assert f"<h1>{firm.name}</h1>" in off.get(firm.domain).initial
    assert f" in {firm.city}, {firm.state}" in on_text and firm.name in on_text
    assert f" in {firm.city}, {firm.state}</h1>" in on.get(firm.domain).initial
