# AEO Simulator

A simulator of how AI search agents find and recommend B2B service providers. It measures which website features help a small firm get recommended when a business buyer asks an AI assistant for providers (AEO: agentic engine optimization).

The design adapts ACES (Allouah, Besbes, Figueroa, Kanoria and Kumar, *What Is Your AI Agent Buying?*, arXiv 2508.02630) from product shopping to B2B provider search. ACES shows an agent one page of eight products. Here the agent works the way AI search works for a buyer: it searches, reads company websites, and returns a ranked shortlist. That splits every outcome into two measurable stages:

1. **Retrieval**: did the search step surface the firm?
2. **Selection**: once the agent saw the firm, did it recommend it?

## How a run works

```
scenario generator  ->  buyer agent  ->  web_search  ->  mock search engine (BM25 / hybrid)
   (seeded brief,          (LLM or          fetch_page ->  mock web (fictional firm sites,
    market, feature         baseline)       recommend       seven AEO switches per firm)
    switches)                   |
                                v
                     episodes.jsonl  ->  analysis: funnel + conditional logit per stage
```

- **Firms** are fictional companies generated from a fixed corpus seed, so the same 40 firms recur across scenarios and the analysis can give each firm its own baseline (firm fixed effects).
- **Scenarios** are seeded: scenario 17 with seed 11 is identical on every run and on every machine.
- **Feature switches**, drawn independently per firm per scenario:

| Switch | What changes on the site |
|---|---|
| `markdown` | a clean markdown copy of the whole site at `/index.md`, linked from every page |
| `llms_txt` | `/llms.txt`, a plain-text summary written for AI crawlers |
| `schema_jsonld` | schema.org structured data (services, area served, credentials) in each page |
| `js_rendered` | the first HTML a fetcher receives is an empty shell; a script fills in the content |
| `faq_page` | `/faq.html` restating services, service areas and certifications |
| `directory_listing` | the firm appears on a third-party directory site |
| `descriptive_heading` | the main heading (H1) names the services and city ("Port drayage and refrigerated trucking in Chicago, Illinois"); off, it carries the firm's name |

- **Crawler and fetcher policies** (`web:` in the config) set what the search engine indexes and what the agent reads. Real AI products differ here, so these are experiment settings. Example: with `fetcher: prefer_markdown` the agent reads `/index.md` whenever a site offers it, which also rescues JavaScript-only sites.

## Install

Python 3.11 or newer, Git, and [uv](https://docs.astral.sh/uv/) (`brew install uv` on a Mac).

```bash
git clone https://github.com/ruszinn/aeo-simulator.git ~/code/aeo-simulator
cd ~/code/aeo-simulator
uv sync --extra dev --extra anthropic   # exact package versions from uv.lock, into .venv/
cp .env.sample .env                     # then paste your API keys into .env
uv run pytest -q                        # 10 tests, about 5 seconds
```

`uv sync` reads `uv.lock`, the file that pins the exact version of every package, so every co-author gets the same environment. Add `--extra openai` or `--extra google` for other providers. Prefix commands with `uv run` (or activate `.venv`).

## Quick start (free, offline)

```bash
uv run python -m aeosim preview configs/pilot.yaml --scenario 3   # one scenario: brief, market, what the agent reads
uv run python -m aeosim run configs/pilot.yaml                    # 40 scenarios x 2 offline agents
uv run python -m aeosim analyze runs/pilot/<commit>               # the run command prints this path
uv run python -m aeosim serve configs/pilot.yaml --scenario 3     # browse the mock web at localhost:8000
```

The pilot uses two offline agents. `Keyword baseline` ranks firms by how many of the brief's key terms appear on pages it read. `Scripted LLM (test)` is a fake chat model that drives the real LLM tool-calling loop. Both cost nothing and let you check the whole pipeline first.

## Running with real models

1. Pick models in `models/` (Claude, GPT, Gemini and Azure OpenAI examples included) or copy one and edit `provider` and `model`. Any provider LangChain's `init_chat_model` supports works.
2. Add the model files to the `models:` list of a config.
3. Do a small paid test first: `uv run python -m aeosim run configs/retrieval.yaml --limit 5`, then read the transcripts in `runs/main_ranked/<commit>/episodes.jsonl`.
4. Run the full experiment. Interrupted runs resume where they stopped when you rerun the same command on the same commit.

**Cost.** Each episode is one buyer task. Input tokens per episode grow with page reads, because each read adds up to `max_fetch_chars` characters (6,000 by default, about 1,500 tokens) to the conversation, and the whole conversation is resent on every step. Expect roughly 15,000 to 40,000 input tokens per episode. Total cost = episodes x models x tokens per episode x the provider's price. The report prints the exact token totals of each run, so a `--limit 20` run gives you a measured per-episode number to multiply out.

## Experiments

| Config | Question | Notes |
|---|---|---|
| `configs/retrieval.yaml` | Which features change whether a firm surfaces and gets recommended? | results in score order |
| `configs/selection.yaml` | Given the agent saw a firm, which features change whether it recommends it? | results in random order; analysis controls for position |
| `configs/seller.yaml` | If one firm adopts a change, how much does its recommendation rate move? | paired before/after on identical scenarios |

Seller strategies: `aeo_checklist` (switch on markdown, llms.txt, schema and FAQ, switch off JS rendering), `keyword_enrich` (offline copy rewrite with service terms and areas), `llm_rewrite` (an LLM rewrites the copy within the fact sheet; set `rewriter_model`).

```bash
uv run python -m aeosim seller configs/seller.yaml   # writes runs/<name>/<commit>/seller_report.md
```

### Validation against real AI search

```bash
uv run python -m aeosim briefs --category freight_trucking --n 60 --out validation
# paste briefs into ChatGPT search, Perplexity, etc.; record answers in validation_results_template.csv
uv run python -m aeosim audit urls.txt --out audit.csv   # detects the same features on the real recommended sites
```

Compare the features of firms real AI search recommends with what the simulator predicts. Matching directions are the evidence that the simulator's findings carry over to live products.

## Reading the analysis

`runs/<name>/<commit>/analysis/` holds:

- `panel.csv`: one row per firm per episode, with `surfaced`, `best_rank`, `seen`, `fetched`, `recommended`, `top_pick`, the seven switches and the fit controls.
- `funnel.csv`: stage rates for each feature, off versus on, per model.
- `clogit_*.csv` and `report.md`: conditional logit estimates. Each firm is compared only with the other firms in the same scenario. An odds ratio of 2.0 on `markdown` in the selection stage means a firm the agent saw was twice as likely (in odds) to make the shortlist when its site offered markdown, holding the other switches, the brief fit and the firm's own baseline fixed.

The fit controls (`service_match`, `region_match`, `cert_match`) measure whether the agent recommends firms that actually match the brief. The report's "share of recommendations that fully fit the brief" is a quick quality check on each model.

## Configuration reference

`experiment:` name, category (`freight_trucking` or `contract_packaging`), n_scenarios, seed, corpus_seed, pool_size, market_size, min_full_fit, shortlist_size.
`features:` probability each switch is on.
`web:` crawler_renders_js, crawler_reads_llms_txt, crawler_reads_jsonld, fetcher (`prefer_markdown` or `html_text`), fetcher_renders_js, fetcher_keeps_jsonld, max_fetch_chars.
`search:` method (`bm25` or `hybrid`), mode (`ranked` or `shuffled`), results_per_query, max_per_domain.
`agent:` max_searches, max_fetches, max_steps, save_transcripts.
`models:` list of model files. `max_workers:` parallel episodes. `output_dir:` where runs go.

Add a service category by adding an entry to `aeosim/categories.py`.

## Project layout

```
aeosim/
  categories.py   service categories, regions, certifications, buyer types
  market.py       fictional firms, website copy, buyer briefs, brief-fit scores
  site.py         renders each firm's site with the seven switches; directory site
  web.py          mock web: crawler view and fetcher view
  search.py       BM25 and hybrid search, ranked or shuffled
  scenario.py     seeded scenario generator with override hooks
  agents/         episode tools and limits, LLM loop, keyword baseline, scripted test model
  runner.py       parallel, resumable experiment runner
  analysis.py     panel, funnel, conditional logit per stage
  provenance.py   links each run folder to its Git commit
  seller.py       paired seller experiment
  validation.py   brief export and live-site feature audit
  serve.py        browse a scenario in a browser
configs/          pilot, retrieval, selection, seller
models/           model files
tests/            pipeline tests
.github/workflows automatic test run on every push
uv.lock           exact package versions
```

## Where things live

| What | Where | Why |
|---|---|---|
| Code, configs, model files, tests, `uv.lock` | this GitHub repo | full history of every change; co-authors share one copy |
| Run outputs (`runs/`) | shared storage (Baruch OneDrive/SharePoint or lab Azure storage); final data on Zenodo or Hugging Face Datasets | logs are large and regenerate from code + config + seed |
| API keys (`.env`) and `.venv/` | your laptop only | `.gitignore` keeps both out of Git |

**Every result links to its code.** Runs write to `runs/<experiment>/<commit>/`, where `<commit>` is the first 8 characters of the Git commit that ran. `run_info.json` in that folder records the full commit, branch, tag and start time, and each episode carries the same ID in `code_version`. A run stops before starting when the repo has uncommitted changes; `--allow-dirty` overrides this for throwaway tests and labels the folder `<commit>-dirty`. To reproduce any result: `git checkout <commit>`, `uv sync`, rerun the config.

**Working on changes.** Make each change on a branch (`git switch -c add-hybrid-test`), open a pull request into `main`, and merge once the automatic test check passes. GitHub Actions (`.github/workflows/tests.yml`) runs the full test suite on every push. Tag the commit behind any numbers that go into the paper (`git tag v0.2-pilot && git push --tags`). Track the open design decisions as GitHub Issues.

## Status

Tested here: every module, end to end, with the offline agents (10 passing tests, plus 150- to 400-episode runs of the retrieval, selection and seller experiments in freight trucking and a 30-scenario check of contract packaging). Tool schemas bind cleanly to the Anthropic and OpenAI LangChain clients.

Still to verify on a machine with API keys and open internet: live LLM episodes, the `hybrid` search method (downloads an embedding model), `llm_rewrite`, and `audit` against real sites.

## Design decisions to confirm with the research team

1. Service categories to study (freight trucking and contract packaging are built in).
2. The outcome: shortlist of three (current default) or a single top pick (`shortlist_size: 1`).
3. Which models, and how many scenarios per model (ACES used 500 per category).
4. Crawler and fetcher policies to treat as the main setting; the others become robustness checks.
5. Whether firm copy should be LLM-written for more natural variety in writing style (currently three template styles).
