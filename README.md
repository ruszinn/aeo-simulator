# AEO Simulator

A simulator of how AI search agents find and recommend B2B service providers. It measures which website features help a small firm get recommended when a business buyer asks an AI assistant for providers (AEO: agentic engine optimization).

The design adapts ACES (Allouah, Besbes, Figueroa, Kanoria and Kumar, *What Is Your AI Agent Buying?*, arXiv 2508.02630) from product shopping to B2B provider search. ACES shows an agent one page of eight products. Here the agent works the way AI search works for a buyer: it searches, reads company websites, and returns a ranked shortlist. Every outcome splits into two stages:

1. **Retrieval**: did the search step surface the firm?
2. **Selection**: once the agent saw the firm, did it recommend it?

Two things shape each outcome, and the simulator varies them separately:

- **The model**: the real LLM (Claude, GPT, Gemini), called through its API. It decides what to search, which pages to read and whom to recommend.
- **The pipeline**: the crawler, index, snippets and page reader around the model. Real products differ here, so each one gets a **platform profile**. Running every model under every profile separates model effects from pipeline effects.

## How a run works

```
scenario generator  ->  buyer agent  ->  web_search  ->  mock search engine (BM25)
   (seeded brief,          (LLM or          fetch_page ->  mock web (fictional firm sites,
    market, feature         baseline)       recommend       seven AEO switches per firm)
    switches)                   |                    ^
                                v                    |
                     episodes.jsonl          platform profile sets crawler,
                                             snippet, fetcher and agent limits
```

- **Firms** are fictional companies generated from a fixed corpus seed, so the same 40 firms recur across scenarios and the analysis can give each firm its own baseline (firm fixed effects).
- **Scenarios** are seeded: scenario 17 with seed 11 is identical on every run, every model and every profile.
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

## Platform profiles

Each file in `profiles/` bundles one product's pipeline. The comments in each file cite the evidence and mark which settings are assumptions.

| Profile | Crawler | Snippets | Page reading | Agent limits |
|---|---|---|---|---|
| `chatgpt_thinking` | skips JavaScript | start at the main heading | HTML converted to Markdown | 5 searches, 5 page reads |
| `chatgpt_instant` | skips JavaScript | start at the main heading | none (snippet-only) | 2 searches, 0 page reads |
| `gemini` | renders JavaScript | meta description | HTML converted to Markdown, JavaScript run | 5 searches, 5 page reads |
| `claude` | skips JavaScript | meta description | HTML converted to Markdown | 5 searches, 5 page reads |
| `copilot` | skips JavaScript | meta description | HTML converted to Markdown | queries cut to 6 words; 3 searches, 3 page reads |

ChatGPT's pipeline is the best documented, so the simulator's defaults (used when a config lists no profile) match `chatgpt_thinking`. The other profiles are approximations of undisclosed systems.

**What the pipeline pieces do:**
- **Crawler view** decides what the search engine indexes: the raw HTML (scripts unrun) or the rendered page, with or without llms.txt files and structured data.
- **Snippets** are what a results list shows for each page. `h1` starts at the page's main heading and runs on through the page text, as measured for ChatGPT. `meta` uses the meta description.
- **The fetcher** decides what the agent reads when it opens a page. `markdown_convert` parses the HTML, drops scripts, frames and structured data, and writes the rest as Markdown (headings become `#`, links become `[text](url)`), as measured for ChatGPT's cache. `fetcher_main_content: true` also drops menus, footers and cookie banners, the way a browser's reader view does; it's a robustness setting. `html_text` and `prefer_markdown` (read the site's own `/index.md`) remain available.

## Agent modes

- **`search`** (default): the agent searches the mock web, reads pages and recommends. Results reflect both stages.
- **`forced_exposure`**: the agent gets one list of every firm in the market, in random order, with each homepage's title and snippet, and searches nothing. Every firm is seen, so differences in recommendation come from how the model judges the firms. Use it to measure selection cleanly.
- **Snippet-only** is any profile with `max_fetches: 0` (like `chatgpt_instant`): the page-reading tool is removed and the agent judges firms from titles and snippets alone.

## Install

Python 3.11 or newer, Git, and [uv](https://docs.astral.sh/uv/) (`brew install uv` on a Mac).

```bash
git clone https://github.com/ruszinn/aeo-simulator.git ~/code/aeo-simulator
cd ~/code/aeo-simulator
uv sync --extra dev --extra anthropic   # exact package versions from uv.lock, into .venv/
cp .env.sample .env                     # then paste your API keys into .env
uv run pytest -q                        # 21 tests, about 10 seconds
```

`uv sync` reads `uv.lock`, the file that pins the exact version of every package, so every co-author gets the same environment. Add `--extra openai` or `--extra google` for other providers. Prefix commands with `uv run` (or activate `.venv`).

## Quick start (free, offline)

```bash
uv run python -m aeosim preview configs/pilot.yaml --scenario 3   # one scenario: brief, market, what the agent reads
uv run python -m aeosim run configs/pilot.yaml                    # 40 scenarios x 2 offline agents x 2 profiles
uv run python -m aeosim analyze runs/pilot/<commit>               # the run command prints this path
uv run python -m aeosim serve configs/pilot.yaml --scenario 3     # browse the mock web at localhost:8000
```

The pilot uses two offline agents. `Keyword baseline` ranks firms by how many of the brief's key terms appear in snippets and pages it read. `Scripted LLM (test)` is a fake chat model that drives the real LLM tool-calling loop. Both cost nothing and check the whole pipeline first.

## Running with real models

1. Model files in `models/` cover Claude Sonnet 5.5, GPT-6.2 Sol, Gemini 3.7 Flash and Azure OpenAI (IDs checked November 2026). Copy one and edit `provider` and `model` for others; any provider LangChain's `init_chat_model` supports works.
2. Uncomment the model files in a config's `models:` list.
3. Do a small paid test first: `uv run python -m aeosim run configs/main.yaml --limit 5`, then read the transcripts in `runs/main/<commit>/episodes.jsonl`.
4. Run the full experiment. Interrupted runs resume where they stopped when you rerun the same command on the same commit.

**Safeguards:**
- **Budget cap.** A config's `budget:` block (`max_input_tokens`, `max_output_tokens`) stops the run from starting new episodes once total tokens pass a cap. Tokens already in the output file count, so a resumed run respects the same cap.
- **Exact model versions.** Each episode logs the model ID requested and the exact version the API reports (`model_version`), plus the temperature.
- **Prompt caching.** For Anthropic models (`prompt_cache: true`, the default), each step marks the shared conversation prefix for caching, so repeated input is billed at the cached rate. OpenAI caches long prompts automatically. Cache reads and writes are logged per episode.

**Cost.** Input tokens per episode grow with page reads, because each read adds up to `max_fetch_chars` characters (6,000 by default, about 1,500 tokens) to the conversation, and the whole conversation is resent on every step. Expect roughly 15,000 to 40,000 input tokens per episode in page-reading profiles and far fewer in snippet-only ones. Total cost = episodes x models x profiles x tokens per episode x the provider's price. A `--limit 5` run gives a measured per-episode number to multiply out.

## Experiments

| Config | Question | Design |
|---|---|---|
| `configs/main.yaml` | Which features change whether a firm gets found and recommended, and does that differ by model and pipeline? | agents search; models x `chatgpt_thinking` and `gemini` |
| `configs/exposure.yaml` | Given every firm is seen, which features change whether the model recommends it? | forced exposure; models x thinking and instant modes |
| `configs/seller.yaml` | If one firm adopts a change, how much does its recommendation rate move? | paired before/after on identical scenarios |

Seller strategies: `aeo_checklist` (switch on markdown, llms.txt, schema, FAQ and a descriptive heading; switch off JS rendering), `keyword_enrich` (offline copy rewrite with service terms and areas), `llm_rewrite` (an LLM rewrites the copy; a rewrite that claims a certification or service the firm lacks is retried once, then rejected, and each check is saved).

```bash
uv run python -m aeosim seller configs/seller.yaml   # writes runs/<name>/<commit>/seller_report.md
```

### Validation against real AI search

```bash
uv run python -m aeosim briefs --category freight_trucking --n 60 --out validation
# paste briefs into ChatGPT search, Claude, Gemini, Copilot and Perplexity; record answers in validation_results_template.csv
uv run python -m aeosim audit urls.txt --out audit.csv   # detects the same features on the real recommended sites
```

Each live product bundles its own model with its own pipeline, so validation measures the two together. The simulator separates them; validation checks that the combined results line up with live products.

## Reading the output

Each episode in `episodes.jsonl` records the brief, the market with every firm's switches and brief fit, every search (the query, the query actually sent, and each result's rank, title and snippet), every page read, the recommendations, token usage and the model version.

`analyze` writes `runs/<name>/<commit>/analysis/` with `panel.csv` (one row per firm per episode), `funnel.csv` (stage rates per feature) and `report.md`, separately for each model and profile combination. The statistical analysis is still the first version: see the open items below.

## Configuration reference

`experiment:` name, category (`freight_trucking` or `contract_packaging`), n_scenarios, seed, corpus_seed, pool_size, market_size, min_full_fit, shortlist_size.
`features:` probability each switch is on.
`profiles:` list of profile names; every model runs under each one.
`web:` crawler_renders_js, crawler_reads_llms_txt, crawler_reads_jsonld, snippet_source (`h1` or `meta`), snippet_chars, fetcher (`markdown_convert`, `html_text` or `prefer_markdown`), fetcher_main_content, fetcher_renders_js, fetcher_keeps_jsonld, max_fetch_chars.
`search:` method (`bm25` or `hybrid`), mode (`ranked` or `shuffled`), results_per_query, max_per_domain, max_query_words.
`agent:` mode (`search` or `forced_exposure`), max_searches, max_fetches, max_steps, save_transcripts.
`budget:` max_input_tokens, max_output_tokens.
`models:` list of model files. `max_workers:` parallel episodes. `output_dir:` where runs go.

A profile's settings override the config's own `web`, `search` and `agent` blocks. Add a service category by adding an entry to `aeosim/categories.py`; add a platform by adding a file to `profiles/`.

## Project layout

```
aeosim/
  categories.py   service categories, regions, certifications, buyer types
  market.py       fictional firms, website copy, buyer briefs, brief-fit scores
  site.py         renders each firm's site with the seven switches; directory site
  web.py          mock web: crawler view, snippets, Markdown converter, fetcher
  search.py       BM25 and hybrid search, ranked or shuffled
  scenario.py     seeded scenario generator with override hooks
  agents/         episode tools, modes and limits; LLM loop; keyword baseline; scripted test model
  runner.py       parallel, resumable runner over models x profiles, with a token budget
  analysis.py     panel, funnel, first-version choice models
  provenance.py   links each run folder to its Git commit
  seller.py       paired seller experiment and the rewrite fact check
  validation.py   brief export and live-site feature audit
  serve.py        browse a scenario in a browser
profiles/         platform pipelines: chatgpt_thinking, chatgpt_instant, gemini, claude, copilot
configs/          pilot, main, exposure, seller
models/           model files
tests/            pipeline tests
.github/workflows automatic test run on every push
uv.lock           exact package versions
```

## Where things live

| What | Where | Why |
|---|---|---|
| Code, configs, profiles, model files, tests, `uv.lock` | this GitHub repo | full history of every change; co-authors share one copy |
| Run outputs (`runs/`) | shared storage (Baruch OneDrive/SharePoint or lab Azure storage); final data on Zenodo or Hugging Face Datasets | logs are large and regenerate from code + config + seed |
| API keys (`.env`) and `.venv/` | your laptop only | `.gitignore` keeps both out of Git |

**Every result links to its code.** Runs write to `runs/<experiment>/<commit>/`, where `<commit>` is the first 8 characters of the Git commit that ran. `run_info.json` records the full commit, branch, tag and start time, and each episode carries the same ID in `code_version`. A run stops before starting when the repo has uncommitted changes; `--allow-dirty` overrides this for throwaway tests and labels the folder `<commit>-dirty`. To reproduce any result: `git checkout <commit>`, `uv sync`, rerun the config.

**Working on changes.** Make each change on a branch, open a pull request into `main`, and merge once the automatic test check passes. Tag the commit behind any numbers that go into the paper (`git tag v0.2-pilot && git push --tags`). Track the open design decisions as GitHub Issues.

## Status

Tested here with the offline agents: every module end to end, 21 passing tests, and offline runs of every profile and mode. Tool schemas and prompt-caching markers format correctly for the Anthropic and OpenAI LangChain clients.

Still to verify on a machine with API keys and open internet: live LLM episodes, the `hybrid` search method (downloads an embedding model), `llm_rewrite`, and `audit` against real sites.

**Open items:**
- **Statistics** (next phase): a pooled model with feature x model and feature x profile terms, scenario-level bootstrap error bars, interactions for mechanism-backed feature pairs, written hypotheses, a power analysis and multiple-testing corrections. In `search` mode, the selection-stage model (recommended, among firms seen) is biased, because features help decide which firms get seen; use `forced_exposure` runs for selection, and the all-firms outcome for search runs.
- **Design decisions for the research team:**
  1. Is the paper about models, pipelines, or both?
  2. Service categories (freight trucking and contract packaging are built in).
  3. Outcome: shortlist of three (default) or a single top pick (`shortlist_size: 1`).
  4. Which models and profiles, and how many scenarios per cell.
  5. LLM-written site copy in place of templates.
  6. Further switches and brief options: real versus styled headings, brief styles (keyword, conversational, detailed request), unstated requirements.
  7. Market realism: large competitors and third-party "top 10" pages; realistic-looking domains in place of `.example`.
