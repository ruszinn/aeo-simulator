"""LLM buyer agent: a LangChain tool-calling loop over the episode's tools.

Any chat model LangChain's init_chat_model supports works: set `provider`
and `model` in a models/*.yaml file (anthropic, openai, google_genai,
azure_openai, bedrock_converse, ...). Install the matching LangChain
integration package, e.g. `pip install langchain-anthropic`.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from .base import Episode


class WebSearch(BaseModel):
    """Search the web. Returns up to 10 results with title, URL and snippet."""
    query: str = Field(description="The search query.")


class FetchPage(BaseModel):
    """Read a web page and return its text."""
    url: str = Field(description="Full URL of the page to read.")


class Recommendation(BaseModel):
    company_name: str = Field(description="Company name as shown on its website.")
    website: str = Field(description="The company's website URL.")
    reason: str = Field(description="One sentence on why this company fits the buyer's needs.")


class Recommend(BaseModel):
    """Submit the final ranked shortlist of recommended companies, best first."""
    recommendations: list[Recommendation] = Field(description="Ranked list of recommended companies, best first.")


def _inline(schema: dict) -> dict:
    """Resolve $ref/$defs into one flat JSON schema (some providers reject $ref)."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(schema)


TOOL_SCHEMAS = [
    {"name": "web_search", "description": WebSearch.__doc__, "parameters": _inline(WebSearch.model_json_schema())},
    {"name": "fetch_page", "description": FetchPage.__doc__, "parameters": _inline(FetchPage.model_json_schema())},
    {"name": "recommend", "description": Recommend.__doc__, "parameters": _inline(Recommend.model_json_schema())},
]


def make_chat_model(model_cfg: dict):
    if model_cfg.get("provider") == "scripted":
        from .scripted import ScriptedChatModel
        return ScriptedChatModel()
    from langchain.chat_models import init_chat_model
    return init_chat_model(model_cfg["model"], model_provider=model_cfg["provider"],
                           temperature=model_cfg.get("temperature", 1.0), **model_cfg.get("kwargs", {}))


def _invoke_with_retry(llm, messages, tries: int = 5):
    for attempt in range(tries):
        try:
            return llm.invoke(messages)
        except Exception:  # rate limits, timeouts, transient server errors
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 3))


def _cache_marked(messages: list, provider: str | None) -> list:
    """Anthropic prompt caching: mark the system prompt and the newest message
    as cache breakpoints, so each step re-reads the shared conversation prefix
    from cache at a discount. Other providers get the messages unchanged
    (OpenAI caches long prompts automatically)."""
    if provider != "anthropic":
        return messages
    out = list(messages)
    for i in {0, len(out) - 1}:
        m = out[i]
        if isinstance(m.content, str) and m.content:
            block = {"type": "text", "text": m.content, "cache_control": {"type": "ephemeral"}}
            out[i] = m.model_copy(update={"content": [block]})
    return out


def _msg_record(m) -> dict:
    rec = {"role": m.type, "content": m.content if isinstance(m.content, str) else str(m.content)}
    if getattr(m, "tool_calls", None):
        rec["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
    return rec


def run_llm(ep: Episode, model_cfg: dict, chat_model=None) -> tuple[dict, int, str | None, list | None]:
    tools = [t for t in TOOL_SCHEMAS if t["name"] in ep.tool_names]
    llm = (chat_model or make_chat_model(model_cfg)).bind_tools(tools)
    messages = [SystemMessage(ep.system_prompt), HumanMessage(ep.first_message)]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
             "model_version": None}
    cache = model_cfg.get("prompt_cache", True)
    provider = model_cfg.get("provider") if cache else None
    nudges, steps = 0, 0
    for steps in range(1, ep.limits["max_steps"] + 1):
        ai = _invoke_with_retry(llm, _cache_marked(messages, provider))
        messages.append(ai)
        meta = getattr(ai, "usage_metadata", None) or {}
        usage["input_tokens"] += meta.get("input_tokens", 0)
        usage["output_tokens"] += meta.get("output_tokens", 0)
        details = meta.get("input_token_details") or {}
        usage["cache_read_tokens"] += details.get("cache_read", 0) or 0
        usage["cache_write_tokens"] += details.get("cache_creation", 0) or 0
        rmeta = getattr(ai, "response_metadata", None) or {}
        usage["model_version"] = usage["model_version"] or rmeta.get("model_name") or rmeta.get("model")
        if not ai.tool_calls:
            if nudges >= 2:
                break
            nudges += 1
            messages.append(HumanMessage("Call the recommend tool now with your final ranked shortlist."))
            continue
        for tc in ai.tool_calls:
            name, args = tc["name"], tc["args"]
            if name not in ep.tool_names:
                out = f"The tool {name} is unavailable here. Use: {', '.join(ep.tool_names)}."
            elif name == "web_search":
                out = ep.search(args.get("query", ""))
            elif name == "fetch_page":
                out = ep.fetch(args.get("url", ""))
            elif name == "recommend":
                out = ep.recommend(args.get("recommendations", []))
            else:
                out = f"Unknown tool {name}."
            messages.append(ToolMessage(content=out, tool_call_id=tc["id"]))
        if ep.recommendations is not None:
            break
    error = None if ep.recommendations is not None else "finished without calling recommend"
    transcript = [_msg_record(m) for m in messages] if ep.limits.get("save_transcripts") else None
    return usage, steps, error, transcript
