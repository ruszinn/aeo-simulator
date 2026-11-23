"""A fake chat model that follows a fixed script, for testing the LLM loop
without API keys: search with the brief's text (or use the results already in
the message, in forced-exposure mode), read the first three result sites if a
page-reading tool is available, and recommend them in order. Select it with
`provider: scripted`."""

from __future__ import annotations

import re
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

URL_RE = re.compile(r"https://[a-z0-9.-]+\.example/[^\s]*")


class ScriptedChatModel(BaseChatModel):
    tool_names: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tool_names": [t["name"] for t in tools]})

    def _call(self, name: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}],
                         usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                         response_metadata={"model_name": "scripted-test-1"})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        first = next(m for m in messages if isinstance(m, HumanMessage))
        text = first.content if isinstance(first.content, str) else str(first.content)
        searching = "web_search" in self.tool_names
        if searching and not tool_msgs:
            msg = self._call("web_search", {"query": text})
        else:
            results = tool_msgs[0].content if searching else text
            homes, seen = [], set()
            for u in URL_RE.findall(results):
                dom = u.split("/")[2]
                if dom not in seen and "directory" not in dom:
                    seen.add(dom)
                    homes.append(f"https://{dom}/")
            n_fetched = len(tool_msgs) - (1 if searching else 0)
            if "fetch_page" in self.tool_names and n_fetched < min(3, len(homes)):
                msg = self._call("fetch_page", {"url": homes[n_fetched]})
            else:
                recs = [{"company_name": h.split("/")[2], "website": h, "reason": "scripted"} for h in homes[:3]]
                msg = self._call("recommend", {"recommendations": recs})
        return ChatResult(generations=[ChatGeneration(message=msg)])
