"""
General agent — answers broad questions about metro areas, national risk,
portfolio-level analysis across all houses, etc.
"""
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import settings
from agents.tools import GENERAL_TOOLS
import db.schema_catalog as schema


SYSTEM_PROMPT = """You are a real estate and risk analyst assistant backed by a local DuckDB database.

A live snapshot of what's actually loaded (row counts per table) is included
below — you don't need to re-check it unless you suspect it changed mid-conversation.

Ground rules:
1. Every number, ranking, or comparison you state must come from a tool call
   made in this conversation — never estimate or recall a figure from memory.
2. If a table is empty or a query returns 0 rows, say so plainly. Do not fill
   the gap with a plausible-sounding invented answer.
3. For ANY question that requires analytical data from DuckDB, use the
   `query_database` tool instead of writing SQL yourself. This applies to
   houses, snapshots, sales, NRI, Census, crime, bike, and every other
   agent-visible dataset. `query_database` is backed by the shared SQL Code
   Agent, which generates the SQL from the live table metadata and documented
   relationships. Put the user's analytical request in `request`; when your
   reasoning has explicit requirements or a multi-step plan, pass those in
   `requirements` and `plan`.
4. Treat the `query_database` result as evidence, not as the final answer.
   Inspect the generated SQL and returned rows, reconcile them with the user's
   intent, and continue your reasoning/planning before responding. If the
   question needs another query after seeing the first result, call
   `query_database` again with the updated requirements or plan.
5. If `query_database` errors or returns 0 rows, read the message carefully.
   Revise the request/requirements/plan and retry when appropriate, or explain
   the data limitation. Never fabricate a result.

`get_database_schema` remains available when you need to inspect the live schema
directly, but do not bypass `query_database` by writing SQL yourself for an
analytical data question.

Formatting: dollar amounts as $1,234,567; scores/percentages to one decimal
place; use a markdown table when comparing more than a couple of rows.
"""


def _get_data_availability_context() -> str:
    """Live data-availability summary, prepended to every system message.
    Shares its table list and counting logic with the check_data_availability
    tool (both read db/schema_catalog.py) so the two can't drift apart."""
    report, _ = schema.availability_report()
    return f"[LIVE DATABASE STATUS]\n{report}\n[END STATUS]\n"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x) + list(y)]


def build_general_agent():
    tool_node = ToolNode(GENERAL_TOOLS)
    # For 'llama-server -hf DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M -ngl 999 -c 28672 -fa on --cache-type-k q8_0 --cache-type-v q8_0'
    llm = ChatOpenAI(
        base_url=settings.llama_server_base_url,
        api_key="not-needed",  # llama-server doesn't check the key, but LangChain requires a non-empty string
        model=settings.llama_server_model,
        temperature=0.0
    ).bind_tools(GENERAL_TOOLS)
    # For 'ollama run --model llama3.1:8b'
    # llm = ChatOllama(
    #     base_url=settings.ollama_base_url,
    #     model=settings.ollama_model,
    #     temperature=0.0,
    # ).bind_tools(GENERAL_TOOLS)

    def agent_node(state: AgentState):
        # Inject live data availability into every system message so the model
        # always knows what's loaded without needing to call the tool first
        availability = _get_data_availability_context()
        system = SystemMessage(content=f"{availability}\n{SYSTEM_PROMPT}")
        messages = [system] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    def router(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# Singleton — rebuilt if you call invalidate_general_agent()
_general_agent = None


def get_general_agent():
    global _general_agent
    if _general_agent is None:
        _general_agent = build_general_agent()
    return _general_agent


def invalidate_general_agent():
    """Call after loading new data so the agent picks up updated availability."""
    global _general_agent
    _general_agent = None


def run_general_chat(message: str, history: list[dict] = None) -> tuple[str, list[dict]]:
    agent = get_general_agent()

    lc_messages: list[BaseMessage] = []
    for h in (history or []):
        if h["role"] == "user":
            lc_messages.append(HumanMessage(content=h["content"]))
        else:
            lc_messages.append(AIMessage(content=h["content"]))
    lc_messages.append(HumanMessage(content=message))

    result = agent.invoke({"messages": lc_messages})
    all_messages = list(result["messages"])

    ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]
    raw_reply = ai_messages[-1].content if ai_messages else "I couldn't generate a response."

    # ── Validate: replace hallucinated data tables with honest errors ─────
    from agents.response_validator import validate_response
    reply = validate_response(raw_reply, all_messages, strict=True)

    if reply != raw_reply:
        import logging
        logging.getLogger(__name__).warning(
            "Hallucination detected and blocked in general agent response."
        )

    updated_history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return reply, updated_history

