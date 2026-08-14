"""
House agent — answers questions about a specific house.
Uses LangGraph ReAct pattern with house-bound tools.

General (non-house-specific) questions are redirected to the General Chat —
per the system prompt below, this is the agent's own call, not a pre-filter.
"""
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import settings
from agents.tools import make_house_tools


SYSTEM_PROMPT = """You are a helpful real estate assistant analyzing a specific home.

Your tools give you access to:
- Structured house details (price, beds, baths, sqft, scores)
- FEMA National Risk Index data for the census tract
- A stored property description (injected below if available)
- Price estimation using comparable listings and sold homes

Guidelines:
- Always use your tools before answering questions that require data
- For price estimation, use estimate_price_with_code first, then supplement with your analysis
- If the user provides a description (e.g. copy-pasted from Redfin/Zillow), acknowledge it
  and confirm it has been saved. Do NOT call a tool to save it — that happens automatically.
- If the user asks a GENERAL question that isn't about this specific house — metro-area
  risk rankings, national trends, comparing cities, or questions about OTHER houses — don't
  try to answer it with these tools; they're scoped to this one property. Instead reply
  exactly: "That sounds like a general question about multiple cities or national trends.
  Please use the **General Chat** button to ask that — it has access to the full MSA,
  census, and NRI datasets for broad comparisons!"
- Be concise but thorough. Format numbers clearly (e.g., $450,000 not 450000).
- When discussing risk, explain what the NRI scores mean in plain language.
"""


def _build_system_prompt(house_id: str) -> str:
    """
    Build the system prompt, injecting the stored description directly so the
    agent can answer questions about it even in a fresh session with no history.
    This avoids the agent having to call search_house_documents every turn.
    """
    import db.vector_store as vs
    desc = vs.get_description(house_id)
    if desc:
        return (SYSTEM_PROMPT +
                f"\n\n━━━ STORED PROPERTY DESCRIPTION ━━━\n{desc['text']}\n"
                "━━━ END DESCRIPTION ━━━\n"
                "Use the above description to answer questions about this property. "
                "You do NOT need to call search_house_documents to access it.")
    return SYSTEM_PROMPT


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x) + list(y)]
    house_id: str


def build_house_agent(house_id: str):
    """Build and return a compiled LangGraph house agent for the given house."""

    tools: list[BaseTool] = make_house_tools(house_id)
    tool_node = ToolNode(tools)
    # For 'llama-server -hf DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M -ngl 999 -c 28672 -fa on --cache-type-k q8_0 --cache-type-v q8_0'
    llm = ChatOpenAI(
        base_url=settings.llama_server_base_url,
        api_key="not-needed",  # llama-server doesn't check the key, but LangChain requires a non-empty string
        model=settings.llama_server_model,
        temperature=0.0
    ).bind_tools(tools)
    # For 'ollama run --model llama3.1:8b'
    # llm = ChatOllama(
    #     base_url=settings.ollama_base_url,
    #     model=settings.ollama_model,
    #     temperature=0.0,
    # ).bind_tools(tools)

    def agent_node(state: AgentState):
        system_prompt = _build_system_prompt(state["house_id"])
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
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


def run_house_chat(house_id: str, message: str,
                   history: list[dict] = None) -> tuple[str, list[dict]]:
    """
    Run one turn of house-specific chat.
    Returns (response_text, updated_history).
    history is a list of {"role": "user"|"assistant", "content": "..."}.
    """
    agent = build_house_agent(house_id)

    # Build LangChain message history
    lc_messages: list[BaseMessage] = []
    for h in (history or []):
        if h["role"] == "user":
            lc_messages.append(HumanMessage(content=h["content"]))
        else:
            lc_messages.append(AIMessage(content=h["content"]))
    lc_messages.append(HumanMessage(content=message))

    result = agent.invoke({"messages": lc_messages, "house_id": house_id})

    # Extract last AI message
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "I couldn't generate a response."

    updated_history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return reply, updated_history
