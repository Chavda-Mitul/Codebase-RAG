"""LangGraph agentic pipeline with tool-calling and conversation memory."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.agents.state import AgentState
from src.agents.memory import get_memory
from src.agents.tools import (
    search_codebase, run_code_snippet,
    generate_mermaid_diagram, suggest_refactors,
)
from src.qa.prompts import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Structured output schema for tool planning
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    tool_name: Literal[
        "search_codebase",
        "run_code_snippet",
        "generate_mermaid_diagram",
        "suggest_refactors",
    ]
    argument: str = Field(description="Primary argument for the tool")


class ToolPlan(BaseModel):
    reasoning: str = Field(description="Why these tools (or none) are needed")
    calls: list[ToolCall] = Field(description="Ordered list of tool calls; empty if no tools needed")


_PLAN_SYSTEM = """You are an intelligent code analysis assistant with access to tools.

Available tools:
- search_codebase: Search the codebase knowledge graph for relevant code entities. Use for most questions.
- run_code_snippet: Execute a Python snippet and return stdout/stderr. Use only when asked to run or test code.
- generate_mermaid_diagram: Generate a Mermaid diagram. Use when asked to visualize, diagram, or map relationships between entities. Argument: comma-separated entity names.
- suggest_refactors: Get refactoring suggestions for a function. Use when asked to improve or refactor. Argument: function name.

Rules:
- Always call search_codebase first when you need codebase context.
- Return an empty calls list ONLY if the question can be fully answered from conversation history alone.
- Do not call the same tool twice with the same argument.
"""

_PLAN_USER = """Conversation history:
{history}

Current question: {question}

Decide which tools to invoke."""

_GENERATE_SYSTEM = SYSTEM_PROMPT + "\nYou also have access to tool outputs below. Incorporate them in your answer."

_GENERATE_USER = """Conversation history:
{history}

Tool results:
{tool_results}

Codebase context:
{context}

Question: {question}

Provide a comprehensive answer. If a Mermaid diagram was generated, mention it is shown in the trace panel."""


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_agent_nodes(client: Neo4jClient):
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    planner = llm.with_structured_output(ToolPlan)

    def inject_memory(state: AgentState) -> dict:
        memory = get_memory(state["session_id"])
        return {"history": memory.get_history_text()}

    def plan_tools(state: AgentState) -> dict:
        plan: ToolPlan = planner.invoke([
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": _PLAN_USER.format(
                history=state.get("history") or "(no history)",
                question=state["question"],
            )},
        ])
        return {"tool_plan": [c.model_dump() for c in plan.calls]}

    def execute_tools(state: AgentState) -> dict:
        tool_calls: list[dict] = []
        documents = list(state.get("documents") or [])

        for call in state.get("tool_plan") or []:
            tool_name = call["tool_name"]
            argument = call["argument"]
            try:
                if tool_name == "search_codebase":
                    text, nodes = search_codebase(argument, client)
                    output = text
                    documents.extend(n for n in nodes if n.node_id not in {d.node_id for d in documents})
                elif tool_name == "run_code_snippet":
                    output = run_code_snippet(argument)
                elif tool_name == "generate_mermaid_diagram":
                    output = generate_mermaid_diagram(argument, client)
                elif tool_name == "suggest_refactors":
                    output = suggest_refactors(argument, client, llm)
                else:
                    output = f"[Error] Unknown tool: {tool_name}"
            except Exception as e:
                output = f"[Error] {tool_name} failed: {e}"

            tool_calls.append({"tool": tool_name, "input": argument, "output": output})

        return {"tool_calls": tool_calls, "documents": documents}

    def generate_with_tools(state: AgentState) -> dict:
        from src.qa.chain import _format_context
        tool_results = "\n\n".join(
            f"[{tc['tool']}({tc['input'][:60]})]\n{tc['output'][:800]}"
            for tc in (state.get("tool_calls") or [])
        ) or "(no tool results)"

        context = _format_context(state.get("documents") or [])

        response = llm.invoke([
            {"role": "system", "content": _GENERATE_SYSTEM},
            {"role": "user", "content": _GENERATE_USER.format(
                history=state.get("history") or "(no history)",
                tool_results=tool_results,
                context=context or "(no context)",
                question=state["question"],
            )},
        ])
        return {"answer": response.content}

    def save_memory(state: AgentState) -> dict:
        memory = get_memory(state["session_id"])
        memory.add_turn(state["question"], state["answer"])
        return {}

    return {
        "inject_memory": inject_memory,
        "plan_tools": plan_tools,
        "execute_tools": execute_tools,
        "generate_with_tools": generate_with_tools,
        "save_memory": save_memory,
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def decide_after_planning(state: AgentState) -> str:
    return "execute_tools" if state.get("tool_plan") else "generate_with_tools"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph(client: Neo4jClient):
    nodes = make_agent_nodes(client)
    graph = StateGraph(AgentState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("inject_memory")
    graph.add_edge("inject_memory", "plan_tools")
    graph.add_conditional_edges(
        "plan_tools",
        decide_after_planning,
        {"execute_tools": "execute_tools", "generate_with_tools": "generate_with_tools"},
    )
    graph.add_edge("execute_tools", "generate_with_tools")
    graph.add_edge("generate_with_tools", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


def run_agent(question: str, client: Neo4jClient, session_id: str) -> AgentState:
    """Run the agentic pipeline and return the final state."""
    pipeline = build_agent_graph(client)
    initial: AgentState = {
        "question": question,
        "session_id": session_id,
        "history": "",
        "tool_plan": [],
        "tool_calls": [],
        "documents": [],
        "answer": "",
        "iteration": 0,
    }
    return pipeline.invoke(initial)
