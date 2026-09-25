#!/usr/bin/env python3
"""Tiny ReAct-style tool-calling agent loop.

Thought (planner) -> Action (tool call) -> Observation -> ... -> Final Answer.
No LLM API: the planner is heuristic; the answer synthesizer is template-based.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from planner import Plan, PlannedCall, plan_tools
from tools import ToolSpec, execute_tool, build_registry


@dataclass
class Step:
    thought: str
    action: str | None
    action_args: dict[str, Any] | None
    observation: dict[str, Any] | None
    ok: bool = True


@dataclass
class AgentResult:
    query: str
    plan: Plan
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    tool_successes: int = 0
    tool_failures: int = 0
    n_steps: int = 0


def _synthesize_answer(query: str, steps: list[Step]) -> str:
    """Template final answer from successful observations."""
    bits: list[str] = []
    for st in steps:
        if not st.ok or not st.observation:
            continue
        obs = st.observation
        if not obs.get("ok"):
            bits.append(f"Tool {st.action} failed: {obs.get('error')}")
            continue
        result = obs.get("result") or {}
        name = st.action
        if name == "calculator":
            bits.append(f"Calculation: {result.get('display', result)}")
        elif name == "weather":
            if "error" in result:
                bits.append(f"Weather: unknown city ({result.get('city')}).")
            else:
                bits.append(
                    f"Weather in {result.get('city')}: {result.get('temp_c')}°C, "
                    f"{result.get('condition')}, humidity {result.get('humidity')}%."
                )
        elif name == "datetime_info":
            kind = result.get("kind")
            if kind == "weekday":
                bits.append(f"Today is {result.get('weekday')} ({result.get('date')}).")
            elif kind == "tomorrow":
                bits.append(f"Tomorrow is {result.get('weekday')} ({result.get('date')}).")
            elif kind == "year":
                bits.append(f"The year is {result.get('year')}.")
            else:
                bits.append(f"Current datetime (anchor): {result.get('datetime', result.get('date'))}.")
        elif name == "kb_lookup":
            if result.get("hit"):
                bits.append(f"Fact ({result.get('topic')}): {result.get('fact')}")
            else:
                bits.append(f"No KB hit for '{result.get('topic')}'.")
        elif name == "string_ops":
            bits.append(f"String {result.get('operation')}: {result.get('result')}")
        else:
            bits.append(f"{name}: {result}")
    if not bits:
        return (
            f"I could not select a tool for: {query!r}. "
            "Try asking about math, weather, dates, or AI/ML topics in the KB."
        )
    return " ".join(bits)


def run_agent(
    query: str,
    registry: dict[str, ToolSpec] | None = None,
    *,
    max_steps: int = 4,
) -> AgentResult:
    """Execute a ReAct-style loop for one user query."""
    registry = registry or build_registry()
    plan = plan_tools(query, registry)
    result = AgentResult(query=query, plan=plan)

    if not plan.calls:
        thought = "No tool scored above threshold; answer without tools."
        result.steps.append(Step(thought=thought, action=None, action_args=None, observation=None, ok=True))
        result.answer = _synthesize_answer(query, result.steps)
        result.n_steps = len(result.steps)
        return result

    for i, call in enumerate(plan.calls[:max_steps]):
        thought = (
            f"Step {i + 1}: tool '{call.tool}' looks useful "
            f"(score={call.score:.1f}, {call.reason})."
        )
        obs = execute_tool(registry, call.tool, call.args)
        ok = bool(obs.get("ok"))
        result.steps.append(
            Step(
                thought=thought,
                action=call.tool,
                action_args=dict(call.args),
                observation=obs,
                ok=ok,
            )
        )
        result.tools_called.append(call.tool)
        if ok:
            result.tool_successes += 1
        else:
            result.tool_failures += 1

    result.answer = _synthesize_answer(query, result.steps)
    result.n_steps = len(result.steps)
    return result


def format_trace(result: AgentResult) -> str:
    lines = [f"Query: {result.query}", f"Plan: {[c.tool for c in result.plan.calls]}"]
    for i, st in enumerate(result.steps, 1):
        lines.append(f"  Thought {i}: {st.thought}")
        if st.action:
            lines.append(f"  Action  {i}: {st.action}({st.action_args})")
            lines.append(f"  Obs     {i}: {st.observation}")
    lines.append(f"Final Answer: {result.answer}")
    return "\n".join(lines)
