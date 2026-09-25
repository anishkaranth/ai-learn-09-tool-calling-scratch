#!/usr/bin/env python3
"""Labeled evaluation tasks for the toy tool-calling agent."""
from __future__ import annotations

from typing import Any


# Each task:
#   id, query, gold_tools (ordered preferred set), gold_contains (substrings that
#   must appear in the final answer — case-insensitive), optional gold_args_check
TASKS: list[dict[str, Any]] = [
    # --- single-tool: calculator ---
    {
        "id": "calc_add",
        "query": "Calculate 17 + 25",
        "gold_tools": ["calculator"],
        "gold_contains": ["42"],
        "multi_step": False,
    },
    {
        "id": "calc_mul",
        "query": "What is 12 times 8?",
        "gold_tools": ["calculator"],
        "gold_contains": ["96"],
        "multi_step": False,
    },
    {
        "id": "calc_div",
        "query": "Compute 144 / 12",
        "gold_tools": ["calculator"],
        "gold_contains": ["12"],
        "multi_step": False,
    },
    {
        "id": "calc_sqrt",
        "query": "What is sqrt(81)?",
        "gold_tools": ["calculator"],
        "gold_contains": ["9"],
        "multi_step": False,
    },
    {
        "id": "calc_pct",
        "query": "Calculate 15% of 200",
        "gold_tools": ["calculator"],
        "gold_contains": ["30"],
        "multi_step": False,
    },
    # --- weather ---
    {
        "id": "wx_blr",
        "query": "What is the weather in Bangalore?",
        "gold_tools": ["weather"],
        "gold_contains": ["28", "partly cloudy"],
        "multi_step": False,
    },
    {
        "id": "wx_london",
        "query": "Tell me the temperature in London",
        "gold_tools": ["weather"],
        "gold_contains": ["14", "light rain"],
        "multi_step": False,
    },
    {
        "id": "wx_tokyo",
        "query": "Weather forecast for Tokyo please",
        "gold_tools": ["weather"],
        "gold_contains": ["22", "clear"],
        "multi_step": False,
    },
    # --- datetime ---
    {
        "id": "dt_weekday",
        "query": "What day of the week is today?",
        "gold_tools": ["datetime_info"],
        "gold_contains": ["friday"],
        "multi_step": False,
    },
    {
        "id": "dt_tomorrow",
        "query": "What date is tomorrow?",
        "gold_tools": ["datetime_info"],
        "gold_contains": ["2026-09-26", "saturday"],
        "multi_step": False,
    },
    {
        "id": "dt_year",
        "query": "What year is it right now?",
        "gold_tools": ["datetime_info"],
        "gold_contains": ["2026"],
        "multi_step": False,
    },
    # --- KB ---
    {
        "id": "kb_rag",
        "query": "What is RAG?",
        "gold_tools": ["kb_lookup"],
        "gold_contains": ["retrieval"],
        "multi_step": False,
    },
    {
        "id": "kb_lora",
        "query": "Explain LoRA",
        "gold_tools": ["kb_lookup"],
        "gold_contains": ["low-rank"],
        "multi_step": False,
    },
    {
        "id": "kb_react",
        "query": "Tell me about ReAct",
        "gold_tools": ["kb_lookup"],
        "gold_contains": ["reasoning", "actions"],
        "multi_step": False,
    },
    {
        "id": "kb_python",
        "query": "Who created Python?",
        "gold_tools": ["kb_lookup"],
        "gold_contains": ["guido"],
        "multi_step": False,
    },
    # --- string ops ---
    {
        "id": "str_upper",
        "query": "Convert the word 'neuron' to uppercase",
        "gold_tools": ["string_ops"],
        "gold_contains": ["NEURON"],
        "multi_step": False,
    },
    {
        "id": "str_reverse",
        "query": "Reverse the string 'agent'",
        "gold_tools": ["string_ops"],
        "gold_contains": ["tnega"],
        "multi_step": False,
    },
    {
        "id": "str_words",
        "query": "What is the word count of 'tool calling agent'?",
        "gold_tools": ["string_ops"],
        "gold_contains": ["3"],
        "multi_step": False,
    },
    # --- multi-step / compound ---
    {
        "id": "multi_calc_wx",
        "query": "Calculate 9 * 7 and tell me the weather in Mumbai",
        "gold_tools": ["calculator", "weather"],
        "gold_contains": ["63", "31", "humid"],
        "multi_step": True,
    },
    {
        "id": "multi_kb_dt",
        "query": "What is an embedding and what day is today?",
        "gold_tools": ["kb_lookup", "datetime_info"],
        "gold_contains": ["vector", "friday"],
        "multi_step": True,
    },
    {
        "id": "multi_wx_calc",
        "query": "Weather in Delhi then compute 100 - 37",
        "gold_tools": ["weather", "calculator"],
        "gold_contains": ["34", "haze", "63"],
        "multi_step": True,
    },
    {
        "id": "multi_str_kb",
        "query": "Uppercase the word 'lora' and explain LoRA",
        "gold_tools": ["string_ops", "kb_lookup"],
        "gold_contains": ["LORA", "low-rank"],
        "multi_step": True,
    },
    {
        "id": "multi_three",
        "query": "What is sqrt(49), weather in London, and what year is it?",
        "gold_tools": ["calculator", "weather", "datetime_info"],
        "gold_contains": ["7", "14", "2026"],
        "multi_step": True,
    },
]


def answer_matches(answer: str, gold_contains: list[str]) -> bool:
    a = answer.lower()
    return all(g.lower() in a for g in gold_contains)


def tool_selection_ok(called: list[str], gold_tools: list[str]) -> bool:
    """True if every gold tool was called (order-insensitive; extras ok)."""
    return set(gold_tools).issubset(set(called))
