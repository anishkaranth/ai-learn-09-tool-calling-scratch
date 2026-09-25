#!/usr/bin/env python3
"""World state + tool handlers (calculator, weather, KB, strings).

No paid APIs: calculator, datetime, fake weather, knowledge-base lookup,
and a simple string helper. Educational stand-in for LLM function calling.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
}

def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):  # py<3.8 compat
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCS:
            raise ValueError(f"function not allowed: {name}")
        args = [_eval_node(a) for a in node.args]
        return float(_FUNCS[name](*args))
    if isinstance(node, ast.Name) and node.id in ("pi", "e"):
        return float(getattr(math, node.id))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")

def safe_calc(expression: str) -> float:
    """Evaluate a math expression with a whitelist AST walker."""
    expr = expression.strip().replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return float(_eval_node(tree))

FAKE_WEATHER: dict[str, dict[str, Any]] = {
    "bangalore": {"temp_c": 28, "condition": "partly cloudy", "humidity": 65},
    "bengaluru": {"temp_c": 28, "condition": "partly cloudy", "humidity": 65},
    "mumbai": {"temp_c": 31, "condition": "humid", "humidity": 78},
    "delhi": {"temp_c": 34, "condition": "haze", "humidity": 40},
    "london": {"temp_c": 14, "condition": "light rain", "humidity": 82},
    "tokyo": {"temp_c": 22, "condition": "clear", "humidity": 55},
    "new york": {"temp_c": 18, "condition": "overcast", "humidity": 60},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 75},
}

KNOWLEDGE_BASE: dict[str, str] = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "numpy": "NumPy is the fundamental package for scientific computing with Python arrays.",
    "rag": "RAG (Retrieval-Augmented Generation) retrieves relevant documents then generates grounded answers.",
    "react": "ReAct interleaves Reasoning traces with Actions (tool calls) and Observations.",
    "lora": "LoRA (Low-Rank Adaptation) fine-tunes large models by updating small low-rank matrices.",
    "transformer": "Transformers use self-attention to model token relationships without recurrence.",
    "embedding": "An embedding is a dense vector representation of text, images, or other modalities.",
    "tool calling": "Tool calling (function calling) lets a model invoke external tools via structured schemas.",
}

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-Schema-like
    keywords: list[str] = field(default_factory=list)
    handler: Callable[..., Any] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

def _tool_calculator(expression: str) -> dict[str, Any]:
    value = safe_calc(expression)
    return {"expression": expression, "value": value, "display": f"{expression} = {value}"}

def _tool_datetime(query: str = "now") -> dict[str, Any]:
    """Deterministic 'now' anchored to 2026-09-25 for reproducible smoke tests."""
    now = datetime(2026, 9, 25, 16, 0, 0)
    q = (query or "now").strip().lower()
    if "tomorrow" in q:
        d = now.date() + timedelta(days=1)
        return {"query": query, "date": d.isoformat(), "weekday": d.strftime("%A"), "kind": "tomorrow"}
    if "yesterday" in q:
        d = now.date() - timedelta(days=1)
        return {"query": query, "date": d.isoformat(), "weekday": d.strftime("%A"), "kind": "yesterday"}
    if "weekday" in q or "day of week" in q or "what day" in q:
        return {
            "query": query,
            "date": now.date().isoformat(),
            "weekday": now.strftime("%A"),
            "kind": "weekday",
        }
    if "year" in q:
        return {"query": query, "year": now.year, "kind": "year"}
    return {
        "query": query,
        "datetime": now.isoformat(sep=" "),
        "date": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "kind": "now",
    }

def _tool_weather(city: str) -> dict[str, Any]:
    key = city.strip().lower()
    if key not in FAKE_WEATHER:
        for k, v in FAKE_WEATHER.items():
            if key in k or k in key:
                return {"city": k.title(), **v, "source": "fake_weather"}
        return {"city": city, "error": "unknown_city", "known": sorted(FAKE_WEATHER)}
    return {"city": key.title(), **FAKE_WEATHER[key], "source": "fake_weather"}

def _tool_kb_lookup(topic: str) -> dict[str, Any]:
    key = topic.strip().lower()
    if key in KNOWLEDGE_BASE:
        return {"topic": key, "fact": KNOWLEDGE_BASE[key], "hit": True}
    best, best_score = None, 0.0
    tokens = set(re.findall(r"[a-z0-9]+", key))
    for k, fact in KNOWLEDGE_BASE.items():
        kt = set(re.findall(r"[a-z0-9]+", k))
        score = len(tokens & kt) / max(1, len(tokens | kt))
        if score > best_score:
            best, best_score = k, score
    if best is not None and best_score >= 0.3:
        return {"topic": best, "fact": KNOWLEDGE_BASE[best], "hit": True, "score": round(best_score, 3)}
    return {"topic": topic, "fact": None, "hit": False, "hint": "try: " + ", ".join(sorted(KNOWLEDGE_BASE))}

def _tool_string_ops(text: str, operation: str = "upper") -> dict[str, Any]:
    op = (operation or "upper").strip().lower()
    if op == "upper":
        out = text.upper()
    elif op == "lower":
        out = text.lower()
    elif op == "title":
        out = text.title()
    elif op == "reverse":
        out = text[::-1]
    elif op == "word_count":
        out = str(len(re.findall(r"\b\w+\b", text)))
    elif op == "char_count":
        out = str(len(text))
    else:
        return {"error": f"unknown operation: {op}", "allowed": ["upper", "lower", "title", "reverse", "word_count", "char_count"]}
    return {"text": text, "operation": op, "result": out}

def build_registry() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec(
            name="calculator",
            description="Evaluate a math expression ( +, -, *, /, **, sqrt, etc. ).",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
            keywords=["calculate", "compute", "math", "sum", "product", "multiply", "divide",
                      "plus", "minus", "sqrt", "square", "percentage", "percent", "+", "-", "*", "/"],
            handler=_tool_calculator,
        ),
        ToolSpec(
            name="datetime_info",
            description="Return date/time facts (today, tomorrow, weekday, year). Deterministic anchor.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "e.g. 'now', 'tomorrow', 'weekday'"},
                },
                "required": [],
            },
            keywords=["date", "time", "today", "tomorrow", "yesterday", "weekday", "day of week", "year", "clock"],
            handler=_tool_datetime,
        ),
        ToolSpec(
            name="weather",
            description="Fake weather lookup for a small set of cities.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
            keywords=["weather", "temperature", "temp", "forecast", "humidity", "rain", "hot", "cold"],
            handler=_tool_weather,
        ),
        ToolSpec(
            name="kb_lookup",
            description="Look up a short fact from a toy AI/ML knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to look up"},
                },
                "required": ["topic"],
            },
            keywords=["what is", "who is", "define", "explain", "meaning of", "tell me about",
                      "python", "numpy", "rag", "react", "lora", "transformer", "embedding", "tool calling"],
            handler=_tool_kb_lookup,
        ),
        ToolSpec(
            name="string_ops",
            description="Simple string transforms: upper/lower/title/reverse/word_count/char_count.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "operation": {"type": "string", "enum": ["upper", "lower", "title", "reverse", "word_count", "char_count"]},
                },
                "required": ["text"],
            },
            keywords=["uppercase", "lowercase", "reverse", "word count", "characters", "title case"],
            handler=_tool_string_ops,
        ),
    ]
    return {s.name: s for s in specs}

def execute_tool(registry: dict[str, ToolSpec], name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name not in registry:
        return {"ok": False, "error": f"unknown tool: {name}"}
    spec = registry[name]
    assert spec.handler is not None
    try:
        props = set(spec.parameters.get("properties", {}).keys())
        filtered = {k: v for k, v in args.items() if k in props}
        result = spec.handler(**filtered)
        return {"ok": True, "tool": name, "args": filtered, "result": result}
    except Exception as exc:  # noqa: BLE001 — educational agent should surface tool errors
        return {"ok": False, "tool": name, "args": args, "error": str(exc)}

def list_schemas(registry: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    return [s.schema() for s in registry.values()]
