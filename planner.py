#!/usr/bin/env python3
"""Heuristic tool planner — stands in for an LLM function-calling head.

Scores each registered tool against the user query with keyword overlap +
light regex argument extractors, then returns an ordered plan of tool calls.
Supports multi-tool plans for compound questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tools import ToolSpec, FAKE_WEATHER, KNOWLEDGE_BASE

@dataclass
class PlannedCall:
    tool: str
    args: dict[str, Any]
    score: float
    reason: str

@dataclass
class Plan:
    query: str
    calls: list[PlannedCall] = field(default_factory=list)
    needs_tools: bool = True

_MATH_RE = re.compile(
    r"(?:"
    r"[\d.]+\s*(?:[\+\-\*/\^]|plus|minus|times|multiplied by|divided by|over)\s*[\d.]+"
    r"|sqrt\s*\([^)]+\)"
    r"|\d+\s*%\s*of\s*[\d.]+"
    r"|[\d\.\s\+\-\*/\^\(\)]{3,}"
    r")",
    re.I,
)

def _normalize_math(raw: str) -> str | None:
    s = raw.strip()
    s = re.sub(r"\bplus\b", "+", s, flags=re.I)
    s = re.sub(r"\bminus\b", "-", s, flags=re.I)
    s = re.sub(r"\btimes\b|\bmultiplied by\b", "*", s, flags=re.I)
    s = re.sub(r"\bdivided by\b|\bover\b", "/", s, flags=re.I)
    m = re.search(r"([\d.]+)\s*%\s*of\s*([\d.]+)", s, re.I)
    if m:
        return f"({m.group(1)}/100)*{m.group(2)}"
    cleaned = re.sub(r"[^0-9eE\.\+\-\*/\^\(\)\s,]", "", s)
    cleaned = cleaned.replace("^", "**").strip()
    if re.search(r"\d", cleaned) and re.search(r"[\+\-\*/]|sqrt|\*\*", cleaned):
        return cleaned
    m2 = re.search(r"sqrt\s*\(\s*([\d.]+)\s*\)", s, re.I)
    if m2:
        return f"sqrt({m2.group(1)})"
    return None

def extract_math(query: str) -> str | None:
    for m in _MATH_RE.finditer(query):
        expr = _normalize_math(m.group(0))
        if expr:
            return expr
    return _normalize_math(query)

def extract_city(query: str) -> str | None:
    q = query.lower()
    for city in sorted(FAKE_WEATHER.keys(), key=len, reverse=True):
        if city in q:
            return city
    m = re.search(r"(?:weather|temperature|temp|forecast)\s+(?:in|for|at)\s+([a-zA-Z][a-zA-Z\s]+?)(?:\?|$|,|\.| and | then )", query, re.I)
    if m:
        return m.group(1).strip()
    return None

def extract_kb_topic(query: str) -> str | None:
    q = query.lower()
    for topic in sorted(KNOWLEDGE_BASE.keys(), key=len, reverse=True):
        if topic in q:
            return topic
    m = re.search(
        r"(?:what is|what's|who is|define|explain|tell me about|meaning of)\s+(.+?)(?:\?|$)",
        query,
        re.I,
    )
    if m:
        return m.group(1).strip().rstrip(".")
    return None

def extract_string_args(query: str) -> dict[str, Any] | None:
    q = query.lower()
    op = None
    if "uppercase" in q or "upper case" in q or "to upper" in q:
        op = "upper"
    elif "lowercase" in q or "lower case" in q or "to lower" in q:
        op = "lower"
    elif "title case" in q or "capitalize" in q:
        op = "title"
    elif "reverse" in q:
        op = "reverse"
    elif "word count" in q or "how many words" in q:
        op = "word_count"
    elif "character count" in q or "how many characters" in q or "char count" in q:
        op = "char_count"
    if op is None:
        return None
    m = re.search(r"['\"]([^'\"]+)['\"]", query)
    if m:
        return {"text": m.group(1), "operation": op}
    m2 = re.search(r"(?:word|string|text|phrase)\s+['\"]?([A-Za-z0-9_\- ]+)['\"]?", query, re.I)
    if m2:
        return {"text": m2.group(1).strip(), "operation": op}
    return {"text": "hello", "operation": op}  # weak fallback

def extract_datetime_query(query: str) -> str:
    q = query.lower()
    if "tomorrow" in q:
        return "tomorrow"
    if "yesterday" in q:
        return "yesterday"
    if "weekday" in q or "day of the week" in q or "day of week" in q or "what day" in q:
        return "weekday"
    if "year" in q:
        return "year"
    return "now"

def score_tool(query: str, spec: ToolSpec) -> tuple[float, str]:
    q = query.lower()
    hits = [kw for kw in spec.keywords if kw.lower() in q]
    score = float(len(hits))
    reason = f"kw={hits}" if hits else "kw=[]"

    if spec.name == "calculator" and extract_math(query):
        score += 3.0
        reason += "+math"
    if spec.name == "weather" and extract_city(query):
        score += 3.0
        reason += "+city"
    if spec.name == "kb_lookup" and extract_kb_topic(query):
        score += 2.5
        reason += "+topic"
    if spec.name == "string_ops" and extract_string_args(query) is not None:
        score += 3.0
        reason += "+strop"
    if spec.name == "datetime_info":
        dt_kw = ["date", "today", "tomorrow", "yesterday", "weekday", "day of", "what day", "year", "time now"]
        if any(k in q for k in dt_kw):
            score += 2.5
            reason += "+dt"

    return score, reason

def build_call(query: str, spec: ToolSpec, score: float, reason: str) -> PlannedCall | None:
    if spec.name == "calculator":
        expr = extract_math(query)
        if not expr:
            return None
        return PlannedCall("calculator", {"expression": expr}, score, reason)
    if spec.name == "weather":
        city = extract_city(query)
        if not city:
            return None
        return PlannedCall("weather", {"city": city}, score, reason)
    if spec.name == "kb_lookup":
        topic = extract_kb_topic(query)
        if not topic:
            return None
        return PlannedCall("kb_lookup", {"topic": topic}, score, reason)
    if spec.name == "string_ops":
        args = extract_string_args(query)
        if not args:
            return None
        return PlannedCall("string_ops", args, score, reason)
    if spec.name == "datetime_info":
        return PlannedCall("datetime_info", {"query": extract_datetime_query(query)}, score, reason)
    return None

def plan_tools(
    query: str,
    registry: dict[str, ToolSpec],
    *,
    score_threshold: float = 2.0,
    max_calls: int = 3,
) -> Plan:
    """Score all tools; emit ordered multi-tool plan for compound queries."""
    scored: list[tuple[float, str, ToolSpec]] = []
    for spec in registry.values():
        s, reason = score_tool(query, spec)
        scored.append((s, reason, spec))
    scored.sort(key=lambda t: t[0], reverse=True)

    calls: list[PlannedCall] = []
    used: set[str] = set()
    for s, reason, spec in scored:
        if s < score_threshold:
            continue
        if spec.name in used:
            continue
        call = build_call(query, spec, s, reason)
        if call is None:
            continue
        calls.append(call)
        used.add(spec.name)
        if len(calls) >= max_calls:
            break

    if len(calls) < 2 and re.search(r"\b(?:and|then|;)\b", query, re.I):
        parts = re.split(r"\b(?:and then|and|, then|;|,)\b", query, flags=re.I)
        for part in parts:
            part = part.strip()
            if len(part) < 4:
                continue
            sub = plan_tools(part, registry, score_threshold=score_threshold, max_calls=1)
            for c in sub.calls:
                if c.tool not in used:
                    calls.append(c)
                    used.add(c.tool)
            if len(calls) >= max_calls:
                break

    return Plan(query=query, calls=calls, needs_tools=bool(calls))
