#!/usr/bin/env python3
"""Tool-calling agent smoke: plan -> execute -> eval -> results/."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from agent import run_agent
from smoke_plots import make_plots, write_results_md
from tasks import TASKS, answer_matches, tool_selection_ok
from tools import build_registry

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42  # deterministic world; planner is rule-based

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    registry = build_registry()
    per_task = []
    n_sel_ok = 0
    n_ans_ok = 0
    n_multi = 0
    n_multi_ok = 0
    n_single = 0
    n_single_ok = 0
    total_tool_calls = 0
    total_tool_ok = 0
    tool_usage: Counter[str] = Counter()
    total_steps = 0

    for task in TASKS:
        ar = run_agent(task["query"], registry)
        sel_ok = tool_selection_ok(ar.tools_called, task["gold_tools"])
        ans_ok = answer_matches(ar.answer, task["gold_contains"])
        multi = bool(task.get("multi_step"))

        total_tool_calls += ar.tool_successes + ar.tool_failures
        total_tool_ok += ar.tool_successes
        total_steps += ar.n_steps
        for tname in ar.tools_called:
            tool_usage[tname] += 1

        if sel_ok:
            n_sel_ok += 1
        if ans_ok:
            n_ans_ok += 1
        if multi:
            n_multi += 1
            if ans_ok:
                n_multi_ok += 1
        else:
            n_single += 1
            if ans_ok:
                n_single_ok += 1

        per_task.append(
            {
                "id": task["id"],
                "query": task["query"],
                "gold_tools": task["gold_tools"],
                "tools_called": ar.tools_called,
                "multi_step": multi,
                "tool_selection_ok": sel_ok,
                "answer_ok": ans_ok,
                "n_steps": ar.n_steps,
                "tool_successes": ar.tool_successes,
                "tool_failures": ar.tool_failures,
                "answer": ar.answer,
                "plan_scores": [
                    {"tool": c.tool, "score": c.score, "args": c.args, "reason": c.reason}
                    for c in ar.plan.calls
                ],
            }
        )

    n_tasks = len(TASKS)
    tool_call_success_rate = (total_tool_ok / total_tool_calls) if total_tool_calls else 0.0
    tool_selection_accuracy = n_sel_ok / n_tasks
    answer_accuracy = n_ans_ok / n_tasks
    multi_step_accuracy = (n_multi_ok / n_multi) if n_multi else 0.0
    single_step_accuracy = (n_single_ok / n_single) if n_single else 0.0
    mean_tools = (sum(len(r["tools_called"]) for r in per_task) / n_tasks) if n_tasks else 0.0
    mean_steps = total_steps / n_tasks if n_tasks else 0.0

    usage_dict = {name: int(tool_usage[name]) for name in sorted(tool_usage)}
    for name in registry:
        usage_dict.setdefault(name, 0)

    plot_names = make_plots(
        RESULTS,
        tool_call_success_rate=tool_call_success_rate,
        tool_selection_accuracy=tool_selection_accuracy,
        answer_accuracy=answer_accuracy,
        multi_step_accuracy=multi_step_accuracy,
        single_step_accuracy=single_step_accuracy,
        tool_usage={k: v for k, v in usage_dict.items() if v > 0 or k in registry},
    )
    runtime_s = time.perf_counter() - t0

    metrics = {
        "project": "ai-learn-09-tool-calling-scratch",
        "seed": SEED,
        "n_tasks": n_tasks,
        "n_multi_step": n_multi,
        "n_single_step": n_single,
        "n_tools": len(registry),
        "tool_names": sorted(registry.keys()),
        "tool_call_success_rate": round(tool_call_success_rate, 4),
        "tool_selection_accuracy": round(tool_selection_accuracy, 4),
        "answer_accuracy": round(answer_accuracy, 4),
        "multi_step_accuracy": round(multi_step_accuracy, 4),
        "single_step_accuracy": round(single_step_accuracy, 4),
        "n_tool_selection_ok": n_sel_ok,
        "n_answer_ok": n_ans_ok,
        "n_multi_ok": n_multi_ok,
        "n_single_ok": n_single_ok,
        "total_tool_calls": total_tool_calls,
        "total_tool_ok": total_tool_ok,
        "mean_tools_per_task": round(mean_tools, 4),
        "mean_steps_per_task": round(mean_steps, 4),
        "tool_usage": usage_dict,
        "runtime_s": round(runtime_s, 3),
        "per_task": per_task,
    }

    shot = {
        "project": metrics["project"],
        "n_tasks": n_tasks,
        "n_tools": metrics["n_tools"],
        "tool_call_success_rate": metrics["tool_call_success_rate"],
        "tool_selection_accuracy": metrics["tool_selection_accuracy"],
        "answer_accuracy": metrics["answer_accuracy"],
        "multi_step_accuracy": metrics["multi_step_accuracy"],
        "single_step_accuracy": metrics["single_step_accuracy"],
        "mean_tools_per_task": metrics["mean_tools_per_task"],
        "runtime_s": metrics["runtime_s"],
        "pass": bool(
            metrics["tool_call_success_rate"] >= 0.95
            and metrics["answer_accuracy"] >= 0.9
            and metrics["multi_step_accuracy"] >= 0.8
        ),
    }

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2) + "\n", encoding="utf-8")
    write_results_md(RESULTS, metrics, plot_names)

    print("=== ai-learn-09-tool-calling-scratch smoke ===")
    print(json.dumps(shot, indent=2))
    print(f"Wrote {RESULTS}/ ({', '.join(plot_names)})")

if __name__ == "__main__":
    main()
