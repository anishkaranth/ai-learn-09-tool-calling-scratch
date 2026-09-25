# AI Learn 09 — Toy Tool-Calling Agent from Scratch

Build a **ReAct-style tool-calling agent** without an LLM API: a tool registry with JSON-schema specs, a heuristic planner that picks tools from a query, execute → observe, and a template final answer.

Phase B (AI components) — follows toy RAG (`ai-learn-08`). Self-contained; does not import that repo.

## Learning goals

- **Tool registry**: name, description, JSON-Schema-like parameters, pure-Python handlers
- **Function-calling sketch**: planner scores tools (keywords + arg extractors) and emits structured calls
- **ReAct loop**: Thought → Action → Observation → Final Answer (multi-step for compound queries)
- **Safe tools**: AST-whitelisted calculator, deterministic datetime anchor, fake weather, toy KB, string ops
- **Eval**: tool-call success, tool-selection accuracy, answer accuracy (single- vs multi-step)

## Brief architecture

```
Query
  │
  ▼
Planner  ──score tools──►  PlannedCall[]  (tool, args, score)
  │
  ▼
Agent loop (ReAct sketch)
  Thought → Action(tool, args) → Observation → …
  │
  ▼
Template synthesizer → Final Answer
```

Tools registered in smoke:

| Tool | Purpose |
|------|---------|
| `calculator` | Safe AST math (`+ - * / ** sqrt …`) |
| `datetime_info` | Today / tomorrow / weekday / year (fixed 2026-09-25 anchor) |
| `weather` | Fake city weather lookup |
| `kb_lookup` | Tiny AI/ML fact KB |
| `string_ops` | upper / lower / reverse / word_count / … |

## Layout

```
world.py               # tool handlers, registry, JSON-schema specs
tools.py               # thin re-export shim (from world)
planner.py             # keyword/regex planner + arg extractors
agent.py               # ReAct-style loop + answer synthesis
tasks.py               # labeled eval suite (single + multi-step)
smoke_plots.py         # SVG plots + RESULTS.md
run_smoke.py           # end-to-end smoke -> results/
notebooks/tool_calling_scratch.ipynb
results/               # committed metrics + SVG plots
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

Runs on CPU in under a second. See `results/RESULTS.md` for the latest smoke metrics.

## What you'll learn next (Phase B)

Eval harnesses, LoRA / light fine-tune, multimodal toy — then a full end-to-end AI app milestone (Phase C).
