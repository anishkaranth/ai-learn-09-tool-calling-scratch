#!/usr/bin/env python3
"""Hand-SVG plot + RESULTS.md writers for tool-calling smoke."""
from __future__ import annotations
from pathlib import Path
from typing import Any

def _bars(path: Path, title: str, items: list[tuple[str, float, str]], w: int = 520, h: int = 280) -> None:
    ml, mt, ch = 70, 40, h - 90
    n, gap = len(items), 12
    bw = (w - ml - 20 - gap * (n - 1)) / n
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="#fff"/>',
           f'<text x="{w/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold">{title}</text>',
           f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ch}" stroke="#333"/>', f'<line x1="{ml}" y1="{mt+ch}" x2="{w-20}" y2="{mt+ch}" stroke="#333"/>']
    for i, (lab, val, col) in enumerate(items):
        x = ml + i * (bw + gap); bh = max(0.0, min(1.0, val)) * ch; y = mt + ch - bh
        out += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" stroke="#222"/>',
                f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{val:.2f}</text>',
                f'<text x="{x+bw/2:.1f}" y="{mt+ch+16}" text-anchor="middle" font-family="sans-serif" font-size="10">{lab}</text>']
    out.append('</svg>\n'); path.write_text('\n'.join(out), encoding='utf-8')

def make_plots(results_dir: Path, *, tool_call_success_rate: float, tool_selection_accuracy: float, answer_accuracy: float, multi_step_accuracy: float, single_step_accuracy: float, tool_usage: dict[str, int]) -> list[str]:
    results_dir.mkdir(exist_ok=True)
    names = []
    p = results_dir / 'headline_metrics.svg'
    _bars(p, 'ai-learn-09 tool-calling smoke', [('tool-call', tool_call_success_rate, '#2a9d8f'), ('selection', tool_selection_accuracy, '#264653'), ('answer', answer_accuracy, '#e9c46a'), ('multi', multi_step_accuracy, '#f4a261'), ('single', single_step_accuracy, '#e76f51')])
    names.append(p.name)
    usage = [(k, v) for k, v in sorted(tool_usage.items()) if v > 0]
    mx = max((v for _, v in usage), default=1)
    h = 40 + 28 * max(len(usage), 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="{h}">', '<rect width="100%" height="100%" fill="#fff"/>', '<text x="240" y="22" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold">Tool usage</text>']
    for i, (name, count) in enumerate(usage):
        y = 40 + i * 28; bw = 200 * (count / mx)
        parts += [f'<text x="10" y="{y+12}" font-family="sans-serif" font-size="11">{name}</text>', f'<rect x="120" y="{y}" width="{bw:.1f}" height="16" fill="#457b9d" stroke="#222"/>', f'<text x="{120+bw+6:.1f}" y="{y+12}" font-family="sans-serif" font-size="11">{count}</text>']
    parts.append('</svg>\n'); p = results_dir / 'tool_usage.svg'; p.write_text('\n'.join(parts), encoding='utf-8'); names.append(p.name)
    p = results_dir / 'single_vs_multi.svg'
    _bars(p, 'Single vs multi accuracy', [('single', single_step_accuracy, '#8ecae6'), ('multi', multi_step_accuracy, '#fb8500')], w=360, h=260)
    names.append(p.name); return names

def write_results_md(results_dir: Path, metrics: dict[str, Any], plot_names: list[str]) -> None:
    lines = ['# Results — ai-learn-09-tool-calling-scratch', '', 'Smoke run of a **toy ReAct-style tool-calling agent**.', '', '## Headline metrics', '', '| Metric | Value |', '|--------|------:|',
             f"| Tasks | {metrics['n_tasks']} |", f"| Multi-step tasks | {metrics['n_multi_step']} |", f"| Tool-call success rate | {metrics['tool_call_success_rate']:.4f} |",
             f"| Tool-selection accuracy | {metrics['tool_selection_accuracy']:.4f} |", f"| Answer accuracy | {metrics['answer_accuracy']:.4f} |",
             f"| Multi-step answer accuracy | {metrics['multi_step_accuracy']:.4f} |", f"| Single-step answer accuracy | {metrics['single_step_accuracy']:.4f} |",
             f"| Mean tools / task | {metrics['mean_tools_per_task']:.2f} |", f"| Runtime (s) | {metrics['runtime_s']:.3f} |", f"| Tools registered | {metrics['n_tools']} |", '', '## Tool usage', '']
    for name, count in metrics['tool_usage'].items():
        lines.append(f'- `{name}`: {count}')
    lines += ['', '## Plots', ''] + [f'![{pn}]({pn})' for pn in plot_names] + ['', '## Per-task snapshot', '', '| id | multi | tools called | sel_ok | ans_ok |', '|----|:-----:|--------------|:------:|:------:|']
    for row in metrics['per_task']:
        lines.append(f"| {row['id']} | {str(row['multi_step']).lower()} | `{', '.join(row['tools_called'])}` | {'Y' if row['tool_selection_ok'] else 'N'} | {'Y' if row['answer_ok'] else 'N'} |")
    lines.append(''); (results_dir / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
