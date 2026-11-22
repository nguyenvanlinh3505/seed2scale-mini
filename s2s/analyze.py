"""Figures and tables mirroring the paper's evidence, from whatever stages have finished.
Hình và bảng tương ứng với bằng chứng trong paper, từ những giai đoạn đã xong.

  fig_collector.png  collector success per iteration and arm        (paper Fig. 7)
  fig_scaling.png    target success vs self-evolution iteration     (paper Fig. 5)
  table_targets.md   seen/unseen success for every target model      (paper Table I + Sec. IV-G)
  table_vlv.md       what the VLV rewards: score vs smoothness/length (paper Fig. 8, Table IV)
Usage / Cách dùng:  python -m s2s.analyze --work work
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2s.config import CFG

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARM_COLORS = {"s2s": "#2a78d6", "gate_only": "#eb6834", "no_filter": "#1baf7a"}  # fixed order / thứ tự cố định
ARM_LABELS = {"s2s": "Seed2Scale* (gate + VLV)", "gate_only": "SuperTiny⁻ (gate only)",
              "no_filter": "No filter"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []


def collector_table(runs: Path) -> dict[str, list[dict]]:
    out = {}
    for arm in ARM_COLORS:
        rows = []
        for i in range(CFG.iterations):
            s = runs / arm / f"iter_{i}" / "summary.json"
            if s.exists():
                rows.append(json.loads(s.read_text(encoding="utf-8")))
        if rows:
            out[arm] = rows
    return out


def target_results(results: Path) -> dict[str, dict]:
    return {p.stem.removeprefix("eval_"): json.loads(p.read_text(encoding="utf-8"))["results"]
            for p in sorted(results.glob("eval_*.json"))}


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def vlv_table(runs: Path) -> str:
    rows = [r for i in range(CFG.iterations) for r in read_jsonl(runs / "s2s" / f"iter_{i}" / "meta.jsonl")
            if r["gate_pass"] and r["vlv_score"] is not None]
    if not rows:
        return "No VLV-scored trajectories yet. / Chưa có quỹ đạo nào được VLV chấm.\n"
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]
    keys = ["len", "tv", "jerk", "hf_ratio"]
    lines = [f"VLV-scored successful rollouts: {len(rows)} (accepted {len(acc)}, rejected {len(rej)}); "
             f"score mean {np.mean([r['vlv_score'] for r in rows]):.2f}", "",
             "| metric | accepted mean | rejected mean | Spearman(score, metric) |", "|---|---|---|---|"]
    for k in keys:
        a = np.nanmean([r[k] for r in acc]) if acc else float("nan")
        b = np.nanmean([r[k] for r in rej]) if rej else float("nan")
        lines.append(f"| {k} | {a:.4f} | {b:.4f} | {spearman([r['vlv_score'] for r in rows], [r[k] for r in rows]):+.2f} |")
    hist = np.bincount(np.clip(np.round([r["vlv_score"] for r in rows]).astype(int), 0, 10), minlength=11)
    lines += ["", "Score histogram 0..10: " + " ".join(map(str, hist.tolist()))]
    return "\n".join(lines) + "\n"


def targets_md(res: dict[str, dict]) -> str:
    if not res:
        return "No target models evaluated yet. / Chưa đánh giá model mục tiêu nào.\n"
    lines = ["| target | source (seen) | unseen mean | " + " | ".join(CFG.unseen_worlds) + " |",
             "|---|---|---|" + "---|" * len(CFG.unseen_worlds)]
    for name, r in res.items():
        src = r.get("source", {})
        un = [r[w]["success_rate"] for w in CFG.unseen_worlds if w in r]
        fmt = lambda d: f"{d['success_rate']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]" if d else "-"
        lines.append(f"| {name} | {fmt(src)} | {np.mean(un):.2f} | "
                     + " | ".join(f"{r[w]['success_rate']:.2f}" if w in r else "-" for w in CFG.unseen_worlds) + " |")
    return "\n".join(lines) + "\n"


def plots(coll: dict, res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    if coll:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for arm, rows in coll.items():
            ax.plot([r["iteration"] for r in rows], [r["collector_success_rate"] for r in rows], marker="o",
                    lw=2, ms=6, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        ax.set_xlabel("self-evolution iteration")
        ax.set_ylabel("collector success rate")
        ax.grid(axis="y", color="#e4e3df")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_collector.png", dpi=180)
        plt.close(fig)

    pts = [(0, "seed_only"), (2, "s2s_i2"), (4, "s2s_i4"), (8, "s2s_i8")]
    pts = [(i, n) for i, n in pts if n in res and "source" in res[n]]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        x = [i for i, _ in pts]
        y = [res[n]["source"]["success_rate"] for _, n in pts]
        lo = [res[n]["source"]["ci95"][0] for _, n in pts]
        hi = [res[n]["source"]["ci95"][1] for _, n in pts]
        ax.fill_between(x, lo, hi, color=ARM_COLORS["s2s"], alpha=0.15, lw=0)
        ax.plot(x, y, marker="o", lw=2, ms=6, color=ARM_COLORS["s2s"], label="SmolVLA on D(i), Seed2Scale*")
        for arm in ("gate_only", "no_filter"):
            n = f"{arm}_i8"
            if n in res:
                d = res[n]["source"]
                ax.errorbar([8.15 if arm == "gate_only" else 8.3], [d["success_rate"]],
                            yerr=[[d["success_rate"] - d["ci95"][0]], [d["ci95"][1] - d["success_rate"]]],
                            fmt="s", ms=7, color=ARM_COLORS[arm], label=f"SmolVLA, {ARM_LABELS[arm]}")
        ax.set_xlabel("self-evolution iteration (0 = seeds only)")
        ax.set_ylabel("SmolVLA success (source world)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e4e3df")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_scaling.png", dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="work")
    p.add_argument("--runs-name", default="runs")
    p.add_argument("--results-name", default="results")
    a = p.parse_args()
    work = Path(a.work)
    runs = work / a.runs_name
    out = work / a.results_name
    out.mkdir(parents=True, exist_ok=True)
    coll = collector_table(runs)
    res = target_results(out)
    plots(coll, res, out)
    md = ["# Seed2Scale-mini results / Kết quả", "", "## Collector per iteration / Collector theo vòng", ""]
    for arm, rows in coll.items():
        md.append(f"- {ARM_LABELS[arm]}: success " + ", ".join(f"{r['collector_success_rate']:.2f}" for r in rows)
                  + f"; |D| after last iteration = {rows[-1]['dataset_trajs_after']}")
    md += ["", "## Target models (SmolVLA) / Model mục tiêu", "", targets_md(res),
           "## What the VLV rewards / VLV thưởng cho điều gì", "", vlv_table(runs)]
    pw = runs / "s2s" / "parallel_worlds" / "summary.json"
    if pw.exists():
        md += ["## Parallel Worlds", "", "```", pw.read_text(encoding="utf-8"), "```"]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
