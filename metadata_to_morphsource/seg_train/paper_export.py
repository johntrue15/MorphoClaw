"""Generate paper-ready statistics, plots, and tables from the ledger.

The publication argument is:

    "An LLM-driven nnInteractive seed loop, combined with a confidence-
     gated student model trained iteratively on its own outputs, can
     reach human-curator parity (Dice >= X) on MorphoSource specimens
     without any new manual annotation."

To support that claim a reviewer needs:

- ``mode_summary.csv``        — Dice / volume agreement / time per mode.
- ``round_progression.csv``   — student val Dice per round, mode mix.
- ``per_specimen.csv``        — long-form table for ANOVA / per-taxon stats.
- ``vs_human.csv``            — cases with curated GT: human (voxelised
                                 mesh) vs nnInteractive vs student.
- ``dice_progression.png``    — student Dice over rounds.
- ``mode_mix.png``            — stacked bar of segmentation mode per
                                 round (router decisions).
- ``time_per_mode.png``       — runtime distribution.

All plots are produced with matplotlib only; everything else is plain
CSV so it's easy to drop into a paper.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .experiment_ledger import ExperimentLedger


log = logging.getLogger("seg_train.paper")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _safe_std(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0 if xs else None
    return statistics.pstdev(xs)


def _write_csv(path: Path, rows: list[dict],
               columns: Optional[list[str]] = None) -> None:
    if not rows:
        path.write_text("")
        return
    cols = columns or sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def aggregate_by_mode(episodes: list[dict]) -> list[dict]:
    by_mode: dict[str, list[dict]] = {}
    for ep in episodes:
        by_mode.setdefault(ep.get("mode", "unknown"), []).append(ep)

    rows: list[dict] = []
    for mode, items in sorted(by_mode.items()):
        rows.append({
            "mode": mode,
            "n": len(items),
            "n_with_gt": sum(1 for ep in items if ep.get("has_ground_truth")),
            "mean_dice": _safe_mean(ep.get("dice") for ep in items),
            "std_dice": _safe_std(ep.get("dice") for ep in items),
            "mean_iou": _safe_mean(ep.get("iou") for ep in items),
            "mean_recall": _safe_mean(ep.get("recall") for ep in items),
            "mean_precision": _safe_mean(ep.get("precision") for ep in items),
            "mean_hausdorff_mm": _safe_mean(
                ep.get("hausdorff_mm") for ep in items
            ),
            "mean_volume_diff_pct": _safe_mean(
                ep.get("volume_difference_pct") for ep in items
            ),
            "mean_duration_s": _safe_mean(ep.get("duration_s") for ep in items),
            "mean_n_prompts": _safe_mean(ep.get("n_prompts") for ep in items),
        })
    return rows


def aggregate_by_round(episodes: list[dict]) -> list[dict]:
    by_round: dict[int, list[dict]] = {}
    for ep in episodes:
        by_round.setdefault(int(ep.get("round_index", 0)), []).append(ep)
    rows: list[dict] = []
    for r, items in sorted(by_round.items()):
        modes = {}
        for ep in items:
            modes[ep.get("mode")] = modes.get(ep.get("mode"), 0) + 1
        student_dices = [
            ep.get("dice") for ep in items
            if ep.get("mode") == "student" and ep.get("dice") is not None
        ]
        nni_dices = [
            ep.get("dice") for ep in items
            if ep.get("mode") in ("nninteractive", "student_corrected")
            and ep.get("dice") is not None
        ]
        rows.append({
            "round_index": r,
            "n_episodes": len(items),
            "modes": json.dumps(modes, sort_keys=True),
            "n_student": modes.get("student", 0),
            "n_nninteractive": modes.get("nninteractive", 0),
            "n_corrected": modes.get("student_corrected", 0),
            "n_human_gt": modes.get("human_gt", 0),
            "mean_student_dice": _safe_mean(student_dices),
            "mean_nninteractive_dice": _safe_mean(nni_dices),
            "fraction_routed_to_nni": (
                sum(1 for ep in items if ep.get("routed_to_nninteractive"))
                / len(items) if items else None
            ),
        })
    return rows


def vs_human_table(episodes: list[dict]) -> list[dict]:
    """Per-specimen table comparing each model's segmentation to GT."""
    by_specimen: dict[str, dict] = {}
    for ep in episodes:
        media = ep.get("media_id")
        if not media:
            continue
        rec = by_specimen.setdefault(media, {"media_id": media})
        rec["taxonomy"] = ep.get("taxonomy", rec.get("taxonomy", ""))
        mode = ep.get("mode")
        if ep.get("dice") is not None:
            rec[f"{mode}_dice"] = ep["dice"]
        if ep.get("duration_s"):
            rec[f"{mode}_duration_s"] = ep["duration_s"]
        if ep.get("n_prompts"):
            rec[f"{mode}_n_prompts"] = ep["n_prompts"]
    return list(by_specimen.values())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots")
        return None


def plot_dice_progression(round_rows: list[dict], out_path: Path) -> Optional[str]:
    plt = _matplotlib()
    if plt is None or not round_rows:
        return None
    rounds = [r["round_index"] for r in round_rows]
    student = [r.get("mean_student_dice") for r in round_rows]
    nni = [r.get("mean_nninteractive_dice") for r in round_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(s is not None for s in student):
        ax.plot(rounds, [s if s is not None else math.nan for s in student],
                marker="o", label="student (autonomous)", color="#1f77b4")
    if any(n is not None for n in nni):
        ax.plot(rounds, [n if n is not None else math.nan for n in nni],
                marker="s", label="nnInteractive (LLM-driven)", color="#d62728")
    ax.axhline(0.85, linestyle="--", color="grey", alpha=0.5,
               label="graduation Dice = 0.85")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Dice vs human-curated GT")
    ax.set_title("Student vs nnInteractive Dice across rounds")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def plot_mode_mix(round_rows: list[dict], out_path: Path) -> Optional[str]:
    plt = _matplotlib()
    if plt is None or not round_rows:
        return None
    import numpy as np

    rounds = [r["round_index"] for r in round_rows]
    bars = {
        "human_gt": [r.get("n_human_gt", 0) for r in round_rows],
        "nninteractive": [r.get("n_nninteractive", 0) for r in round_rows],
        "student_corrected": [r.get("n_corrected", 0) for r in round_rows],
        "student": [r.get("n_student", 0) for r in round_rows],
    }
    bottom = np.zeros(len(rounds))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "human_gt": "#2ca02c",
        "nninteractive": "#d62728",
        "student_corrected": "#ff7f0e",
        "student": "#1f77b4",
    }
    for label, vals in bars.items():
        ax.bar(rounds, vals, bottom=bottom, label=label,
               color=colors.get(label))
        bottom += np.array(vals)
    ax.set_xlabel("Round")
    ax.set_ylabel("Episodes")
    ax.set_title("Segmentation source mix per round")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def plot_time_per_mode(mode_rows: list[dict], out_path: Path) -> Optional[str]:
    plt = _matplotlib()
    if plt is None or not mode_rows:
        return None
    rows = [r for r in mode_rows if r.get("mean_duration_s") is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r["mean_duration_s"])
    labels = [r["mode"] for r in rows]
    means = [r["mean_duration_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, means, color="#4c72b0")
    ax.set_xlabel("Mean duration (s)")
    ax.set_title("Wall-clock time per segmentation mode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_paper_artifacts(
    ledger: ExperimentLedger,
    output_dir: os.PathLike | str,
    *,
    paper_tag: Optional[str] = None,
    include_plots: bool = True,
) -> dict:
    """Write CSVs + plots + a Markdown summary derived from the ledger.

    Set ``include_plots=False`` to skip the matplotlib path entirely
    (useful for environments without matplotlib, or in unit tests that
    want to avoid the plotting backend).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    episodes = ledger.episodes(paper_tag=paper_tag) if paper_tag else list(ledger)
    if not episodes:
        log.warning("Ledger is empty; nothing to export")
        return {"n_episodes": 0, "output_dir": str(out)}

    mode_rows = aggregate_by_mode(episodes)
    round_rows = aggregate_by_round(episodes)
    specimen_rows = vs_human_table(episodes)

    _write_csv(out / "mode_summary.csv", mode_rows)
    _write_csv(out / "round_progression.csv", round_rows)
    _write_csv(out / "vs_human.csv", specimen_rows)
    _write_csv(out / "per_episode.csv", episodes)

    plots: dict = {}
    if include_plots:
        plots = {
            "dice_progression": plot_dice_progression(
                round_rows, out / "dice_progression.png"
            ),
            "mode_mix": plot_mode_mix(round_rows, out / "mode_mix.png"),
            "time_per_mode": plot_time_per_mode(
                mode_rows, out / "time_per_mode.png"
            ),
        }

    # Markdown summary
    md_path = out / "paper_summary.md"
    md_path.write_text(_render_markdown(
        episodes=episodes, mode_rows=mode_rows,
        round_rows=round_rows, paper_tag=paper_tag,
    ))

    return {
        "n_episodes": len(episodes),
        "output_dir": str(out),
        "csvs": {
            "mode_summary": str(out / "mode_summary.csv"),
            "round_progression": str(out / "round_progression.csv"),
            "vs_human": str(out / "vs_human.csv"),
            "per_episode": str(out / "per_episode.csv"),
        },
        "plots": plots,
        "summary_markdown": str(md_path),
    }


def _render_markdown(*, episodes: list[dict], mode_rows: list[dict],
                     round_rows: list[dict],
                     paper_tag: Optional[str]) -> str:
    lines = [
        "# Iterative Segmentation Experiment — paper data",
        "",
    ]
    if paper_tag:
        lines.append(f"**Paper tag:** `{paper_tag}`  ")
    lines += [
        f"**Total episodes:** {len(episodes)}",
        "",
        "## Mode summary",
        "",
        "| Mode | n | n w/GT | Mean Dice | Std Dice | Mean IoU | Mean H₉₅ (mm) | Mean ΔV (%) | Mean s/seg |",
        "|------|--:|------:|---------:|--------:|--------:|------------:|-----------:|----------:|",
    ]
    for r in mode_rows:
        def _fmt(v: Optional[float], digits: int = 4) -> str:
            return f"{v:.{digits}f}" if v is not None else "—"
        lines.append(
            f"| `{r['mode']}` | {r['n']} | {r['n_with_gt']} | "
            f"{_fmt(r['mean_dice'])} | {_fmt(r['std_dice'])} | "
            f"{_fmt(r['mean_iou'])} | {_fmt(r['mean_hausdorff_mm'], 2)} | "
            f"{_fmt(r['mean_volume_diff_pct'], 2)} | "
            f"{_fmt(r['mean_duration_s'], 1)} |"
        )

    lines += ["", "## Per-round progression", "",
              "| Round | Episodes | Student n | nnI n | Corrected n | Mean student Dice | Mean nnI Dice | % routed to nnI |",
              "|------:|--------:|---------:|-----:|-----------:|----------------:|------------:|---------------:|"]
    for r in round_rows:
        def _fmt(v: Optional[float]) -> str:
            return f"{v:.4f}" if v is not None else "—"
        frac = r.get("fraction_routed_to_nni")
        frac_str = f"{frac * 100:.1f}%" if frac is not None else "—"
        lines.append(
            f"| {r['round_index']} | {r['n_episodes']} | "
            f"{r['n_student']} | {r['n_nninteractive']} | "
            f"{r['n_corrected']} | "
            f"{_fmt(r['mean_student_dice'])} | "
            f"{_fmt(r['mean_nninteractive_dice'])} | {frac_str} |"
        )

    lines += ["", "## Plots", "",
              "![Dice progression](dice_progression.png)",
              "",
              "![Mode mix](mode_mix.png)",
              "",
              "![Time per mode](time_per_mode.png)",
              "",
              "## Files",
              "",
              "- `per_episode.csv` — every recorded segmentation event",
              "- `mode_summary.csv` — per-mode aggregate metrics",
              "- `round_progression.csv` — round-by-round student progress",
              "- `vs_human.csv` — per-specimen comparison vs curated GT",
              ""]
    return "\n".join(lines)
