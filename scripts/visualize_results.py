#!/usr/bin/env python3
"""
EdgeAL Professional Visualization Suite

Generates publication-quality figures for the active learning benchmark:
1. mAP vs Budget curves (Track A & B)
2. Uncertainty signal comparison radar chart
3. Active learning pipeline diagram
4. Per-class AP breakdown
5. Seed variance analysis
6. COCO-style results table

Usage:
    python scripts/visualize_results.py --results-dir results/al_benchmark --output-dir docs/figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# Set professional style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.3)
sns.set_palette("colorblind")

# Color scheme
COLORS = {
    "random": "#7f7f7f",
    "entropy": "#1f77b4",
    "max_entropy": "#aec7e8",
    "least_confidence": "#ff7f0e",
    "margin": "#2ca02c",
    "disagreement": "#d62728",
    "disagreement_div": "#ff9896",
    "entropy_div": "#9467bd",
    "hybrid": "#8c564b",
    "trackA": "#1f77b4",
    "trackB": "#d62728",
}

ARM_LABELS = {
    "random": "Random",
    "entropy": "Mean Entropy",
    "max_entropy": "Max Entropy",
    "least_confidence": "Least Confidence",
    "margin": "Margin",
    "disagreement": "Disagreement",
    "disagreement_div": "Disagreement + Div",
    "entropy_div": "Entropy + Div",
    "hybrid": "Hybrid (Entropy+Div)",
}


def load_results(results_dir: Path) -> Dict:
    """Load aggregated results and summary."""
    summary_files = list(results_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary.json found in {results_dir}")
    
    with open(summary_files[0]) as f:
        summary = json.load(f)
    
    agg_files = list(results_dir.glob("*_aggregated.json"))
    if not agg_files:
        raise FileNotFoundError(f"No aggregated.json found in {results_dir}")
    
    with open(agg_files[0]) as f:
        aggregated = json.load(f)
    
    return {"summary": summary, "aggregated": aggregated}


def plot_map_vs_budget(
    aggregated: Dict,
    output_path: Path,
    title: str = "Active Learning Performance vs Annotation Budget",
    metric: str = "mAP50_95",
    tracks: Optional[List[str]] = None,
):
    """Plot mAP vs budget percentage for each arm."""
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    
    arms_in_order = [
        "random", "entropy", "max_entropy", "least_confidence", 
        "margin", "disagreement", "disagreement_div", "entropy_div", "hybrid"
    ]
    
    # Extract data
    for arm in arms_in_order:
        if arm not in ARM_LABELS:
            continue
        
        budgets = []
        means = []
        stds = []
        
        for key, val in aggregated.items():
            if val["arm"] == arm and metric in val["metrics"]:
                budgets.append(val["k"])
                means.append(val["metrics"][metric]["mean"])
                stds.append(val["metrics"][metric]["std"])
        
        if not budgets:
            continue
        
        # Sort by budget
        sorted_data = sorted(zip(budgets, means, stds))
        budgets, means, stds = zip(*sorted_data)
        
        # Convert to percentage (assuming total pool ~4000)
        budgets_pct = [b / 40 for b in budgets]
        
        color = COLORS.get(arm, "#000000")
        ax.errorbar(
            budgets_pct, means, yerr=stds,
            label=ARM_LABELS.get(arm, arm),
            color=color,
            marker="o",
            capsize=5,
            linewidth=2.5,
            markersize=9,
            alpha=0.9,
            elinewidth=1.5,
        )
    
    ax.set_xlabel("Annotation Budget (% of Training Pool)", fontsize=15, labelpad=10)
    ax.set_ylabel("mAP@50:95" if metric == "mAP50_95" else "mAP@50", fontsize=15, labelpad=10)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max(budgets_pct) * 1.15 if budgets_pct else 10.5)
    ax.set_ylim(bottom=0)
    
    # Add reference line for random baseline
    random_means = []
    for key, val in aggregated.items():
        if val["arm"] == "random" and metric in val["metrics"]:
            random_means.append(val["metrics"][metric]["mean"])
    if random_means:
        ax.axhline(y=np.mean(random_means), color="gray", linestyle="--", 
                   alpha=0.6, linewidth=1.5, label="Random Baseline (mean)")
    
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_track_comparison(
    aggregated: Dict,
    output_path: Path,
    metric: str = "mAP50_95",
):
    """Compare Track A vs Track B at each budget."""
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    
    budgets = [2, 4, 6, 8, 10]
    trackA_arms = ["entropy", "least_confidence", "margin", "disagreement", "hybrid"]
    trackB_arms = ["entropy", "disagreement", "disagreement_div", "entropy_div"]
    
    x = np.arange(len(budgets))
    width = 0.32
    
    for i, (track, arms, color) in enumerate([
        ("Track A (Human-Label)", trackA_arms, COLORS["trackA"]),
        ("Track B (Pseudo-Label)", trackB_arms, COLORS["trackB"]),
    ]):
        track_means = []
        track_stds = []
        
        for pct in budgets:
            k = int(4000 * pct / 100)
            arm_means = []
            arm_stds = []
            for arm in arms:
                for key, val in aggregated.items():
                    if val["arm"] == arm and val.get("k") == k and metric in val["metrics"]:
                        arm_means.append(val["metrics"][metric]["mean"])
                        arm_stds.append(val["metrics"][metric]["std"])
                        break
            if arm_means:
                track_means.append(np.mean(arm_means))
                track_stds.append(np.mean(arm_stds))
            else:
                track_means.append(0)
                track_stds.append(0)
        
        offset = width / 2 if i == 1 else -width / 2
        bars = ax.bar(
            x + offset, track_means, width,
            yerr=track_stds,
            label=track,
            color=color,
            alpha=0.8,
            capsize=5,
            edgecolor="black",
            linewidth=0.8,
            error_kw={"linewidth": 1.5, "capsize": 5},
        )
    
    ax.set_xlabel("Annotation Budget (%)", fontsize=15, labelpad=10)
    ax.set_ylabel("mAP@50:95" if metric == "mAP50_95" else "mAP@50", fontsize=15, labelpad=10)
    ax.set_title(f"Track A vs Track B: {metric.replace('_', '@')} by Budget", fontsize=17, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}%" for b in budgets], fontsize=13)
    ax.legend(fontsize=13, framealpha=0.95, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, len(budgets) - 0.5)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_uncertainty_radar(
    aggregated: Dict,
    output_path: Path,
    budget_pct: int = 10,
):
    """Radar chart comparing uncertainty methods at a specific budget."""
    k = int(4000 * budget_pct / 100)
    
    categories = [
        "Mean Entropy",
        "Max Entropy", 
        "Least Confidence",
        "Margin",
        "Disagreement",
        "Disagreement+Div",
        "Entropy+Div",
    ]
    
    arms = ["entropy", "max_entropy", "least_confidence", "margin", 
            "disagreement", "disagreement_div", "entropy_div"]
    
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True, subplot_kw=dict(polar=True))
    
    # Random baseline
    random_val = 0
    for key, val in aggregated.items():
        if val["arm"] == "random" and val.get("k") == k and "mAP50_95" in val["metrics"]:
            random_val = val["metrics"]["mAP50_95"]["mean"]
            break
    
    # Get values for each arm
    values = []
    for arm in arms:
        for key, val in aggregated.items():
            if val["arm"] == arm and val.get("k") == k and "mAP50_95" in val["metrics"]:
                values.append(val["metrics"]["mAP50_95"]["mean"])
                break
        else:
            values.append(0)
    
    # Normalize by random baseline (show gain over random)
    if random_val > 0:
        values_norm = [(v - random_val) / random_val * 100 for v in values]
    else:
        values_norm = values
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_norm += values_norm[:1]
    angles += angles[:1]
    
    ax.plot(angles, values_norm, "o-", linewidth=2.5, color=COLORS["trackA"], alpha=0.9, markersize=8)
    ax.fill(angles, values_norm, alpha=0.18, color=COLORS["trackA"])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Gain over Random (%)", fontsize=13, labelpad=20)
    ax.set_title(f"Uncertainty Method Comparison at {budget_pct}% Budget", 
                 fontsize=16, fontweight="bold", pad=25)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(min(values_norm), -5), top=max(max(values_norm), 5) * 1.2)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_seed_variance(
    aggregated: Dict,
    output_path: Path,
    metric: str = "mAP50_95",
):
    """Box plot showing variance across seeds for each arm."""
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    
    arms_in_order = [
        "random", "entropy", "least_confidence", "margin", 
        "disagreement", "disagreement_div", "entropy_div", "hybrid"
    ]
    
    data_to_plot = []
    labels = []
    
    for arm in arms_in_order:
        if arm not in ARM_LABELS:
            continue
        
        # Collect all seed values at 10% budget
        k = int(4000 * 10 / 100)
        arm_values = []
        for key, val in aggregated.items():
            if val["arm"] == arm and val.get("k") == k and metric in val["metrics"]:
                if "values" in val["metrics"][metric]:
                    arm_values.extend(val["metrics"][metric]["values"])
        
        if arm_values:
            data_to_plot.append(arm_values)
            labels.append(ARM_LABELS[arm])
    
    if not data_to_plot:
        print("No data for seed variance plot")
        return
    
    bp = ax.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color="red", linewidth=2.5),
        medianprops=dict(color="black", linewidth=2),
        widths=0.6,
        showcaps=True,
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    
    # Color boxes
    for patch, arm in zip(bp["boxes"], [a for a in arms_in_order if a in ARM_LABELS]):
        patch.set_facecolor(COLORS.get(arm, "#cccccc"))
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    
    ax.set_ylabel("mAP@50:95" if metric == "mAP50_95" else "mAP@50", fontsize=15, labelpad=10)
    ax.set_title(f"Seed Variance Analysis (10% Budget, 5 Seeds)", fontsize=17, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=12, ha="right", fontsize=12)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_active_learning_pipeline(output_path: Path):
    """Create a professional pipeline diagram for the active learning workflow."""
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    
    # Define boxes with more spacing
    boxes = [
        (0.8, 7, 3.2, 2.2, "Unlabeled\nPool (TRAIN_POOL)\n4,000 COCO images", "#e8f4fd"),
        (0.8, 2.5, 3.2, 2.2, "Held-out\nTEST Set\n1,000 images\nHuman Labels", "#ffe8e8"),
        (5.2, 7, 3.2, 2.2, "Student Model\n(YOLOv8n ONNX)\nUncertainty Signals", "#fff3e0"),
        (5.2, 2.5, 3.2, 2.2, "Teacher Model\n(RT-DETR)\nPseudo-Labels", "#f3e5f5"),
        (10.2, 4.5, 3.2, 2.2, "Acquisition\nFunction\n(Uncertainty/Diversity)", "#e8f5e9"),
        (13.8, 4.5, 3.2, 2.2, "Selected\nSubset (K images)", "#fff8e1"),
    ]
    
    # Draw boxes
    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="black",
            linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=12, fontweight="bold")
    
    # Draw arrows with proper spacing
    arrows = [
        (4.0, 8.1, 5.2, 8.1, "inference"),
        (4.0, 3.6, 5.2, 3.6, "inference"),
        (8.4, 8.1, 10.2, 5.6, "entropy, margin,\nleast-conf, etc."),
        (8.4, 3.6, 10.2, 5.6, "disagreement"),
        (13.4, 5.6, 13.8, 5.6, "top-K"),
    ]
    
    for sx, sy, ex, ey, label in arrows:
        ax.annotate("",
            xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="black", shrinkA=5, shrinkB=5),
        )
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx, my + 0.35, label, ha="center", va="bottom", fontsize=10, color="gray", fontweight="bold")
    
    # Track labels
    ax.text(2.4, 9.8, "Track A: Human-Label Active Learning", fontsize=15, fontweight="bold", color=COLORS["trackA"])
    ax.text(2.4, 1.0, "Track B: Pseudo-Label Data Engine", fontsize=15, fontweight="bold", color=COLORS["trackB"])
    
    # Dashed line separating tracks
    ax.axhline(y=5.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
    
    # Training & Evaluation boxes
    train_box = mpatches.FancyBboxPatch(
        (10.2, 1.0), 3.2, 1.8,
        boxstyle="round,pad=0.15",
        facecolor="#fce4ec", edgecolor="black", linewidth=1.5,
    )
    ax.add_patch(train_box)
    ax.text(11.8, 1.9, "Training\n(YOLOv8n fine-tune)", fontsize=13, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))
    
    eval_box = mpatches.FancyBboxPatch(
        (13.8, 1.0), 3.2, 1.8,
        boxstyle="round,pad=0.15",
        facecolor="#e8eaf6", edgecolor="black", linewidth=1.5,
    )
    ax.add_patch(eval_box)
    ax.text(15.4, 1.9, "Evaluation\n(mAP@50:95 on TEST)", fontsize=13, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))
    
    # Arrows
    ax.annotate("",
        xy=(11.8, 1.0), xytext=(11.8, 3.5),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="black", shrinkA=5, shrinkB=5),
    )
    ax.annotate("",
        xy=(15.4, 1.0), xytext=(15.4, 1.0),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="black"),
    )
    
    ax.set_title("EdgeAL Active Learning Pipeline", fontsize=20, fontweight="bold", pad=25)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.5)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.5)
    plt.close()
    print(f"Saved: {output_path}")


def create_results_table(
    aggregated: Dict,
    output_path: Path,
    metric: str = "mAP50_95",
):
    """Create a professional LaTeX-style results table."""
    budgets = [2, 4, 6, 8, 10]
    arms = ["random", "entropy", "least_confidence", "margin", 
            "disagreement", "disagreement_div", "entropy_div", "hybrid"]
    
    # Build table data
    rows = []
    for arm in arms:
        if arm not in ARM_LABELS:
            continue
        row = {"Method": ARM_LABELS[arm]}
        for pct in budgets:
            k = int(4000 * pct / 100)
            val = None
            for key, v in aggregated.items():
                if v["arm"] == arm and v.get("k") == k and metric in v["metrics"]:
                    m = v["metrics"][metric]
                    val = f"{m['mean']:.4f} ± {m['std']:.4f}"
                    break
            row[f"{pct}%"] = val or "—"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    
    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False, column_format="l" + "c" * len(budgets))
    with open(output_path.with_suffix(".tex"), "w") as f:
        f.write(latex)
    
    # Save as markdown
    md = df.to_markdown(index=False)
    with open(output_path.with_suffix(".md"), "w") as f:
        f.write(md)
    
    print(f"Saved table: {output_path.with_suffix('.csv')}, .tex, .md")


def main():
    parser = argparse.ArgumentParser(description="Generate professional figures for EdgeAL benchmark")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing benchmark results")
    parser.add_argument("--output-dir", type=str, default="docs/figures",
                        help="Output directory for figures")
    parser.add_argument("--metric", type=str, default="mAP50_95",
                        choices=["mAP50", "mAP50_95", "mAP75", "precision", "recall"],
                        help="Primary metric to plot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    data = load_results(results_dir)
    aggregated = data["aggregated"]

    print("Generating figures...")
    
    # 1. mAP vs Budget (main figure)
    plot_map_vs_budget(
        aggregated,
        output_dir / f"map_vs_budget_{args.metric}.png",
        metric=args.metric,
    )
    
    # 2. Track comparison
    plot_track_comparison(
        aggregated,
        output_dir / f"track_comparison_{args.metric}.png",
        metric=args.metric,
    )
    
    # 3. Radar chart at 10% budget
    plot_uncertainty_radar(
        aggregated,
        output_dir / f"uncertainty_radar_{args.metric}_10pct.png",
        budget_pct=10,
    )
    
    # 4. Seed variance
    plot_seed_variance(
        aggregated,
        output_dir / f"seed_variance_{args.metric}.png",
        metric=args.metric,
    )
    
    # 5. Pipeline diagram
    plot_active_learning_pipeline(
        output_dir / "al_pipeline.png",
    )
    
    # 6. Results table
    create_results_table(
        aggregated,
        output_dir / f"results_table_{args.metric}",
        metric=args.metric,
    )
    
    print(f"\nAll figures saved to {output_dir}")
    print("Generated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()