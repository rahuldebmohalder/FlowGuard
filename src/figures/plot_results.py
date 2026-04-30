"""
FlowGuard — Figure Generation for CCS Paper
All figures at 300 DPI, PDF format, publication-ready.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

from src.utils.helpers import load_config, ensure_dir

logger = logging.getLogger("flowguard.figures")


class FigureGenerator:
    """Generates all paper figures from experiment results."""

    def __init__(self, cfg: Dict = None, output_dir: str = None):
        self.cfg = cfg or load_config()
        self.output_dir = ensure_dir(
            output_dir or self.cfg["paths"]["figures"]
        )
        self.dpi = self.cfg["figures"]["dpi"]
        self.fmt = self.cfg["figures"]["format"]
        self.font_size = self.cfg["figures"]["font_size"]

        plt.rcParams.update({
            "font.size": self.font_size,
            "font.family": "serif",
            "axes.labelsize": self.font_size,
            "axes.titlesize": self.font_size + 2,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 2,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
        })
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#2CA58D",
            "danger": "#C73E1D",
            "gray": "#6C757D",
            "light": "#ADB5BD",
        }

    def _savefig(self, fig, name: str):
        # Save PDF (for paper)
        pdf_path = self.output_dir / f"{name}.pdf"
        fig.savefig(str(pdf_path), format="pdf", dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved figure: {pdf_path}")
        # Save PNG (for preview / presentation)
        png_path = self.output_dir / f"{name}.png"
        fig.savefig(str(png_path), format="png", dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved figure: {png_path}")
        plt.close(fig)


    #  Figure 1: Detection Accuracy by FG Category

    def fig_detection_by_category(self, e1_results: Dict):
        cats = e1_results.get("findings_by_category", {})
        if not cats:
            logger.warning("No category data for fig_detection_by_category")
            return

        labels = sorted(cats.keys())
        values = [cats[k] for k in labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=self.colors["primary"],
                      edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=10)

        ax.set_xlabel("Vulnerability Category")
        ax.set_ylabel("Number of Findings")
        ax.set_title("FlowGuard Static Detection: Findings by Category")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._savefig(fig, "fig1_detection_by_category")

    #  Figure 2: Behavioral Model Comparison

    def fig_behavioral_comparison(self, e3_results: Dict):
        avg = e3_results.get("average", {})
        if not avg:
            return

        models = list(avg.keys())
        metrics = ["auroc", "f1", "precision", "recall"]
        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = [self.colors["primary"], self.colors["secondary"],
                  self.colors["accent"], self.colors["success"]]

        for i, metric in enumerate(metrics):
            vals = [avg[m][metric]["mean"] for m in models]
            errs = [avg[m][metric]["std"] for m in models]
            ax.bar(x + i * width, vals, width, yerr=errs, label=metric.upper(),
                   color=colors[i], capsize=3, edgecolor="white", linewidth=0.3)

        ax.set_xlabel("Anomaly Detection Model")
        ax.set_ylabel("Score")
        ax.set_title("Behavioral Detection: Model Comparison (5-Fold CV)")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._savefig(fig, "fig2_behavioral_comparison")

    # Figure 3: Fusion Benefit (KEY FIGURE)

    def fig_fusion_benefit(self, e4_results: Dict):
        configs = e4_results.get("configs", {})
        if not configs:
            return

        config_names = ["static_only", "behavioral_only", "full_fusion"]
        display_names = ["Static Only", "Behavioral Only", "Full FlowGuard"]
        metrics = ["auroc", "f1", "precision", "recall"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: grouped bars
        ax = axes[0]
        x = np.arange(len(config_names))
        width = 0.2
        colors = [self.colors["primary"], self.colors["secondary"],
                  self.colors["accent"], self.colors["success"]]
        for i, metric in enumerate(metrics):
            vals = [configs[c].get(metric, 0) for c in config_names]
            ax.bar(x + i * width, vals, width, label=metric.upper(),
                   color=colors[i], edgecolor="white", linewidth=0.3)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(display_names)
        ax.set_ylabel("Score")
        ax.set_title("(a) Detection Performance by Configuration")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Right: fusion lift with bootstrap CI
        ax2 = axes[1]
        bootstrap = e4_results.get("bootstrap", {})
        labels = ["Static", "Behavioral", "Fusion"]
        f1_vals = [
            configs.get("static_only", {}).get("f1", 0),
            configs.get("behavioral_only", {}).get("f1", 0),
            configs.get("full_fusion", {}).get("f1", 0),
        ]
        bar_colors = [self.colors["gray"], self.colors["gray"],
                      self.colors["success"]]
        bars = ax2.bar(labels, f1_vals, color=bar_colors, edgecolor="white")

        if bootstrap:
            sig_marker = "★" if bootstrap.get("significant") else ""
            ax2.set_title(
                f"(b) F1 Comparison — Lift: "
                f"{bootstrap.get('mean_lift', 0):.3f} "
                f"(p={bootstrap.get('p_value', 1):.3f}) {sig_marker}"
            )

        ax2.set_ylabel("F1 Score")
        ax2.set_ylim(0, 1.1)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        self._savefig(fig, "fig3_fusion_benefit")

    # Figure 4: Scalability

    def fig_scalability(self, timing_csv: str = None):
        if timing_csv is None:
            timing_csv = str(Path(self.cfg["paths"]["results"]) / "e6_timings.csv")
        if not Path(timing_csv).exists():
            logger.warning("No timing CSV found for scalability figure")
            return

        df = pd.read_csv(timing_csv)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram of total time
        ax = axes[0]
        ax.hist(df["total_static_time"] * 1000, bins=30,
                color=self.colors["primary"], edgecolor="white", alpha=0.8)
        ax.axvline(df["total_static_time"].median() * 1000, color=self.colors["danger"],
                   linestyle="--", label=f"Median: {df['total_static_time'].median()*1000:.1f}ms")
        ax.set_xlabel("Analysis Time (ms)")
        ax.set_ylabel("Number of Contracts")
        ax.set_title("(a) Static Analysis Time Distribution")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Right: time vs source length
        ax2 = axes[1]
        ax2.scatter(df["source_length"] / 1000, df["total_static_time"] * 1000,
                    alpha=0.4, s=15, color=self.colors["secondary"])
        ax2.set_xlabel("Source Code Length (KB)")
        ax2.set_ylabel("Analysis Time (ms)")
        ax2.set_title("(b) Analysis Time vs. Contract Complexity")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        self._savefig(fig, "fig4_scalability")

    # Figure 5: Ablation Study

    def fig_ablation(self, e7_results: Dict):
        full_f1 = e7_results.get("full_system_f1", 0)
        ablations = e7_results.get("ablations", {})
        if not ablations:
            return

        labels = list(ablations.keys())
        deltas = [ablations[k]["delta_f1"] for k in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [self.colors["danger"] if d > 0 else self.colors["success"]
                  for d in deltas]
        bars = ax.barh(labels, deltas, color=colors, edgecolor="white")

        ax.set_xlabel("ΔF1 (drop from full system)")
        ax.set_title(f"Ablation Study — Full System F1: {full_f1:.3f}")
        ax.axvline(0, color="black", linewidth=0.5)

        for bar, delta in zip(bars, deltas):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{delta:+.3f}", va="center", fontsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        self._savefig(fig, "fig5_ablation")

    # Figure 6: Risk Score Distribution

    def fig_risk_distribution(self, risk_df: pd.DataFrame):
        if risk_df is None or risk_df.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        if "label" in risk_df.columns and "fused_risk_score" in risk_df.columns:
            benign = risk_df[risk_df["label"] == 0]["fused_risk_score"]
            anomalous = risk_df[risk_df["label"] == 1]["fused_risk_score"]

            ax.hist(benign, bins=30, alpha=0.6, label="Benign",
                    color=self.colors["success"], edgecolor="white")
            ax.hist(anomalous, bins=30, alpha=0.6, label="Anomalous",
                    color=self.colors["danger"], edgecolor="white")
            ax.axvline(0.5, color="black", linestyle="--", label="Threshold")
            ax.set_xlabel("Fused Risk Score")
            ax.set_ylabel("Count")
            ax.set_title("Risk Score Distribution: Benign vs. Anomalous")
            ax.legend()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._savefig(fig, "fig6_risk_distribution")

    # Generate All

    def generate_all(self, results: Dict, risk_df: pd.DataFrame = None):
        if "E1" in results:
            self.fig_detection_by_category(results["E1"])
        if "E3" in results:
            self.fig_behavioral_comparison(results["E3"])
        if "E4" in results:
            self.fig_fusion_benefit(results["E4"])
        self.fig_scalability()
        if "E7" in results:
            self.fig_ablation(results["E7"])
        if risk_df is not None:
            self.fig_risk_distribution(risk_df)
        logger.info(f"All figures generated in {self.output_dir}")
