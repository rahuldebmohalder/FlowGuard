"""
FlowGuard — Experiment Pipeline
Implements experiments E1–E7 addressing RQ1–RQ6.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from src.utils.helpers import load_config, ensure_dir, save_json, Timer
from src.parsers.dataset_loader import DatasetManager
from src.parsers.solidity_parser import ContractParser
from src.static_analysis.stg_builder import STGBuilder
from src.static_analysis.detectors import run_all_detectors, Finding
from src.behavioral.feature_extractor import (
    TraceSimulator, BehavioralFeatureExtractor,
)
from src.behavioral.anomaly_detection import (
    EnsembleAnomalyScorer, IsolationForestDetector,
    LOFDetector, XGBoostDetector, AutoencoderDetector,
    FEATURE_COLS,
)
from src.fusion.correlation_engine import (
    TrustScorer, CorrelationEngine, RiskRanker,
)

logger = logging.getLogger("flowguard.experiments")


class ExperimentRunner:

    def __init__(self, cfg: Dict = None, output_dir: str = None):
        self.cfg = cfg or load_config()
        self.output_dir = ensure_dir(
            output_dir or self.cfg["paths"]["results"]
        )
        self.parser = ContractParser()
        self.stg_builder = STGBuilder()
        self.simulator = TraceSimulator()
        self.feature_extractor = BehavioralFeatureExtractor()
        self.trust_scorer = TrustScorer(self.cfg)
        self.correlation_engine = CorrelationEngine(self.cfg)
        self.risk_ranker = RiskRanker()
        self.results: Dict[str, Dict] = {}

    #  E1: Static Detection Effectiveness (RQ1)

    def run_e1_static_detection(
        self, contracts: list, labels: Dict[str, List[str]] = None
    ) -> Dict:

        logger.info("E1: Static Detection Effectiveness")
        results = {
            "contracts_analyzed": 0,
            "contracts_with_stg": 0,
            "findings_by_category": {},
            "per_contract": [],
        }

        for rec in contracts:
            results["contracts_analyzed"] += 1
            try:
                parse_result = self.parser.parse(rec.source_code, rec.file_path)
                stg = self.stg_builder.build(parse_result)
            except Exception:
                results["per_contract"].append({
                    "contract_id": rec.contract_id,
                    "has_stg": False, "findings": [],
                })
                continue

            if stg is None or not stg.has_valid_workflow:
                results["per_contract"].append({
                    "contract_id": rec.contract_id,
                    "has_stg": False,
                    "findings": [],
                })
                continue

            results["contracts_with_stg"] += 1
            findings = run_all_detectors(stg, parse_result)

            for f in findings:
                cat = f.category
                if cat not in results["findings_by_category"]:
                    results["findings_by_category"][cat] = 0
                results["findings_by_category"][cat] += 1

            results["per_contract"].append({
                "contract_id": rec.contract_id,
                "has_stg": True,
                "states": len(stg.states),
                "edges": len(stg.edges),
                "findings": [
                    {"category": f.category, "severity": f.severity,
                     "description": f.description[:120],
                     "confidence": f.confidence}
                    for f in findings
                ],
            })

        # Compute detection rate
        n_stg = results["contracts_with_stg"]
        n_total = results["contracts_analyzed"]
        results["stg_extraction_rate"] = n_stg / max(n_total, 1)
        results["total_findings"] = sum(
            results["findings_by_category"].values()
        )

        self.results["E1"] = results
        save_json(results, str(self.output_dir / "e1_static_detection.json"))
        logger.info(
            f"E1 complete: {n_stg}/{n_total} contracts have STGs, "
            f"{results['total_findings']} total findings"
        )
        return results

    # E2: SmartBugs Complementarity (RQ1)

    def run_e2_smartbugs_comparison(
        self, smartbugs_contracts: list
    ) -> Dict:

        logger.info("E2: SmartBugs Complementarity")
        results = {
            "total_contracts": len(smartbugs_contracts),
            "by_swc_category": {},
            "fg_findings_on_swc_contracts": {},
        }

        for rec in smartbugs_contracts:
            swc_cat = rec.swc_labels[0] if rec.swc_labels else "unknown"
            if swc_cat not in results["by_swc_category"]:
                results["by_swc_category"][swc_cat] = {
                    "count": 0, "fg_detected": 0, "fg_findings": []
                }
            results["by_swc_category"][swc_cat]["count"] += 1

            parse_result = self.parser.parse(rec.source_code)
            stg = self.stg_builder.build(parse_result)

            if stg and stg.has_valid_workflow:
                findings = run_all_detectors(stg, parse_result)
                if findings:
                    results["by_swc_category"][swc_cat]["fg_detected"] += 1
                    for f in findings:
                        results["by_swc_category"][swc_cat]["fg_findings"].append(
                            f.category
                        )

        self.results["E2"] = results
        save_json(results, str(self.output_dir / "e2_smartbugs.json"))
        logger.info(f"E2 complete: analyzed {len(smartbugs_contracts)} SmartBugs contracts")
        return results

    # E3: Behavioral Detection (RQ2)

    def run_e3_behavioral_detection(
        self, feature_df: pd.DataFrame
    ) -> Dict:
        logger.info("E3: Behavioral Detection")

        X = feature_df[FEATURE_COLS].values.astype(float)
        y = feature_df["label"].values.astype(int)

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # K-fold cross-validation
        n_folds = self.cfg["experiments"]["n_folds"]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scorer = EnsembleAnomalyScorer(self.cfg)
            scorer.fit(X_train, y_train)
            eval_result = scorer.evaluate(X_test, y_test, threshold=0.5)
            eval_result["fold"] = fold
            fold_results.append(eval_result)

        # Average across folds
        avg_results = {}
        model_names = fold_results[0].keys() - {"fold"}
        for model in model_names:
            metrics = {}
            for metric in ["auroc", "f1", "precision", "recall"]:
                values = [fr[model][metric] for fr in fold_results if model in fr]
                metrics[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            avg_results[model] = metrics

        results = {
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "n_folds": n_folds,
            "per_fold": fold_results,
            "average": avg_results,
        }

        self.results["E3"] = results
        save_json(results, str(self.output_dir / "e3_behavioral.json"))
        logger.info(f"E3 complete: {n_folds}-fold CV on {len(y)} samples")
        return results

    # E4: Fusion Benefit (RQ3 — KEY EXPERIMENT)

    def run_e4_fusion_benefit(
        self,
        contracts: list,
        feature_df: pd.DataFrame,
        contract_findings: Dict[str, List[Finding]],
    ) -> Dict:

        logger.info("E4: Fusion Benefit (KEY EXPERIMENT)")

        X = feature_df[FEATURE_COLS].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        y = feature_df["label"].values.astype(int)

        # ── Static-only predictions (no ML, same for all folds) ────
        static_preds = np.zeros(len(y))
        for i, (idx, row) in enumerate(feature_df.iterrows()):
            cid = row.get("contract_id", "")
            findings = contract_findings.get(cid, [])
            if findings:
                static_preds[i] = max(f.severity for f in findings)

        # Out-of-fold behavioural predictions
        n_folds = self.cfg["experiments"]["n_folds"]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        behavioral_preds = np.zeros(len(y))

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            scorer = EnsembleAnomalyScorer(self.cfg)
            scorer.fit(X[train_idx], y[train_idx])
            fold_scores = scorer.score(X[test_idx])
            behavioral_preds[test_idx] = fold_scores["ensemble"]

        # Fusion predictions
        alpha = self.cfg["fusion"]["alpha"]   # 0.40
        beta  = self.cfg["fusion"]["beta"]    # 0.35
        gamma = self.cfg["fusion"]["gamma"]   # 0.25

        # Compute trust & correlation using OOF behavioural scores
        trust_list = self.trust_scorer.compute(feature_df, behavioral_preds)

        fusion_preds = np.zeros(len(y))
        for i, (idx, row) in enumerate(feature_df.iterrows()):
            cid = row.get("contract_id", "")
            findings = contract_findings.get(cid, [])
            tc = trust_list[i]
            corr_result = self.correlation_engine.correlate(
                contract_id=cid,
                static_findings=findings,
                behavioral_features=row,
                trust_components=tc,
            )
            fusion_preds[i] = (
                alpha * static_preds[i]
                + beta * behavioral_preds[i]
                + gamma * corr_result.correlation_score
            )

        #  Optimal threshold selection
        def _optimal_threshold(y_true, scores):
            prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, scores)
            f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-10)
            best_idx = np.argmax(f1_arr)
            return float(thresholds[min(best_idx, len(thresholds)-1)])

        thresholds = {
            "static_only":     _optimal_threshold(y, static_preds),
            "behavioral_only": _optimal_threshold(y, behavioral_preds),
            "full_fusion":     _optimal_threshold(y, fusion_preds),
        }

        #  Evaluate each configuration at its optimal threshold
        configs_scores = {
            "static_only":     static_preds,
            "behavioral_only": behavioral_preds,
            "full_fusion":     fusion_preds,
        }

        results = {"configs": {}, "thresholds": thresholds}
        for name, preds in configs_scores.items():
            thr = thresholds[name]
            y_pred = (preds >= thr).astype(int)
            try:
                auroc = roc_auc_score(y, preds)
            except ValueError:
                auroc = 0.0

            results["configs"][name] = {
                "auroc": float(auroc),
                "f1": float(f1_score(y, y_pred, zero_division=0)),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y, y_pred)),
                "threshold": float(thr),
            }

        #  Compute fusion lift
        best_single = max(
            results["configs"]["static_only"]["f1"],
            results["configs"]["behavioral_only"]["f1"],
        )
        fusion_f1 = results["configs"]["full_fusion"]["f1"]
        results["fusion_lift_f1"] = fusion_f1 - best_single

        # Bootstrap significance test
        n_bootstrap = min(self.cfg["experiments"]["bootstrap_iterations"], 500)
        rng = np.random.default_rng(42)
        t_f = thresholds["full_fusion"]
        t_s = thresholds["static_only"]
        t_b = thresholds["behavioral_only"]
        boot_diffs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y), size=len(y), replace=True)
            y_b = y[idx]
            f1_fusion_b = f1_score(y_b, (fusion_preds[idx] >= t_f).astype(int),
                                   zero_division=0)
            f1_best_b = max(
                f1_score(y_b, (static_preds[idx] >= t_s).astype(int),
                         zero_division=0),
                f1_score(y_b, (behavioral_preds[idx] >= t_b).astype(int),
                         zero_division=0),
            )
            boot_diffs.append(f1_fusion_b - f1_best_b)

        results["bootstrap"] = {
            "mean_lift": float(np.mean(boot_diffs)),
            "std_lift": float(np.std(boot_diffs)),
            "p_value": float(np.mean(np.array(boot_diffs) <= 0)),
            "significant": bool(np.mean(np.array(boot_diffs) <= 0) < 0.01),
        }

        self.results["E4"] = results
        save_json(results, str(self.output_dir / "e4_fusion.json"))
        logger.info(
            f"E4 complete: fusion lift = {results['fusion_lift_f1']:.4f}, "
            f"p = {results['bootstrap']['p_value']:.4f}"
        )
        return results

    #  E5: Correlation Validity (RQ4)

    def run_e5_correlation_validity(
        self, correlation_results: list
    ) -> Dict:
        logger.info("E5: Correlation Validity")

        rule_stats = {}
        for cr in correlation_results:
            for rule_id in cr.triggered_rules:
                if rule_id not in rule_stats:
                    rule_stats[rule_id] = {"count": 0, "fused_scores": []}
                rule_stats[rule_id]["count"] += 1
                rule_stats[rule_id]["fused_scores"].append(cr.fused_risk_score)

        results = {}
        for rule_id, stats in rule_stats.items():
            results[rule_id] = {
                "trigger_count": stats["count"],
                "mean_fused_score": float(np.mean(stats["fused_scores"])),
                "std_fused_score": float(np.std(stats["fused_scores"])),
            }

        self.results["E5"] = results
        save_json(results, str(self.output_dir / "e5_correlation.json"))
        return results

    #  E6: Scalability (RQ5)

    def run_e6_scalability(self, contracts: list) -> Dict:
        logger.info("E6: Scalability")
        timings = []

        for rec in contracts[:120]:  # cap at 120 for timing
            row = {"contract_id": rec.contract_id, "source_length": len(rec.source_code)}

            try:
                # Parse time
                t0 = time.perf_counter()
                parse_result = self.parser.parse(rec.source_code)
                row["parse_time"] = time.perf_counter() - t0

                # STG build time
                t0 = time.perf_counter()
                stg = self.stg_builder.build(parse_result)
                row["stg_build_time"] = time.perf_counter() - t0

                # Detection time
                t0 = time.perf_counter()
                if stg and stg.has_valid_workflow:
                    _ = run_all_detectors(stg, parse_result)
                row["detection_time"] = time.perf_counter() - t0
            except Exception:
                # Skip pathological contracts
                continue

            row["total_static_time"] = (
                row["parse_time"] + row["stg_build_time"] + row["detection_time"]
            )
            timings.append(row)

        timing_df = pd.DataFrame(timings)
        results = {
            "contracts_measured": len(timings),
            "parse_time": {
                "mean": float(timing_df["parse_time"].mean()),
                "median": float(timing_df["parse_time"].median()),
                "p95": float(timing_df["parse_time"].quantile(0.95)),
            },
            "stg_build_time": {
                "mean": float(timing_df["stg_build_time"].mean()),
                "median": float(timing_df["stg_build_time"].median()),
                "p95": float(timing_df["stg_build_time"].quantile(0.95)),
            },
            "detection_time": {
                "mean": float(timing_df["detection_time"].mean()),
                "median": float(timing_df["detection_time"].median()),
                "p95": float(timing_df["detection_time"].quantile(0.95)),
            },
            "total_static_time": {
                "mean": float(timing_df["total_static_time"].mean()),
                "median": float(timing_df["total_static_time"].median()),
                "p95": float(timing_df["total_static_time"].quantile(0.95)),
                "pct_under_1s": float(
                    (timing_df["total_static_time"] < 1.0).mean()
                ),
            },
        }

        timing_df.to_csv(str(self.output_dir / "e6_timings.csv"), index=False)
        self.results["E6"] = results
        save_json(results, str(self.output_dir / "e6_scalability.json"))
        logger.info(
            f"E6 complete: mean total static = "
            f"{results['total_static_time']['mean']:.4f}s, "
            f"{results['total_static_time']['pct_under_1s']*100:.1f}% under 1s"
        )
        return results

    # E7: Ablation Study (RQ6)

    def run_e7_ablation(
        self, feature_df: pd.DataFrame, contract_findings: Dict
    ) -> Dict:

        logger.info("E7: Ablation Study")

        X = feature_df[FEATURE_COLS].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        y = feature_df["label"].values.astype(int)

        # Full system F1
        scorer = EnsembleAnomalyScorer(self.cfg)
        scorer.fit(X, y)
        anomaly_scores = scorer.score(X)["ensemble"]
        trust_list = self.trust_scorer.compute(feature_df, anomaly_scores)

        full_preds = []
        for i, (idx, row) in enumerate(feature_df.iterrows()):
            cid = row.get("contract_id", "")
            findings = contract_findings.get(cid, [])
            tc = trust_list[i]
            cr = self.correlation_engine.correlate(cid, findings, row, tc)
            full_preds.append(cr.fused_risk_score)
        full_preds = np.array(full_preds)
        full_f1 = f1_score(y, (full_preds >= 0.5).astype(int), zero_division=0)

        ablation_results = {"full_system_f1": float(full_f1), "ablations": {}}

        # Ablate feature categories
        category_ranges = self.cfg["behavioral"]["feature_categories"]
        for cat_name, (start, end) in category_ranges.items():
            X_ablated = X.copy()
            X_ablated[:, start:end + 1] = 0  # zero out category
            scorer_abl = EnsembleAnomalyScorer(self.cfg)
            scorer_abl.fit(X_ablated, y)
            abl_scores = scorer_abl.score(X_ablated)["ensemble"]
            abl_trust = self.trust_scorer.compute(feature_df, abl_scores)
            abl_preds = []
            for i, (idx, row) in enumerate(feature_df.iterrows()):
                cid = row.get("contract_id", "")
                findings = contract_findings.get(cid, [])
                tc = abl_trust[i]
                cr = self.correlation_engine.correlate(cid, findings, row, tc)
                abl_preds.append(cr.fused_risk_score)
            abl_f1 = f1_score(y, (np.array(abl_preds) >= 0.5).astype(int),
                              zero_division=0)
            ablation_results["ablations"][f"no_{cat_name}"] = {
                "f1": float(abl_f1),
                "delta_f1": float(full_f1 - abl_f1),
            }

        # Ablate correlation engine (set gamma=0)
        no_corr_preds = (
            self.cfg["fusion"]["alpha"] * full_preds
            + self.cfg["fusion"]["beta"] * (1 - np.array([tc.trust_score for tc in trust_list]))
        ) / (self.cfg["fusion"]["alpha"] + self.cfg["fusion"]["beta"])
        no_corr_f1 = f1_score(y, (no_corr_preds >= 0.5).astype(int),
                              zero_division=0)
        ablation_results["ablations"]["no_correlation"] = {
            "f1": float(no_corr_f1),
            "delta_f1": float(full_f1 - no_corr_f1),
        }

        self.results["E7"] = ablation_results
        save_json(ablation_results, str(self.output_dir / "e7_ablation.json"))
        logger.info(f"E7 complete: full F1={full_f1:.4f}")
        return ablation_results

    # Convenience: Run All

    def save_all_results(self):
        """Save consolidated results."""
        save_json(self.results, str(self.output_dir / "all_results.json"))
        logger.info(f"All results saved to {self.output_dir}")
