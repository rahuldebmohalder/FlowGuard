
"""
FlowGuard — Main Pipeline

"""
import sys, json, argparse, logging
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, setup_logger, save_json, ensure_dir, Timer
from src.parsers.dataset_loader import (
    DatasetManager, ContractRecord, SmartBugsLoader, ScrawlDLoader, SCVulnLoader
)
from src.parsers.workflow_annotator import (
    WorkflowAnnotator, ContractAugmenter, annotation_statistics
)
from src.parsers.solidity_parser import ContractParser
from src.static_analysis.stg_builder import STGBuilder
from src.static_analysis.detectors import run_all_detectors
from src.behavioral.feature_extractor import TraceSimulator, BehavioralFeatureExtractor
from src.behavioral.anomaly_detection import EnsembleAnomalyScorer, FEATURE_COLS
from src.fusion.correlation_engine import TrustScorer, CorrelationEngine, RiskRanker
from src.experiments.runner import ExperimentRunner
from src.figures.plot_results import FigureGenerator

import os
os.environ["FLOWGUARD_FORCE_REGEX"] = "1"


def run_full_pipeline(cfg, logger):
    results_dir = ensure_dir(cfg["paths"]["results"])


    logger.info("PHASE 1: Loading Real Datasets")

    sb_loader = SmartBugsLoader()
    smartbugs = sb_loader.load()

    # SC Vuln: only enum-bearing contracts (the workflow STG candidates)
    sc_vuln_loader = SCVulnLoader()
    sc_vuln = sc_vuln_loader.load(only_enum=True, max_source_chars=20000)

    sd_loader = ScrawlDLoader()
    scrawld = sd_loader.load_labels()

    real_contracts = smartbugs + sc_vuln

    # Load blacklist of contracts that hang the parser on Windows
    try:
        import json
        blacklist_path = Path("src/parsers/blacklist.json")
        # blacklist_path = Path("blacklist.json")
        if blacklist_path.exists():
            blacklist = set(json.loads(blacklist_path.read_text()))
            before = len(real_contracts)
            real_contracts = [r for r in real_contracts if r.contract_id not in blacklist]
            smartbugs = [r for r in smartbugs if r.contract_id not in blacklist]
            sc_vuln = [r for r in sc_vuln if r.contract_id not in blacklist]
            logger.info(f"Blacklist: removed {before - len(real_contracts)} hanging contracts")
    except Exception as e:
        logger.warning(f"Could not load blacklist: {e}")

    logger.info(f"Real Contracts with Source Code")
    logger.info(f"  SmartBugs: {len(smartbugs)} contracts")
    logger.info(f"  SC Vuln:    {len(sc_vuln)} contracts")
    logger.info(f"  TOTAL:     {len(real_contracts)} contracts")
    logger.info(f"Vulnerability Labels (no source)")
    logger.info(f"  ScrawlD:   {scrawld['statistics']['labeled_contracts']} contracts")
    for vt, n in sorted(scrawld['statistics']['vulnerability_distribution'].items(),
                         key=lambda x: -x[1]):
        logger.info(f"    {vt}: {n}")

    # Filter for enum-bearing real contracts (workflow STG candidates)
    enum_contracts = [r for r in real_contracts if "enum " in r.source_code]
    sb_enum = sum(1 for r in enum_contracts if r.dataset == "smartbugs")
    sc_vuln_enum = sum(1 for r in enum_contracts if r.dataset == "sc_vuln")
    logger.info(f"Enum-bearing (STG candidates)")
    logger.info(f"  SmartBugs enum: {sb_enum}")
    logger.info(f"  sc_vuln enum:    {sc_vuln_enum}")
    logger.info(f"  TOTAL enum:     {len(enum_contracts)}")

    save_json({
        "smartbugs": {
            "total": len(smartbugs),
            "with_enum": sb_enum,
            "by_swc": dict(Counter(r.metadata.get("category", "?") for r in smartbugs)),
        },
        "sc_vuln": {
            "total": len(sc_vuln),
            "with_enum": sc_vuln_enum,
            "by_label": dict(Counter(r.metadata.get("normalized_label", "?") for r in sc_vuln)),
        },
        "scrawld": scrawld['statistics'],
        "total_real_with_source": len(real_contracts),
        "total_enum_bearing": len(enum_contracts),
    }, str(results_dir / "dataset_summary.json"))


    logger.info("PHASE 2: Workflow Annotation Layer")

    annotator = WorkflowAnnotator()

    annotations = annotator.annotate_batch(real_contracts, only_enum_bearing=True)

    ann_stats = annotation_statistics(annotations)
    logger.info(f"Annotated: {ann_stats['total_annotated']} enum-bearing contracts")
    logger.info(f"  With workflow (STG built): {ann_stats['with_workflow']}")
    logger.info(f"  With FG findings:           {ann_stats['with_findings']}")
    logger.info(f"  FG distribution: {ann_stats['fg_distribution']}")
    logger.info(f"  By dataset (with findings): {ann_stats['by_dataset_with_findings']}")
    logger.info(f"  Review priority: {ann_stats['by_review_priority']}")

    # Save annotations to JSON for reproducibility / manual validation
    annotations_dump = []
    for ann in annotations:
        if ann.has_workflow:
            annotations_dump.append({
                "contract_id": ann.contract_id,
                "dataset": ann.dataset,
                "n_states": ann.n_states,
                "n_transitions": ann.n_transitions,
                "fg_labels": ann.fg_labels,
                "confidence": round(ann.confidence, 3),
                "review_priority": ann.review_priority,
                "findings": ann.findings,
            })
    save_json(annotations_dump, str(results_dir / "workflow_annotations.json"))

    # Manual validation worksheet — top 20 high-confidence findings
    high_conf = sorted(
        [a for a in annotations if a.fg_labels and a.review_priority == "high"],
        key=lambda a: -a.confidence
    )[:20]
    val_worksheet = [
        {
            "rank": i + 1,
            "contract_id": a.contract_id,
            "dataset": a.dataset,
            "fg_labels": a.fg_labels,
            "confidence": round(a.confidence, 3),
            "manual_label": "TODO: TP / FP / Uncertain",
        }
        for i, a in enumerate(high_conf)
    ]
    save_json(val_worksheet, str(results_dir / "manual_validation_worksheet.json"))


    logger.info("PHASE 3: Augmentation (taxonomy completeness)")

    augmenter = ContractAugmenter(real_contracts)
    augmented = augmenter.generate_augmentations(n_per_category=5)
    logger.info(f"Generated {len(augmented)} augmented contracts (DERIVED from real)")

    # All contracts for static analysis = real + augmented
    all_contracts = real_contracts + augmented


    logger.info("PHASE 4: Static Analysis")

    parser = ContractParser()
    stg_builder = STGBuilder()
    contract_stgs = {}
    contract_findings = {}
    parse_results_map = {}
    stats = Counter()

    for rec in all_contracts:
        stats["total"] += 1
        try:
            pr = parser.parse(rec.source_code, rec.file_path)
            parse_results_map[rec.contract_id] = pr
            stats["parsed"] += 1
        except Exception:
            contract_findings[rec.contract_id] = []
            continue

        stg = stg_builder.build(pr)
        if stg and stg.has_valid_workflow:
            contract_stgs[rec.contract_id] = stg
            stats["with_stg"] += 1
            findings = run_all_detectors(stg, pr)
            contract_findings[rec.contract_id] = findings
            if findings:
                stats["with_findings"] += 1
        else:
            contract_findings[rec.contract_id] = []

    sb_stg = sum(1 for r in smartbugs if r.contract_id in contract_stgs)
    sc_stg = sum(1 for r in sc_vuln if r.contract_id in contract_stgs)
    aug_stg = sum(1 for r in augmented if r.contract_id in contract_stgs)
    sb_find = sum(1 for r in smartbugs if contract_findings.get(r.contract_id))
    sc_find = sum(1 for r in sc_vuln if contract_findings.get(r.contract_id))
    aug_find = sum(1 for r in augmented if contract_findings.get(r.contract_id))

    logger.info(f"Static analysis on {stats['total']} contracts:")
    logger.info(f"  Parsed:        {stats['parsed']}/{stats['total']}")
    logger.info(f"  With STG:      {stats['with_stg']}")
    logger.info(f"  With findings: {stats['with_findings']}")
    logger.info(f"  SmartBugs:  {sb_stg} STGs / {sb_find} with findings")
    logger.info(f"  SC Vuln:     {sc_stg} STGs / {sc_find} with findings")
    logger.info(f"  Augmented:  {aug_stg} STGs / {aug_find} with findings")

    fc = Counter()
    for rec in all_contracts:
        for f in contract_findings.get(rec.contract_id, []):
            fc[f.category] += 1
    logger.info(f"  Findings by category: {dict(fc)}")


    logger.info("PHASE 5: Behavioural Analysis (real-STG-derived)")

    sim = TraceSimulator(seed=cfg["project"]["random_seed"])
    ext = BehavioralFeatureExtractor()
    all_traces = []

    # Cap behaviour simulation to keep runtime reasonable: at most 30 contracts
    selected_for_traces = list(contract_stgs.items())[:30]

    for cid, stg in selected_for_traces:
        # Extract REAL transitions from the contract's STG
        transitions = [
            {"source": e.source_state, "dest": e.dest_state, "function": e.function_name}
            for e in stg.edges
        ]
        has_findings = len(contract_findings.get(cid, [])) > 0

        # Adversaries only on contracts with findings
        n_benign = 30 if has_findings else 50
        n_adversarial = 20 if has_findings else 0

        traces = sim.simulate_for_contract(
            contract_id=cid, states=stg.states,
            transitions=transitions,
            n_benign=n_benign, n_adversarial=n_adversarial,
        )
        all_traces.extend(traces)

    logger.info(f"Generated {len(all_traces)} traces from {len(selected_for_traces)} real STGs")

    feature_df = ext.extract_batch(all_traces)
    logger.info(f"Feature matrix: {feature_df.shape}")
    feature_df.to_csv(str(ensure_dir(cfg["paths"]["processed_data"]) / "features.csv"),
                      index=False)


    logger.info("PHASE 6: Anomaly + Trust + Fusion")

    X = np.nan_to_num(feature_df[FEATURE_COLS].values.astype(float),
                       nan=0.0, posinf=1e10, neginf=-1e10)
    y = feature_df["label"].values.astype(int)

    scorer = EnsembleAnomalyScorer(cfg)
    scorer.fit(X, y)
    all_scores = scorer.score(X)
    for m, s in all_scores.items():
        logger.info(f"  {m}: mean={s.mean():.3f}")

    trust_scorer = TrustScorer(cfg)
    trust_list = trust_scorer.compute(feature_df, all_scores["ensemble"])

    ce = CorrelationEngine(cfg)
    rr = RiskRanker()
    all_corr = []
    for i, (idx, row) in enumerate(feature_df.iterrows()):
        cid = row.get("contract_id", "")
        cr = ce.correlate(cid, contract_findings.get(cid, []), row, trust_list[i])
        all_corr.append(cr)

    risk_df = rr.rank(all_corr)
    risk_df["label"] = y
    risk_df.to_csv(str(results_dir / "risk_rankings.csv"), index=False)

    tr = Counter()
    for cr in all_corr:
        for rule in cr.triggered_rules:
            tr[rule] += 1
    logger.info(f"Correlation rules: {dict(tr)}")


    logger.info("PHASE 7: Experiments")

    runner = ExperimentRunner(cfg)
    runner.run_e1_static_detection(all_contracts)
    runner.run_e2_smartbugs_comparison(smartbugs)
    runner.run_e3_behavioral_detection(feature_df)
    runner.run_e4_fusion_benefit(all_contracts, feature_df, contract_findings)
    runner.run_e5_correlation_validity(all_corr)
    runner.run_e6_scalability(all_contracts)
    runner.run_e7_ablation(feature_df, contract_findings)
    runner.save_all_results()


    logger.info("PHASE 8: Figures + Report")
    FigureGenerator(cfg).generate_all(runner.results, risk_df)

    save_json({
        "datasets": {
            "smartbugs":     {"contracts": len(smartbugs), "with_enum": sb_enum,
                              "with_stg": sb_stg, "with_findings": sb_find},
            "scvuln":        {"contracts": len(sc_vuln), "with_enum": sc_vuln_enum,
                              "with_stg": sc_stg, "with_findings": sc_find},
            "scrawld":       {"labels": scrawld["statistics"]["labeled_contracts"],
                              "vuln_distribution": scrawld["statistics"]["vulnerability_distribution"]},
            "augmented":     {"contracts": len(augmented), "with_stg": aug_stg,
                              "with_findings": aug_find,
                              "note": "DERIVED from real contracts; not real"},
            "totals":        {"real": len(real_contracts), "all": len(all_contracts)},
        },
        "annotation_layer": ann_stats,
        "static": {
            "parsed": stats["parsed"], "with_stg": stats["with_stg"],
            "with_findings": stats["with_findings"],
            "findings_by_category": dict(fc),
        },
        "behavioral": {
            "traces": len(all_traces), "feature_shape": list(feature_df.shape),
            "benign": int((y == 0).sum()), "adversarial": int((y == 1).sum()),
        },
        "fusion": {"rules_triggered": dict(tr)},
        "experiments": runner.results,
    }, str(ensure_dir(cfg["paths"]["reports"]) / "final_report.json"))

    logger.info("FlowGuard COMPLETE")
    logger.info(f"  Real contracts: {len(real_contracts)} ({len(smartbugs)} SmartBugs + {len(sc_vuln)} SCVuln)")
    logger.info(f"  ScrawlD labels: {scrawld['statistics']['labeled_contracts']}")
    logger.info(f"  Augmented (derived): {len(augmented)}")
    logger.info(f"  STGs extracted: {stats['with_stg']}")
    logger.info(f"  Behavioural traces: {len(all_traces)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["project"]["random_seed"] = args.seed
    np.random.seed(args.seed)

    log = setup_logger("flowguard", cfg["paths"]["logs"])
    log.info(f"FlowGuard v{cfg['project']['version']} — REAL UPLOADED DATASETS")

    if args.figures_only:
        from src.utils.helpers import load_json
        r = load_json(str(Path(cfg["paths"]["results"]) / "all_results.json"))
        FigureGenerator(cfg).generate_all(r)
        return

    run_full_pipeline(cfg, log)


if __name__ == "__main__":
    main()
