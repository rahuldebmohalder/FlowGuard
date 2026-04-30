"""
FlowGuard — Workflow Annotation Layer

"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from collections import Counter

from src.parsers.dataset_loader import ContractRecord
from src.parsers.solidity_parser import ContractParser, ParseResult
from src.static_analysis.stg_builder import STGBuilder, STGResult
from src.static_analysis.detectors import run_all_detectors, Finding

logger = logging.getLogger("flowguard.annotation")


@dataclass
class WorkflowAnnotation:
    contract_id: str
    dataset: str
    has_enum: bool
    has_workflow: bool                  # True if STG was built
    n_states: int = 0
    n_transitions: int = 0
    fg_labels: List[str] = field(default_factory=list)        # ["FG-1", "FG-2", ...]
    findings: List[Dict] = field(default_factory=list)        # detailed findings
    confidence: float = 0.0             # 0..1, derived from finding count + severity
    review_priority: str = "low"        # low | medium | high (for manual sampling)
    derived_from: Optional[str] = None  # contract_id if augmentation


class WorkflowAnnotator:

    def __init__(self):
        self.parser = ContractParser()
        self.stg_builder = STGBuilder()

    def annotate(self, record: ContractRecord) -> WorkflowAnnotation:
        ann = WorkflowAnnotation(
            contract_id=record.contract_id,
            dataset=record.dataset,
            has_enum="enum " in record.source_code,
            has_workflow=False,
        )

        try:
            parse_result = self.parser.parse(record.source_code, record.file_path)
        # except Exception:
        except Exception as e:
            logger.warning(f"Parse failed/timed out for {record.contract_id}: {e}")
            return ann

        stg = self.stg_builder.build(parse_result)
        if stg is None or not stg.has_valid_workflow:
            return ann

        ann.has_workflow = True
        ann.n_states = len(stg.states)
        ann.n_transitions = len(stg.edges)

        findings = run_all_detectors(stg, parse_result)
        if not findings:
            return ann

        ann.fg_labels = sorted(set(f.category for f in findings))
        ann.findings = [
            {
                "category": f.category,
                "severity": f.severity,
                "confidence": f.confidence,
                "description": f.description[:200],
                "affected_functions": f.affected_functions[:3],
            }
            for f in findings
        ]

        # Confidence: weighted by max severity and finding count
        max_sev = max(f.severity for f in findings)
        n_distinct_categories = len(set(f.category for f in findings))
        ann.confidence = min(
            max_sev * (1 + 0.1 * (n_distinct_categories - 1)),
            1.0
        )

        # Review priority for manual sampling
        if ann.confidence >= 0.85 and n_distinct_categories >= 2:
            ann.review_priority = "high"
        elif ann.confidence >= 0.70:
            ann.review_priority = "medium"
        else:
            ann.review_priority = "low"

        return ann

    def annotate_batch(
        self,
        records: List[ContractRecord],
        only_enum_bearing: bool = True,
    ) -> List[WorkflowAnnotation]:
        annotations = []
        # i=1
        for rec in records:
            # i=i+1
            # print(i)
            if only_enum_bearing and "enum " not in rec.source_code:
                continue
            ann = self.annotate(rec)
            annotations.append(ann)
        return annotations


def annotation_statistics(annotations: List[WorkflowAnnotation]) -> Dict:
    total = len(annotations)
    with_workflow = sum(1 for a in annotations if a.has_workflow)
    with_findings = sum(1 for a in annotations if a.fg_labels)

    fg_distribution = Counter()
    for a in annotations:
        for lbl in a.fg_labels:
            fg_distribution[lbl] += 1

    by_dataset = Counter(a.dataset for a in annotations)
    by_dataset_with_findings = Counter(
        a.dataset for a in annotations if a.fg_labels
    )

    by_priority = Counter(a.review_priority for a in annotations if a.has_workflow)

    return {
        "total_annotated": total,
        "with_workflow": with_workflow,
        "with_findings": with_findings,
        "fg_distribution": dict(fg_distribution),
        "by_dataset": dict(by_dataset),
        "by_dataset_with_findings": dict(by_dataset_with_findings),
        "by_review_priority": dict(by_priority),
    }


class ContractAugmenter:

    def __init__(self, base_records: List[ContractRecord]):
        # Use real Kaggle/SmartBugs contracts as the augmentation base pool
        self.base_pool = [r for r in base_records if "enum " in r.source_code]
        logger.info(f"Augmentation base pool: {len(self.base_pool)} real enum-bearing contracts")

    def inject_fg1_bypass(self, base: ContractRecord) -> Optional[ContractRecord]:

        source = base.source_code
        # Find the first enum declaration
        m = re.search(r"enum\s+(\w+)\s*\{([^}]+)\}", source)
        if not m:
            return None
        enum_name = m.group(1)
        values = [v.strip() for v in m.group(2).split(",") if v.strip()]
        if len(values) < 2:
            return None

        # Find the workflow state variable of this enum type
        sv_pat = re.compile(rf"\b{enum_name}\s+(?:public\s+|private\s+|internal\s+)?(\w+)")
        sv_match = sv_pat.search(source)
        if not sv_match:
            return None
        wsv_name = sv_match.group(1)

        # Inject shortcut function before the last closing brace of the contract
        target_state = values[-1]
        injection = (
            f"\n    // FLOWGUARD-AUGMENTATION: FG-1 bypass injection\n"
            f"    function fg1AugmentedBypass() public {{\n"
            f"        {wsv_name} = {enum_name}.{target_state};\n"
            f"    }}\n"
        )

        # Find last } in the contract — naive but effective
        last_brace = source.rfind("}")
        if last_brace < 0:
            return None
        new_source = source[:last_brace] + injection + source[last_brace:]

        from src.utils.helpers import contract_hash
        return ContractRecord(
            contract_id=f"aug_FG1_{contract_hash(new_source)}",
            source_code=new_source,
            file_path=f"augmented/FG1_from_{base.contract_id}.sol",
            dataset="augmented",
            solidity_version=base.solidity_version,
            fg_labels=["FG-1"],
            metadata={
                "augmentation_type": "FG1_bypass",
                "derived_from": base.contract_id,
                "base_dataset": base.dataset,
            },
        )

    def inject_fg2_guard_inconsistency(
        self, base: ContractRecord
    ) -> Optional[ContractRecord]:
        source = base.source_code
        # Find a function with onlyOwner modifier
        fn_pat = re.compile(
            r"function\s+(\w+)\s*\([^)]*\)\s+public\s+onlyOwner",
            re.MULTILINE
        )
        fn_match = fn_pat.search(source)
        if not fn_match:
            return None
        guarded_fn = fn_match.group(1)

        # Inject a duplicate without onlyOwner
        injection = (
            f"\n    // FLOWGUARD-AUGMENTATION: FG-2 guard inconsistency\n"
            f"    function {guarded_fn}Backdoor() public {{\n"
            f"        // mirror of {guarded_fn} but without onlyOwner\n"
            f"    }}\n"
        )

        last_brace = source.rfind("}")
        if last_brace < 0:
            return None
        new_source = source[:last_brace] + injection + source[last_brace:]

        from src.utils.helpers import contract_hash
        return ContractRecord(
            contract_id=f"aug_FG2_{contract_hash(new_source)}",
            source_code=new_source,
            file_path=f"augmented/FG2_from_{base.contract_id}.sol",
            dataset="augmented",
            solidity_version=base.solidity_version,
            fg_labels=["FG-2"],
            metadata={
                "augmentation_type": "FG2_guard_inconsistency",
                "derived_from": base.contract_id,
                "base_dataset": base.dataset,
            },
        )

    def generate_augmentations(
        self, n_per_category: int = 5
    ) -> List[ContractRecord]:
        augmented = []
        injectors = [
            ("FG1", self.inject_fg1_bypass),
            ("FG2", self.inject_fg2_guard_inconsistency),
        ]

        for cat_name, injector in injectors:
            count = 0
            for base in self.base_pool:
                if count >= n_per_category:
                    break
                aug = injector(base)
                if aug is not None:
                    augmented.append(aug)
                    count += 1
            logger.info(f"Generated {count} {cat_name} augmentations")

        return augmented
