"""
FlowGuard — Dataset Pipeline (REAL UPLOADED DATASETS)
======================================================

Loads the THREE datasets uploaded by the user:

1. SmartBugs Curated  → 143 real Solidity files with SWC labels
   Path:  data/raw/smartbugs_curated/dataset/<category>/*.sol

2. ScrawlD             → 6,780 contract addresses with multi-tool labels
   Path:  data/raw/scrawld/data/{contracts.csv, majority_result.json}
   NOTE:  ScrawlD provides labels ONLY (no source code).

3. Smart Contract Vulnerability Dataset → 6,502 real Solidity contracts
   Path:  data/raw/sc_vuln/{SC_Vuln_8label.csv, SC_4label.csv}
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import Counter

import pandas as pd

from src.utils.helpers import load_config, ensure_dir, contract_hash, save_json

logger = logging.getLogger("flowguard.datasets")


@dataclass
class ContractRecord:
    contract_id: str
    source_code: str
    file_path: str
    dataset: str
    solidity_version: Optional[str] = None
    swc_labels: List[str] = field(default_factory=list)
    fg_labels: List[str] = field(default_factory=list)
    vulnerability_label: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


_PRAGMA_RE = re.compile(r"pragma\s+solidity\s*[\^>=<~!]*\s*([\d]+\.[\d]+\.?[\d]*)", re.IGNORECASE)

def extract_solidity_version(source: str) -> Optional[str]:
    m = _PRAGMA_RE.search(source)
    return m.group(1) if m else None



class SmartBugsLoader:

    CATEGORY_TO_SWC = {
        "reentrancy": "SWC-107",
        "access_control": "SWC-115",
        "arithmetic": "SWC-101",
        "unchecked_low_level_calls": "SWC-104",
        "denial_of_service": "SWC-128",
        "bad_randomness": "SWC-120",
        "front_running": "SWC-114",
        "time_manipulation": "SWC-116",
        "short_addresses": "SWC-130",
        "other": "SWC-000",
    }

    def __init__(self, base_dir: str = None):
        candidates = [
            Path("data/raw/smartbugs_curated/dataset"),
            Path("data/raw/smartbugs/smartbugs-curated-main/dataset"),
            Path("data/raw/smartbugs/dataset"),
        ]
        if base_dir:
            candidates.insert(0, Path(base_dir))
        self.dataset_root = next((c for c in candidates if c and c.exists()), None)

    def load(self) -> List[ContractRecord]:
        records = []
        if self.dataset_root is None:
            logger.warning("SmartBugs path not found")
            return records

        for category_dir in sorted(self.dataset_root.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name.lower().replace("-", "_")
            swc = self.CATEGORY_TO_SWC.get(category, "SWC-000")

            for sol_file in sorted(category_dir.glob("*.sol")):
                try:
                    source = sol_file.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.warning(f"Failed to read {sol_file}: {e}")
                    continue

                cid = f"sb_{contract_hash(source)}"
                rec = ContractRecord(
                    contract_id=cid,
                    source_code=source,
                    file_path=str(sol_file),
                    dataset="smartbugs",
                    solidity_version=extract_solidity_version(source),
                    swc_labels=[swc],
                    metadata={
                        "category": category,
                        "filename": sol_file.name,
                        "has_enum": "enum " in source,
                    },
                )
                records.append(rec)

        logger.info(f"SmartBugs Curated: loaded {len(records)} contracts")
        return records



class ScrawlDLoader:

    def __init__(self, base_dir: str = None):
        candidates = [
            Path("data/raw/scrawld/data"),
            Path("data/raw/scrawld/ScrawlD-main/data"),
        ]
        if base_dir:
            candidates.insert(0, Path(base_dir))
        self.data_dir = next((c for c in candidates if c and c.exists()), None)

    def load_labels(self) -> Dict:
        if self.data_dir is None:
            logger.warning("ScrawlD path not found")
            return {"addresses": [], "per_contract_vulns": {}, "statistics": {}}

        addresses = []
        contracts_csv = self.data_dir / "contracts.csv"
        if contracts_csv.exists():
            df = pd.read_csv(contracts_csv)
            addresses = df["address"].tolist()

        labels = {}
        majority_json = self.data_dir / "majority_result.json"
        if majority_json.exists():
            with open(majority_json) as f:
                labels = json.load(f)

        vuln_dist = Counter()
        per_contract_vulns = {}
        for fname, vulns in labels.items():
            addr = fname.replace("_ext.sol", "")
            per_contract_vulns[addr] = list(vulns.keys())
            for vt in vulns.keys():
                vuln_dist[vt] += 1

        logger.info(f"ScrawlD: {len(addresses)} addresses, "
                    f"{len(labels)} labeled (no source code)")

        return {
            "addresses": addresses,
            "per_contract_vulns": per_contract_vulns,
            "statistics": {
                "total_addresses": len(addresses),
                "labeled_contracts": len(labels),
                "vulnerability_distribution": dict(vuln_dist),
            },
        }



class SCVulnLoader:

    LABEL_TO_SWC = {
        "reentrancy": "SWC-107",
        "unchecked external call": "SWC-104",
        "integer overflow": "SWC-101",
        "block number dependency": "SWC-116",
        "ether strict equality": "SWC-132",
        "timestamp dependency": "SWC-116",
        "dangerous delegatecall": "SWC-112",
        "ether frozen": "SWC-000",
    }

    def __init__(self, base_dir: str = None):
        candidates = [
            Path("data/raw/sc_vuln"),
            ]

        if base_dir:
            candidates.insert(0, Path(base_dir))
        self.data_dir = next((c for c in candidates if c and c.exists()), None)

    def _normalize_label(self, raw: str) -> str:
        if not isinstance(raw, str):
            return "unknown"
        parts = [p for p in raw.replace("\\", "/").split("/") if p]
        if not parts:
            return "unknown"
        last = re.sub(r"\s*\([^)]+\)\s*", "", parts[-1]).strip().lower()
        return last

    def load(self, max_contracts: int = None, only_enum: bool = False,
             max_source_chars: int = 30000) -> List[ContractRecord]:
        records = []
        if self.data_dir is None:
            logger.warning("sc_vuln path not found")
            return records

        seen = set()
        n_skipped_large = 0
        for csv_file in ["SC_Vuln_8label.csv", "SC_4label.csv"]:
            path = self.data_dir / csv_file
            if not path.exists():
                continue

            df = pd.read_csv(path)
            for idx, row in df.iterrows():
                source = str(row.get("code", ""))
                if len(source.strip()) < 50:
                    continue
                if only_enum and "enum " not in source:
                    continue
                # Skip pathologically large contracts (regex parser backtracking)
                if len(source) > max_source_chars:
                    n_skipped_large += 1
                    continue

                h = contract_hash(source)
                if h in seen:
                    continue
                seen.add(h)

                raw_label = row.get("label", "")
                norm = self._normalize_label(raw_label)
                swc = self.LABEL_TO_SWC.get(norm, "SWC-000")

                cid = f"sc_vul_{h}"
                records.append(ContractRecord(
                    contract_id=cid,
                    source_code=source,
                    file_path=f"{csv_file}#{idx}",
                    dataset="sc_vuln",
                    solidity_version=extract_solidity_version(source),
                    swc_labels=[swc],
                    vulnerability_label=int(row.get("label_encoded", -1))
                                        if pd.notna(row.get("label_encoded")) else None,
                    metadata={
                        "filename": str(row.get("filename", "")),
                        "raw_label": raw_label,
                        "normalized_label": norm,
                        "csv_source": csv_file,
                        "has_enum": "enum " in source,
                    },
                ))

                if max_contracts and len(records) >= max_contracts:
                    return records

        logger.info(f"sc_vuln: loaded {len(records)} unique contracts")
        return records


class DatasetManager:

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or load_config()
        self.records: List[ContractRecord] = []
        self.scrawld_data: Dict = {}

    def load_all(self, max_sc_vuln: int = None) -> "DatasetManager":
        sb = SmartBugsLoader().load()
        scvuln = SCVulnLoader().load(max_contracts=max_sc_vuln)
        scrawld = ScrawlDLoader().load_labels()
        self.records = sb + scvuln
        self.scrawld_data = scrawld
        logger.info(f"Total real contracts with source: {len(self.records)}")
        logger.info(f"  SmartBugs:  {len(sb)}")
        logger.info(f"  SC Vulnerability:     {len(scvuln)}")
        logger.info(f"  ScrawlD:    {scrawld['statistics']['labeled_contracts']} (labels only)")
        return self

    def filter_enum_bearing(self) -> List[ContractRecord]:
        """Return contracts with at least one enum declaration."""
        enum_records = [r for r in self.records if "enum " in r.source_code]
        logger.info(f"Enum-bearing contracts: {len(enum_records)}/{len(self.records)}")
        return enum_records

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append({
                "contract_id": r.contract_id,
                "dataset": r.dataset,
                "solidity_version": r.solidity_version,
                "source_length": len(r.source_code),
                "swc_labels": ",".join(r.swc_labels),
                "has_enum": "enum " in r.source_code,
                "vulnerability_label": r.vulnerability_label,
            })
        return pd.DataFrame(rows)

    def summary(self) -> Dict:
        df = self.to_dataframe()
        return {
            "total_contracts_with_source": len(df),
            "by_dataset": df["dataset"].value_counts().to_dict(),
            "with_enum_by_dataset": df.groupby("dataset")["has_enum"].sum().to_dict(),
            "scrawld_labels": self.scrawld_data.get("statistics", {}),
            "by_solidity_version": df["solidity_version"].value_counts().head(10).to_dict(),
            "mean_source_length": int(df["source_length"].mean()),
        }
