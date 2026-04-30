"""
FlowGuard — Fusion Layer
Trust score computation, static-behavioral correlation, and risk ranking.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.static_analysis.detectors import Finding
from src.utils.helpers import load_config

logger = logging.getLogger("flowguard.fusion")


# Trust Score

@dataclass
class TrustComponents:
    address: str
    reputation: float       # R(v) — historical quality
    consistency: float      # C(v) — behavioral regularity
    engagement: float       # E(v) — legitimate interaction depth
    anomaly: float          # A(v) — anomaly detector output
    trust_score: float      # T(v) = w1*R + w2*C + w3*E − w4*A


class TrustScorer:


    def __init__(self, cfg: Dict = None):
        cfg = cfg or load_config()
        tw = cfg["behavioral"]["trust_weights"]
        self.w1 = tw["w1_reputation"]       # 0.28
        self.w2 = tw["w2_consistency"]       # 0.32
        self.w3 = tw["w3_engagement"]        # 0.25
        self.w4 = tw["w4_anomaly"]           # 0.30
        self.threshold = cfg["behavioral"]["detection_threshold"]  # 0.65

    def compute(
        self,
        features: pd.DataFrame,
        anomaly_scores: np.ndarray,
    ) -> List[TrustComponents]:

        #  Build a normalised copy of the numeric features
        from sklearn.preprocessing import MinMaxScaler
        numeric_cols = [
            "tx_count", "mean_inter_tx_interval", "std_inter_tx_interval",
            "max_burst_rate", "unique_fn_count", "failed_tx_ratio",
            "mean_gas_ratio", "tx_value_entropy",
            "unique_states_visited", "state_transition_freq",
            "most_freq_transition_ratio", "reverse_transition_count",
            "terminal_state_reach_count", "mean_time_in_state",
            "state_sequence_entropy", "repeated_state_visit_count",
            "distinct_roles_assumed", "role_change_freq",
            "admin_fn_call_ratio", "guard_failure_rate",
            "delegatecall_count", "contract_creation_count",
            "unique_interacted_contracts", "self_ref_call_count",
            "total_value_transferred", "mean_value_per_tx",
            "value_variance", "max_single_tx_value",
            "cumulative_balance_delta", "value_gini",
            "token_transfer_count", "flash_loan_indicator",
        ]
        present = [c for c in numeric_cols if c in features.columns]
        norm_df = features.copy()
        if present:
            scaler = MinMaxScaler()
            norm_df[present] = scaler.fit_transform(
                features[present].values.astype(float)
            )

        results = []
        for i, (idx, row) in enumerate(norm_df.iterrows()):
            R = self._reputation(row)
            C = self._consistency(row)
            E = self._engagement(row)
            A = float(anomaly_scores[i]) if i < len(anomaly_scores) else 0.0

            T = self.w1 * R + self.w2 * C + self.w3 * E - self.w4 * A
            T = np.clip(T, 0.0, 1.0)

            results.append(TrustComponents(
                address=features.iloc[i].get("address", f"addr_{i}"),
                reputation=R,
                consistency=C,
                engagement=E,
                anomaly=A,
                trust_score=T,
            ))

        return results

    def _reputation(self, row: pd.Series) -> float:
        fail_rate = row.get("failed_tx_ratio", 0)
        tx_count = row.get("tx_count", 0)       # already [0,1] after scaling
        success_factor = 1.0 - float(fail_rate)
        age_factor = min(float(tx_count) * 2.0, 1.0)  # scale so median ≈ 0.5
        return 0.5 * age_factor + 0.5 * success_factor

    def _consistency(self, row: pd.Series) -> float:
        # After MinMax normalisation the raw CV problem disappears
        std_norm = float(row.get("std_inter_tx_interval", 0.5))
        val_var_norm = float(row.get("value_variance", 0.5))
        time_consistency = 1.0 - std_norm        # low std → high consistency
        value_consistency = 1.0 - val_var_norm
        return np.clip(0.5 * time_consistency + 0.5 * value_consistency, 0, 1)

    def _engagement(self, row: pd.Series) -> float:
        fn_diversity = min(float(row.get("unique_fn_count", 0)) * 2.0, 1.0)
        state_diversity = min(float(row.get("unique_states_visited", 0)) * 2.0, 1.0)
        admin_ratio = float(row.get("admin_fn_call_ratio", 0))
        admin_penalty = max(0.0, admin_ratio - 0.3) * 2.0
        engagement = (0.4 * fn_diversity + 0.4 * state_diversity
                      + 0.2 * (1.0 - admin_penalty))
        return np.clip(engagement, 0.0, 1.0)

    def to_dataframe(self, trust_list: List[TrustComponents]) -> pd.DataFrame:
        rows = []
        for tc in trust_list:
            rows.append({
                "address": tc.address,
                "reputation": tc.reputation,
                "consistency": tc.consistency,
                "engagement": tc.engagement,
                "anomaly": tc.anomaly,
                "trust_score": tc.trust_score,
                "is_anomalous": tc.trust_score < self.threshold,
            })
        return pd.DataFrame(rows)


# Correlation Engine

@dataclass
class CorrelationResult:
    contract_id: str
    address: str
    static_severity: float          # S(c) — max static finding severity
    behavioral_score: float         # B(v) — 1 - T(v)
    correlation_score: float        # Corr(c, v)
    fused_risk_score: float         # R(c, v) = αS + βB + γCorr
    triggered_rules: List[str]
    findings_matched: List[str]     # FG categories matched


class CorrelationEngine:

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or load_config()
        self.alpha = self.cfg["fusion"]["alpha"]   # 0.40
        self.beta = self.cfg["fusion"]["beta"]     # 0.35
        self.gamma = self.cfg["fusion"]["gamma"]   # 0.25

        # Load correlation rules
        self.rules = []
        for rule_id, rule_cfg in self.cfg["fusion"]["correlation_rules"].items():
            self.rules.append({
                "id": rule_id,
                "static": rule_cfg["static"],
                "behavioral": rule_cfg["behavioral"],
                "boost": rule_cfg["boost"],
            })

    def correlate(
        self,
        contract_id: str,
        static_findings: List[Finding],
        behavioral_features: pd.Series,
        trust_components: TrustComponents,
    ) -> CorrelationResult:

        # S(c): max severity across all static findings
        if static_findings:
            static_severity = max(f.severity for f in static_findings)
        else:
            static_severity = 0.0

        # B(v): behavioral anomaly score = 1 - trust
        behavioral_score = 1.0 - trust_components.trust_score

        # Corr(c, v): check correlation rules
        corr_score = 0.0
        triggered_rules = []
        findings_matched = []

        fg_categories = {f.category.replace("-", "") for f in static_findings}

        for rule in self.rules:
            static_match = (
                rule["static"] == "ANY"
                or rule["static"] in fg_categories
                or rule["static"].replace("-", "") in fg_categories
            )

            behavioral_match = self._check_behavioral_condition(
                rule["behavioral"], behavioral_features
            )

            if static_match and behavioral_match:
                corr_score += rule["boost"]
                triggered_rules.append(rule["id"])
                if rule["static"] != "ANY":
                    findings_matched.append(rule["static"])

        corr_score = min(corr_score, 1.0)

        # Fused risk score
        fused = (
            self.alpha * static_severity
            + self.beta * behavioral_score
            + self.gamma * corr_score
        )

        return CorrelationResult(
            contract_id=contract_id,
            address=trust_components.address,
            static_severity=static_severity,
            behavioral_score=behavioral_score,
            correlation_score=corr_score,
            fused_risk_score=fused,
            triggered_rules=triggered_rules,
            findings_matched=findings_matched,
        )

    def _check_behavioral_condition(
        self, be_category: str, features: pd.Series
    ) -> bool:
        checks = {
            "BE1": self._check_rapid_cycling,
            "BE2": self._check_flash_attack,
            "BE3": self._check_threshold_evasion,
            "BE4": self._check_role_oscillation,
            "BE5": self._check_state_probing,
            "BE6": self._check_multi_account,
        }

        # Normalize category key
        key = be_category.replace("-", "")
        check_fn = checks.get(key)
        if check_fn is None:
            return False

        try:
            return check_fn(features)
        except (KeyError, TypeError):
            return False

    def _check_rapid_cycling(self, f: pd.Series) -> bool:

        high_transition_freq = f.get("state_transition_freq", 0) > 0.35
        fast_timing = f.get("mean_inter_tx_interval", 1e9) < 200
        high_reverse = f.get("reverse_transition_count", 0) > 2
        return high_transition_freq and (fast_timing or high_reverse)

    def _check_flash_attack(self, f: pd.Series) -> bool:
        return f.get("max_burst_rate", 0) >= 4

    def _check_threshold_evasion(self, f: pd.Series) -> bool:
        gini = f.get("value_gini", 0)
        tx_count = f.get("tx_count", 0)
        return gini < 0.15 and tx_count > 20

    def _check_role_oscillation(self, f: pd.Series) -> bool:
        return f.get("role_change_freq", 0) > 0.3

    def _check_state_probing(self, f: pd.Series) -> bool:
        return (f.get("unique_fn_count", 0) >= 4
                and f.get("admin_fn_call_ratio", 0) < 0.1
                and f.get("tx_count", 0) > 15)

    def _check_multi_account(self, f: pd.Series) -> bool:
        mean_int = f.get("mean_inter_tx_interval", 1)
        std_int = f.get("std_inter_tx_interval", 1)
        cv = std_int / (mean_int + 1e-10)
        return cv < 0.15 and f.get("tx_count", 0) > 20

    def correlate_batch(
        self,
        contract_id: str,
        static_findings: List[Finding],
        feature_df: pd.DataFrame,
        trust_list: List[TrustComponents],
    ) -> List[CorrelationResult]:
        results = []
        for i, (idx, row) in enumerate(feature_df.iterrows()):
            tc = trust_list[i] if i < len(trust_list) else TrustComponents(
                address=row.get("address", ""), reputation=0.5,
                consistency=0.5, engagement=0.5, anomaly=0.5, trust_score=0.5,
            )
            result = self.correlate(
                contract_id=contract_id,
                static_findings=static_findings,
                behavioral_features=row,
                trust_components=tc,
            )
            results.append(result)
        return results


# Risk Ranking

class RiskRanker:

    def rank(
        self, results: List[CorrelationResult], top_k: int = None
    ) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append({
                "contract_id": r.contract_id,
                "address": r.address,
                "static_severity": round(r.static_severity, 4),
                "behavioral_score": round(r.behavioral_score, 4),
                "correlation_score": round(r.correlation_score, 4),
                "fused_risk_score": round(r.fused_risk_score, 4),
                "triggered_rules": ",".join(r.triggered_rules),
                "findings_matched": ",".join(r.findings_matched),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("fused_risk_score", ascending=False)

        if top_k:
            df = df.head(top_k)

        return df.reset_index(drop=True)
