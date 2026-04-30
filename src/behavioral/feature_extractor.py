"""
FlowGuard — Behavioral Analysis Engine
Transaction trace simulation, 32-feature extraction, and anomaly detection.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from src.utils.helpers import load_config

logger = logging.getLogger("flowguard.behavioral")
warnings.filterwarnings("ignore", category=FutureWarning)


# Trace Data Structures

@dataclass
class Transaction:
    tx_hash: str
    sender: str
    function_name: str
    value_wei: float
    gas_used: int
    gas_limit: int
    block_number: int
    timestamp: int                 # unix seconds
    success: bool
    state_before: Optional[str] = None
    state_after: Optional[str] = None


@dataclass
class AddressTrace:
    address: str
    contract_id: str
    transactions: List[Transaction]
    label: int = 0                 # 0=benign, 1=anomalous
    be_category: Optional[str] = None  # BE-1..BE-6 for adversarial


#  Behavioral Trace Simulator

class TraceSimulator:


    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        cfg = load_config()
        self.benign_count = cfg["behavioral"]["simulation"]["benign_count"]
        self.adv_count = cfg["behavioral"]["simulation"]["adversarial_count"]
        self.min_tx = cfg["behavioral"]["simulation"]["min_tx_per_address"]
        self.max_tx = cfg["behavioral"]["simulation"]["max_tx_per_address"]
        self.time_window = cfg["behavioral"]["simulation"]["time_window_blocks"]

    def simulate_for_contract(
        self,
        contract_id: str,
        states: List[str],
        transitions: List[dict],  # [{source, dest, function}, ...]
        n_benign: int = None,
        n_adversarial: int = None,
    ) -> List[AddressTrace]:

        n_benign = n_benign or self.benign_count
        n_adversarial = n_adversarial or self.adv_count

        # Split benign: 70% clean, 30% noisy
        n_clean_benign = int(n_benign * 0.70)
        n_noisy_benign = n_benign - n_clean_benign

        # Split adversarial: 60% obvious, 40% stealthy
        n_obvious_adv = int(n_adversarial * 0.60)
        n_stealthy_adv = n_adversarial - n_obvious_adv

        traces = []
        be_categories = ["BE-1", "BE-2", "BE-3", "BE-4", "BE-5", "BE-6"]

        # 1. Clean benign
        for i in range(n_clean_benign):
            addr = f"0xBenign{i:04d}"
            txs = self._generate_benign_trace(addr, states, transitions)
            traces.append(AddressTrace(
                address=addr, contract_id=contract_id,
                transactions=txs, label=0,
            ))

        # 2. Noisy benign — high-frequency legitimate usage
        for i in range(n_noisy_benign):
            addr = f"0xNoisyBenign{i:04d}"
            txs = self._generate_noisy_benign_trace(addr, states, transitions)
            traces.append(AddressTrace(
                address=addr, contract_id=contract_id,
                transactions=txs, label=0,
            ))

        # 3. Obvious adversarial
        for i in range(n_obvious_adv):
            addr = f"0xAdversary{i:04d}"
            be_cat = be_categories[i % len(be_categories)]
            txs = self._generate_adversarial_trace(
                addr, states, transitions, be_cat
            )
            traces.append(AddressTrace(
                address=addr, contract_id=contract_id,
                transactions=txs, label=1, be_category=be_cat,
            ))

        # 4. Stealthy adversarial — looks benign on features
        for i in range(n_stealthy_adv):
            addr = f"0xStealth{i:04d}"
            txs = self._generate_stealthy_adversarial(addr, states, transitions)
            traces.append(AddressTrace(
                address=addr, contract_id=contract_id,
                transactions=txs, label=1, be_category="BE-stealth",
            ))

        return traces

    def _generate_benign_trace(
        self, address: str, states: List[str], transitions: List[dict]
    ) -> List[Transaction]:
        n_tx = self.rng.integers(self.min_tx, min(self.max_tx, 100))
        txs = []
        current_block = self.rng.integers(10_000_000, 15_000_000)
        current_state_idx = 0

        fn_names = list({t["function"] for t in transitions}) or ["transfer"]

        for j in range(n_tx):
            # Inter-block interval: Poisson(λ=50) — avg 50 blocks apart
            gap = max(1, int(self.rng.poisson(50)))
            current_block += gap
            timestamp = 1_600_000_000 + current_block * 12  # ~12s per block

            # Forward workflow traversal with some view-only calls
            if transitions and self.rng.random() < 0.3 and current_state_idx < len(states) - 1:
                # State transition
                matching = [t for t in transitions
                            if t.get("source") == states[current_state_idx]]
                if matching:
                    t = matching[self.rng.integers(len(matching))]
                    fn = t["function"]
                    state_before = t["source"]
                    state_after = t["dest"]
                    if state_after in states:
                        current_state_idx = states.index(state_after)
                else:
                    fn = self.rng.choice(fn_names)
                    state_before = states[min(current_state_idx, len(states) - 1)]
                    state_after = state_before
            else:
                fn = self.rng.choice(fn_names)
                state_before = states[min(current_state_idx, len(states) - 1)]
                state_after = state_before

            value = float(self.rng.lognormal(mean=18, sigma=2))  # in wei
            gas_limit = 200_000
            gas_used = int(gas_limit * self.rng.uniform(0.3, 0.9))

            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=address,
                function_name=fn,
                value_wei=value,
                gas_used=gas_used,
                gas_limit=gas_limit,
                block_number=int(current_block),
                timestamp=int(timestamp),
                success=self.rng.random() > 0.02,  # 2% failure rate
                state_before=state_before,
                state_after=state_after,
            ))

        return txs

    def _generate_noisy_benign_trace(
        self, address: str, states: List[str], transitions: List[dict]
    ) -> List[Transaction]:

        n_tx = self.rng.integers(60, 200)
        txs = []
        current_block = self.rng.integers(10_000_000, 15_000_000)
        current_state_idx = 0
        fn_names = list({t["function"] for t in transitions}) or ["transfer"]

        for j in range(n_tx):
            # Short intervals — looks like a bot, Poisson(λ=8)
            gap = max(1, int(self.rng.poisson(8)))
            current_block += gap
            timestamp = 1_600_000_000 + current_block * 12

            # Forward workflow traversal (still orderly)
            if transitions and self.rng.random() < 0.3 and current_state_idx < len(states) - 1:
                matching = [t for t in transitions
                            if t.get("source") == states[current_state_idx]]
                if matching:
                    t = matching[self.rng.integers(len(matching))]
                    fn = t["function"]
                    state_before = t["source"]
                    state_after = t["dest"]
                    if state_after in states:
                        current_state_idx = states.index(state_after)
                else:
                    fn = self.rng.choice(fn_names)
                    state_before = states[min(current_state_idx, len(states) - 1)]
                    state_after = state_before
            else:
                fn = self.rng.choice(fn_names)
                state_before = states[min(current_state_idx, len(states) - 1)]
                state_after = state_before

            value = float(self.rng.lognormal(mean=19, sigma=2.5))
            gas_limit = 250_000
            gas_used = int(gas_limit * self.rng.uniform(0.5, 0.95))

            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=address,
                function_name=fn,
                value_wei=value,
                gas_used=gas_used,
                gas_limit=gas_limit,
                block_number=int(current_block),
                timestamp=int(timestamp),
                success=self.rng.random() > 0.04,  # 4% failure (slightly higher)
                state_before=state_before,
                state_after=state_after,
            ))
        return txs

    def _generate_stealthy_adversarial(
        self, address: str, states: List[str], transitions: List[dict]
    ) -> List[Transaction]:

        n_tx = self.rng.integers(40, 100)
        txs = []
        current_block = self.rng.integers(10_000_000, 15_000_000)
        fn_names = list({t["function"] for t in transitions}) or ["transfer"]

        for j in range(n_tx):
            # Normal-looking inter-block intervals (benign timing)
            gap = max(1, int(self.rng.poisson(35)))
            current_block += gap
            timestamp = 1_600_000_000 + current_block * 12

            # Key difference: cycle through ALL states rapidly
            # (high state_transition_freq, high reverse_transition_count)
            if transitions and self.rng.random() < 0.6:
                t = transitions[j % len(transitions)]
                fn = t["function"]
                state_before = t["source"]
                state_after = t["dest"]
            else:
                fn = self.rng.choice(fn_names)
                state_before = states[j % len(states)] if states else None
                state_after = states[(j + 1) % len(states)] if states else None

            # Normal-looking values
            value = float(self.rng.lognormal(mean=18, sigma=2))
            gas_limit = 200_000
            gas_used = int(gas_limit * self.rng.uniform(0.3, 0.85))

            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=address,
                function_name=fn,
                value_wei=value,
                gas_used=gas_used,
                gas_limit=gas_limit,
                block_number=int(current_block),
                timestamp=int(timestamp),
                success=self.rng.random() > 0.03,
                state_before=state_before,
                state_after=state_after,
            ))
        return txs

    def _generate_adversarial_trace(
        self, address: str, states: List[str],
        transitions: List[dict], be_category: str,
    ) -> List[Transaction]:
        generators = {
            "BE-1": self._gen_rapid_cycling,
            "BE-2": self._gen_flash_attack,
            "BE-3": self._gen_threshold_evasion,
            "BE-4": self._gen_role_oscillation,
            "BE-5": self._gen_state_probing,
            "BE-6": self._gen_multi_account,
        }
        gen_fn = generators.get(be_category, self._gen_rapid_cycling)
        return gen_fn(address, states, transitions)

    def _gen_rapid_cycling(self, addr, states, transitions):
        n_tx = self.rng.integers(50, 200)
        txs = []
        block = self.rng.integers(10_000_000, 15_000_000)
        fn_names = [t["function"] for t in transitions] or ["call"]

        for j in range(n_tx):
            block += self.rng.integers(1, 3)  # 1-2 blocks apart (very fast)
            s_idx = j % max(len(states), 1)
            s_next = (j + 1) % max(len(states), 1)
            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=addr,
                function_name=self.rng.choice(fn_names),
                value_wei=float(self.rng.lognormal(20, 3)),
                gas_used=int(180_000 * self.rng.uniform(0.7, 1.0)),
                gas_limit=300_000,
                block_number=int(block),
                timestamp=1_600_000_000 + int(block) * 12,
                success=True,
                state_before=states[s_idx] if states else None,
                state_after=states[s_next] if states else None,
            ))
        return txs

    def _gen_flash_attack(self, addr, states, transitions):
        n_tx = self.rng.integers(5, 15)
        block = self.rng.integers(10_000_000, 15_000_000)
        fn_names = [t["function"] for t in transitions] or ["execute"]
        txs = []
        for j in range(n_tx):
            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=addr,
                function_name=fn_names[j % len(fn_names)],
                value_wei=float(self.rng.lognormal(22, 1)),  # large values
                gas_used=int(250_000 * self.rng.uniform(0.8, 1.0)),
                gas_limit=500_000,
                block_number=int(block),  # same block!
                timestamp=1_600_000_000 + int(block) * 12,
                success=True,
                state_before=states[j % len(states)] if states else None,
                state_after=states[(j + 1) % len(states)] if states else None,
            ))
        return txs

    def _gen_threshold_evasion(self, addr, states, transitions):
        n_tx = self.rng.integers(30, 100)
        threshold = 1e18  # 1 ETH in wei
        block = self.rng.integers(10_000_000, 15_000_000)
        fn_names = [t["function"] for t in transitions] or ["withdraw"]
        txs = []
        for j in range(n_tx):
            block += self.rng.integers(5, 20)
            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=addr,
                function_name=self.rng.choice(fn_names),
                value_wei=float(threshold * self.rng.uniform(0.95, 0.99)),
                gas_used=int(100_000 * self.rng.uniform(0.5, 0.9)),
                gas_limit=200_000,
                block_number=int(block),
                timestamp=1_600_000_000 + int(block) * 12,
                success=True,
                state_before=states[0] if states else None,
                state_after=states[0] if states else None,
            ))
        return txs

    def _gen_role_oscillation(self, addr, states, transitions):
        n_tx = self.rng.integers(40, 120)
        block = self.rng.integers(10_000_000, 15_000_000)
        admin_fns = ["setOwner", "transferOwnership", "pause", "unpause"]
        user_fns = [t["function"] for t in transitions] or ["transfer"]
        txs = []
        for j in range(n_tx):
            block += self.rng.integers(1, 5)
            fn = self.rng.choice(admin_fns) if j % 2 == 0 else self.rng.choice(user_fns)
            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=addr,
                function_name=fn,
                value_wei=float(self.rng.lognormal(18, 2)),
                gas_used=int(150_000 * self.rng.uniform(0.4, 0.9)),
                gas_limit=200_000,
                block_number=int(block),
                timestamp=1_600_000_000 + int(block) * 12,
                success=self.rng.random() > 0.15,  # higher failure rate
                state_before=states[j % len(states)] if states else None,
                state_after=states[j % len(states)] if states else None,
            ))
        return txs

    def _gen_state_probing(self, addr, states, transitions):
        n_reads = self.rng.integers(20, 60)
        block = self.rng.integers(10_000_000, 15_000_000)
        view_fns = ["getState", "balanceOf", "getOwner", "totalSupply"]
        write_fns = [t["function"] for t in transitions] or ["exploit"]
        txs = []

        # Phase 1: probing
        for j in range(n_reads):
            block += self.rng.integers(1, 3)
            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=addr,
                function_name=self.rng.choice(view_fns),
                value_wei=0,
                gas_used=int(30_000 * self.rng.uniform(0.5, 1.0)),
                gas_limit=50_000,
                block_number=int(block),
                timestamp=1_600_000_000 + int(block) * 12,
                success=True,
                state_before=states[0] if states else None,
                state_after=states[0] if states else None,
            ))

        # Phase 2: targeted writes
        for j in range(self.rng.integers(3, 8)):
            block += 1
            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=addr,
                function_name=self.rng.choice(write_fns),
                value_wei=float(self.rng.lognormal(22, 1)),
                gas_used=int(250_000 * self.rng.uniform(0.8, 1.0)),
                gas_limit=500_000,
                block_number=int(block),
                timestamp=1_600_000_000 + int(block) * 12,
                success=True,
                state_before=states[0] if states else None,
                state_after=states[-1] if states else None,
            ))
        return txs

    def _gen_multi_account(self, addr, states, transitions):

        n_tx = self.rng.integers(40, 100)
        block = self.rng.integers(10_000_000, 15_000_000)
        fn_names = [t["function"] for t in transitions] or ["call"]
        txs = []
        fixed_interval = self.rng.integers(3, 8)  # very regular
        for j in range(n_tx):
            block += fixed_interval  # nearly constant interval
            txs.append(Transaction(
                tx_hash=f"0x{self.rng.integers(0, 2**63):016x}",
                sender=addr,
                function_name=fn_names[j % len(fn_names)],
                value_wei=float(self.rng.lognormal(19, 0.5)),  # low variance
                gas_used=int(100_000),  # constant gas
                gas_limit=200_000,
                block_number=int(block),
                timestamp=1_600_000_000 + int(block) * 12,
                success=True,
                state_before=states[j % len(states)] if states else None,
                state_after=states[(j + 1) % len(states)] if states else None,
            ))
        return txs


#  32-Feature Behavioral Extractor

class BehavioralFeatureExtractor:

    FEATURE_NAMES = [
        # A: Transaction Pattern (0-7)
        "tx_count", "mean_inter_tx_interval", "std_inter_tx_interval",
        "max_burst_rate", "unique_fn_count", "failed_tx_ratio",
        "mean_gas_ratio", "tx_value_entropy",
        # B: State Interaction (8-15)
        "unique_states_visited", "state_transition_freq",
        "most_freq_transition_ratio", "reverse_transition_count",
        "terminal_state_reach_count", "mean_time_in_state",
        "state_sequence_entropy", "repeated_state_visit_count",
        # C: Role & Access (16-23)
        "distinct_roles_assumed", "role_change_freq",
        "admin_fn_call_ratio", "guard_failure_rate",
        "delegatecall_count", "contract_creation_count",
        "unique_interacted_contracts", "self_ref_call_count",
        # D: Value & Economic (24-31)
        "total_value_transferred", "mean_value_per_tx",
        "value_variance", "max_single_tx_value",
        "cumulative_balance_delta", "value_gini",
        "token_transfer_count", "flash_loan_indicator",
    ]

    ADMIN_FN_KEYWORDS = {
        "owner", "admin", "pause", "unpause", "set", "update",
        "transfer_ownership", "transferownership", "grant", "revoke",
        "upgrade", "migrate", "emergency", "kill", "destroy",
    }

    def extract(self, trace: AddressTrace) -> np.ndarray:
        txs = trace.transactions
        if not txs:
            return np.zeros(32)

        features = np.zeros(32)

        # Category A: Transaction Pattern
        features[0] = len(txs)

        timestamps = sorted([t.timestamp for t in txs])
        if len(timestamps) > 1:
            intervals = np.diff(timestamps).astype(float)
            features[1] = np.mean(intervals)
            features[2] = np.std(intervals)
        else:
            features[1] = 0
            features[2] = 0

        # Burst rate: max txs in any single block
        blocks = [t.block_number for t in txs]
        if blocks:
            block_counts = pd.Series(blocks).value_counts()
            features[3] = block_counts.max()

        features[4] = len({t.function_name for t in txs})

        failed = sum(1 for t in txs if not t.success)
        features[5] = failed / max(len(txs), 1)

        gas_ratios = [t.gas_used / max(t.gas_limit, 1) for t in txs]
        features[6] = np.mean(gas_ratios) if gas_ratios else 0

        values = [t.value_wei for t in txs if t.value_wei > 0]
        if values:
            hist, _ = np.histogram(np.log1p(values), bins=10)
            prob = hist / hist.sum() if hist.sum() > 0 else hist
            features[7] = float(scipy_entropy(prob + 1e-10))
        else:
            features[7] = 0

        # Category B: State Interaction
        states_visited = set()
        state_transitions = []
        for t in txs:
            if t.state_before:
                states_visited.add(t.state_before)
            if t.state_after:
                states_visited.add(t.state_after)
            if t.state_before and t.state_after and t.state_before != t.state_after:
                state_transitions.append((t.state_before, t.state_after))

        features[8] = len(states_visited)
        features[9] = len(state_transitions) / max(len(txs), 1)

        if state_transitions:
            trans_counts = pd.Series(
                [f"{a}->{b}" for a, b in state_transitions]
            ).value_counts()
            features[10] = trans_counts.iloc[0] / len(state_transitions)

            # Reverse transitions
            reverse_count = 0
            trans_set = set(state_transitions)
            for a, b in trans_set:
                if (b, a) in trans_set:
                    reverse_count += 1
            features[11] = reverse_count

            # Terminal state reaches
            terminal_keywords = {"completed", "closed", "finished",
                                 "cancelled", "terminated", "finalized"}
            terminal_reaches = sum(
                1 for _, b in state_transitions
                if b.lower() in terminal_keywords
            )
            features[12] = terminal_reaches
        else:
            features[10] = 0
            features[11] = 0
            features[12] = 0

        # Time in state
        if len(txs) > 1 and any(t.state_before for t in txs):
            state_durations = []
            for i in range(len(txs) - 1):
                if txs[i].state_before == txs[i + 1].state_before:
                    dur = txs[i + 1].timestamp - txs[i].timestamp
                    state_durations.append(dur)
            features[13] = np.mean(state_durations) if state_durations else 0
        else:
            features[13] = 0

        # State sequence entropy
        state_seq = [t.state_before for t in txs if t.state_before]
        if state_seq:
            state_counts = pd.Series(state_seq).value_counts(normalize=True)
            features[14] = float(scipy_entropy(state_counts.values + 1e-10))
        else:
            features[14] = 0

        # Repeated visits
        if state_seq:
            visit_counts = pd.Series(state_seq).value_counts()
            features[15] = int(visit_counts[visit_counts > 1].sum())
        else:
            features[15] = 0

        # Category C: Role & Access
        fn_names_lower = [t.function_name.lower() for t in txs]
        admin_calls = sum(
            1 for fn in fn_names_lower
            if any(kw in fn for kw in self.ADMIN_FN_KEYWORDS)
        )
        non_admin_fns = [fn for fn in fn_names_lower
                         if not any(kw in fn for kw in self.ADMIN_FN_KEYWORDS)]

        features[16] = 1 + (1 if admin_calls > 0 else 0)  # distinct roles
        features[17] = 0  # role change freq (approx: transitions between admin/user)
        if len(fn_names_lower) > 1:
            role_changes = 0
            for i in range(len(fn_names_lower) - 1):
                is_admin_i = any(kw in fn_names_lower[i] for kw in self.ADMIN_FN_KEYWORDS)
                is_admin_j = any(kw in fn_names_lower[i + 1] for kw in self.ADMIN_FN_KEYWORDS)
                if is_admin_i != is_admin_j:
                    role_changes += 1
            features[17] = role_changes / max(len(fn_names_lower) - 1, 1)

        features[18] = admin_calls / max(len(txs), 1)
        features[19] = features[5]  # guard failure rate ≈ failed tx ratio
        features[20] = sum(1 for fn in fn_names_lower if "delegatecall" in fn)
        features[21] = sum(1 for fn in fn_names_lower if "create" in fn)
        features[22] = 1  # single contract context in simulation
        features[23] = sum(1 for fn in fn_names_lower if "self" in fn or "this" in fn)

        # Category D: Value & Economic
        all_values = np.array([t.value_wei for t in txs], dtype=float)
        features[24] = np.sum(all_values)
        features[25] = np.mean(all_values) if len(all_values) > 0 else 0
        features[26] = np.var(all_values) if len(all_values) > 0 else 0
        features[27] = np.max(all_values) if len(all_values) > 0 else 0
        features[28] = np.sum(all_values)  # cumulative balance delta (simplified)

        # Gini coefficient
        if len(all_values) > 0 and np.sum(all_values) > 0:
            sorted_vals = np.sort(all_values)
            n = len(sorted_vals)
            index = np.arange(1, n + 1)
            features[29] = float(
                (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals))
                / (n * np.sum(sorted_vals) + 1e-10)
            )
        else:
            features[29] = 0

        # Token transfer count (heuristic: functions with "transfer" in name)
        features[30] = sum(1 for fn in fn_names_lower if "transfer" in fn)

        # Flash loan indicator: borrow + repay in same block
        if blocks:
            block_fns = {}
            for t in txs:
                if t.block_number not in block_fns:
                    block_fns[t.block_number] = []
                block_fns[t.block_number].append(t.function_name.lower())
            flash_loan = any(
                any("borrow" in fn or "flash" in fn for fn in fns)
                and any("repay" in fn or "return" in fn for fn in fns)
                for fns in block_fns.values()
            )
            features[31] = 1.0 if flash_loan else 0.0
        else:
            features[31] = 0.0

        return features

    def extract_batch(self, traces: List[AddressTrace]) -> pd.DataFrame:
        """Extract features for all traces, returning a labeled DataFrame."""
        rows = []
        for trace in traces:
            feats = self.extract(trace)
            row = {name: feats[i] for i, name in enumerate(self.FEATURE_NAMES)}
            row["address"] = trace.address
            row["contract_id"] = trace.contract_id
            row["label"] = trace.label
            row["be_category"] = trace.be_category or ""
            rows.append(row)

        df = pd.DataFrame(rows)
        return df
