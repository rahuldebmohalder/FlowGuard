"""
FlowGuard — Static Vulnerability Detectors (FG-1 through FG-8)
Each detector operates on a STGResult and returns a list of Finding objects.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from itertools import combinations

import networkx as nx

from src.static_analysis.stg_builder import STGResult

logger = logging.getLogger("flowguard.detectors")


# Finding Data Structure

@dataclass
class Finding:
    category: str            # "FG-1", "FG-2", ..., "FG-8"
    severity: float          # 0.0 – 1.0
    description: str
    evidence: Dict = field(default_factory=dict)
    affected_states: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 – 1.0


# Base Detector

class BaseDetector:

    CATEGORY: str = ""
    BASE_SEVERITY: float = 0.5

    def detect(self, stg: STGResult) -> List[Finding]:
        raise NotImplementedError

    def _make_finding(self, desc: str, severity: float = None, **kwargs) -> Finding:
        return Finding(
            category=self.CATEGORY,
            severity=severity or self.BASE_SEVERITY,
            description=desc,
            **kwargs,
        )


# FG-1: State-Transition Bypass

class FG1_BypassDetector(BaseDetector):

    CATEGORY = "FG-1"
    BASE_SEVERITY = 0.95

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []
        G = stg.graph

        if stg.initial_state is None or not stg.terminal_states:
            return findings

        for terminal in stg.terminal_states:
            if terminal not in G.nodes or stg.initial_state not in G.nodes:
                continue

            try:
                all_paths = list(nx.all_simple_paths(
                    G, stg.initial_state, terminal, cutoff=len(stg.states) + 2
                ))
            except nx.NetworkXError:
                continue

            if len(all_paths) < 2:
                continue

            # Find states that appear in ALL paths (required intermediate)
            path_sets = [set(p[1:-1]) for p in all_paths]  # exclude start/end
            if not path_sets:
                continue
            required = path_sets[0]
            for ps in path_sets[1:]:
                required &= ps

            # Now check if any path skips a required state
            for path in all_paths:
                intermediate = set(path[1:-1])
                skipped = required - intermediate
                if skipped:
                    findings.append(self._make_finding(
                        f"Bypass detected: path {' → '.join(path)} skips "
                        f"required states {skipped}",
                        evidence={
                            "path": path,
                            "skipped_states": list(skipped),
                            "terminal": terminal,
                        },
                        affected_states=list(skipped),
                        confidence=0.9,
                    ))

        # Also check for __ANY__ edges that jump directly to terminal
        for terminal in stg.terminal_states:
            if G.has_edge("__ANY__", terminal):
                edge_data = G.get_edge_data("__ANY__", terminal)
                findings.append(self._make_finding(
                    f"Unguarded transition to terminal state '{terminal}' "
                    f"via function '{edge_data.get('function', '?')}'",
                    severity=0.98,
                    evidence={
                        "function": edge_data.get("function"),
                        "terminal": terminal,
                    },
                    affected_states=[terminal],
                    affected_functions=[edge_data.get("function", "")],
                    confidence=0.95,
                ))

        return findings


# FG-2: Guard Inconsistency

class FG2_GuardInconsistencyDetector(BaseDetector):

    CATEGORY = "FG-2"
    BASE_SEVERITY = 0.85

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []

        # Group edges by (source, dest)
        edge_groups: Dict[Tuple[str, str], List] = {}
        for edge in stg.edges:
            key = (edge.source_state, edge.dest_state)
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(edge)

        for (src, dst), edges in edge_groups.items():
            if len(edges) < 2:
                continue

            for i, e1 in enumerate(edges):
                for e2 in edges[i + 1 :]:
                    g1_count = len(e1.guards) + len(e1.required_roles)
                    g2_count = len(e2.guards) + len(e2.required_roles)

                    if g1_count == 0 and g2_count > 0:
                        weak_fn, strong_fn = e1.function_name, e2.function_name
                    elif g2_count == 0 and g1_count > 0:
                        weak_fn, strong_fn = e2.function_name, e1.function_name
                    elif abs(g1_count - g2_count) >= 2:
                        if g1_count < g2_count:
                            weak_fn, strong_fn = e1.function_name, e2.function_name
                        else:
                            weak_fn, strong_fn = e2.function_name, e1.function_name
                    else:
                        continue

                    # Check role inconsistency
                    r1 = set(e1.required_roles)
                    r2 = set(e2.required_roles)
                    role_diff = r2 - r1 if len(r1) < len(r2) else r1 - r2

                    findings.append(self._make_finding(
                        f"Guard inconsistency on transition {src}→{dst}: "
                        f"'{weak_fn}' has weaker guards than '{strong_fn}'",
                        evidence={
                            "weak_function": weak_fn,
                            "strong_function": strong_fn,
                            "weak_guard_count": min(g1_count, g2_count),
                            "strong_guard_count": max(g1_count, g2_count),
                            "role_difference": list(role_diff),
                        },
                        affected_states=[src, dst],
                        affected_functions=[weak_fn, strong_fn],
                    ))

        return findings


# FG-3: Missing State Reset

class FG3_MissingResetDetector(BaseDetector):

    CATEGORY = "FG-3"
    BASE_SEVERITY = 0.80

    RESET_KEYWORDS = {"balance", "amount", "deposit", "allowance",
                      "approval", "vote", "count", "total", "supply"}

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []

        for edge in stg.edges:
            if edge.dest_state not in stg.terminal_states:
                continue

            # Look for writable state that should be reset
            fn = self._find_function(stg, edge.function_name)
            if fn is None:
                continue

            # Check body for mapping/variable resets
            body = fn.body.lower() if fn.body else ""
            has_value_state = any(kw in body for kw in self.RESET_KEYWORDS)
            has_reset = ("= 0" in body or "delete " in body or
                         ".length = 0" in body)

            if has_value_state and not has_reset:
                findings.append(self._make_finding(
                    f"Transition to terminal state '{edge.dest_state}' via "
                    f"'{edge.function_name}' does not reset value-bearing state",
                    evidence={
                        "function": edge.function_name,
                        "terminal_state": edge.dest_state,
                        "value_keywords_found": [
                            kw for kw in self.RESET_KEYWORDS if kw in body
                        ],
                    },
                    affected_states=[edge.dest_state],
                    affected_functions=[edge.function_name],
                    confidence=0.7,
                ))

        return findings

    def _find_function(self, stg, fn_name):

        return None  # populated by pipeline


class FG3_MissingResetDetectorFull(FG3_MissingResetDetector):

    def __init__(self, parse_result=None):
        self.parse_result = parse_result

    def _find_function(self, stg, fn_name):
        if self.parse_result is None:
            return None
        for fn in self.parse_result.functions:
            if fn.name == fn_name:
                return fn
        return None


#  FG-4: Implicit State Dependency

class FG4_ImplicitDependencyDetector(BaseDetector):

    CATEGORY = "FG-4"
    BASE_SEVERITY = 0.75

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []

        # Collect all variables referenced in guards across edges
        guard_vars: Dict[str, List[str]] = {}  # var → list of functions guarding on it
        for edge in stg.edges:
            for g in edge.guards:
                if g.variable and not g.is_state_check:
                    if g.variable not in guard_vars:
                        guard_vars[g.variable] = []
                    guard_vars[g.variable].append(edge.function_name)


        for edge in stg.edges:
            if edge.source_state == "__ANY__":
                for g in edge.guards:
                    if g.variable and g.variable in guard_vars:
                        findings.append(self._make_finding(
                            f"Variable '{g.variable}' used in workflow guards "
                            f"is accessible from unguarded function "
                            f"'{edge.function_name}'",
                            evidence={
                                "variable": g.variable,
                                "unguarded_function": edge.function_name,
                                "guarded_by_functions": guard_vars[g.variable],
                            },
                            affected_functions=[edge.function_name]
                            + guard_vars[g.variable],
                            confidence=0.65,
                        ))

        return findings


# FG-5: Dead State (Unreachable Terminal)

class FG5_DeadStateDetector(BaseDetector):

    CATEGORY = "FG-5"
    BASE_SEVERITY = 0.60

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []
        G = stg.graph

        if not stg.terminal_states:
            return findings

        for state in stg.states:
            if state in stg.terminal_states:
                continue
            if state == stg.initial_state:
                continue
            if state not in G.nodes:
                continue

            # Check if this state can reach ANY terminal
            can_reach_terminal = False
            for terminal in stg.terminal_states:
                if terminal in G.nodes and nx.has_path(G, state, terminal):
                    can_reach_terminal = True
                    break

            if not can_reach_terminal and G.in_degree(state) > 0:
                findings.append(self._make_finding(
                    f"Dead state detected: '{state}' has incoming transitions "
                    f"but no path to any terminal state "
                    f"({stg.terminal_states}). Funds may be locked.",
                    evidence={
                        "dead_state": state,
                        "in_degree": G.in_degree(state),
                        "terminal_states": stg.terminal_states,
                    },
                    affected_states=[state],
                    confidence=0.85,
                ))

        return findings


# FG-6: Workflow Deadlock

class FG6_DeadlockDetector(BaseDetector):

    CATEGORY = "FG-6"
    BASE_SEVERITY = 0.70

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []
        G = stg.graph

        for state in stg.states:
            if state in stg.terminal_states:
                continue
            if state not in G.nodes:
                continue
            if G.out_degree(state) == 0:
                # No outgoing edges at all — already FG-5 territory
                continue

            # Check if ALL outgoing edges have guards
            outgoing_edges = [e for e in stg.edges if e.source_state == state]
            all_guarded = all(len(e.guards) > 0 for e in outgoing_edges)
            all_role_restricted = all(len(e.required_roles) > 0 for e in outgoing_edges)

            if all_guarded and len(outgoing_edges) >= 1:
                # Potential deadlock if guards could be mutually exclusive
                guard_strs = []
                for e in outgoing_edges:
                    guard_strs.extend([g.raw for g in e.guards])

                # Heuristic: check for contradictory patterns
                has_time_guard = any(
                    kw in g for g in guard_strs
                    for kw in ("block.timestamp", "block.number", "now", "deadline")
                )
                has_quorum_guard = any(
                    kw in g for g in guard_strs
                    for kw in ("count", "votes", "quorum", "threshold", "minimum")
                )

                if has_time_guard and has_quorum_guard:
                    findings.append(self._make_finding(
                        f"Potential deadlock at state '{state}': all "
                        f"{len(outgoing_edges)} outgoing transitions require "
                        f"both time and quorum conditions that may be "
                        f"simultaneously unsatisfiable",
                        evidence={
                            "state": state,
                            "outgoing_count": len(outgoing_edges),
                            "guards": guard_strs,
                            "has_time_guard": has_time_guard,
                            "has_quorum_guard": has_quorum_guard,
                        },
                        affected_states=[state],
                        confidence=0.55,
                    ))

        return findings


# FG-7: Privilege Escalation via Workflow

class FG7_PrivilegeEscalationDetector(BaseDetector):

    CATEGORY = "FG-7"
    BASE_SEVERITY = 0.90

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []
        G = stg.graph

        # Classify edges by privilege requirement
        privileged_edges = []
        unprivileged_edges = []

        for edge in stg.edges:
            if edge.required_roles:
                privileged_edges.append(edge)
            else:
                unprivileged_edges.append(edge)

        if not privileged_edges or not unprivileged_edges:
            return findings

        # States reachable only through privileged edges
        privileged_dests = {e.dest_state for e in privileged_edges}
        unprivileged_dests = {e.dest_state for e in unprivileged_edges}

        # Build unprivileged-only subgraph
        G_unpriv = nx.DiGraph()
        for edge in unprivileged_edges:
            G_unpriv.add_edge(edge.source_state, edge.dest_state)
        # Add __ANY__ → dest for unguarded edges
        for edge in unprivileged_edges:
            if edge.source_state == "__ANY__":
                for state in stg.states:
                    G_unpriv.add_edge(state, edge.dest_state)

        # Check if any privileged-only destination is reachable via
        # unprivileged edges from the initial state
        for priv_state in privileged_dests:
            if priv_state in unprivileged_dests:
                # This state has BOTH privileged and unprivileged paths
                priv_fns = [
                    e.function_name for e in privileged_edges
                    if e.dest_state == priv_state
                ]
                unpriv_fns = [
                    e.function_name for e in unprivileged_edges
                    if e.dest_state == priv_state
                ]

                findings.append(self._make_finding(
                    f"Privilege escalation: state '{priv_state}' requires "
                    f"privileged access via {priv_fns} but is also reachable "
                    f"via unprivileged {unpriv_fns}",
                    evidence={
                        "privileged_state": priv_state,
                        "privileged_functions": priv_fns,
                        "unprivileged_functions": unpriv_fns,
                    },
                    affected_states=[priv_state],
                    affected_functions=priv_fns + unpriv_fns,
                    confidence=0.85,
                ))

        return findings


#  FG-8: Temporal Ordering Violation

class FG8_TemporalOrderingDetector(BaseDetector):

    CATEGORY = "FG-8"
    BASE_SEVERITY = 0.65

    # Common ordering pairs (heuristic from naming conventions)
    EXPECTED_ORDERINGS = [
        ("deposit", "withdraw"),
        ("approve", "execute"),
        ("propose", "vote"),
        ("vote", "execute"),
        ("lock", "unlock"),
        ("start", "finish"),
        ("open", "close"),
        ("register", "claim"),
        ("stake", "unstake"),
    ]

    def detect(self, stg: STGResult) -> List[Finding]:
        findings = []
        G = stg.graph

        # Map function names to their transitions
        fn_to_edges = {}
        for edge in stg.edges:
            fn_lower = edge.function_name.lower()
            fn_to_edges[fn_lower] = edge

        for fn_a, fn_b in self.EXPECTED_ORDERINGS:
            # Find edges matching these function name patterns
            edge_a = None
            edge_b = None
            for fn_name, edge in fn_to_edges.items():
                if fn_a in fn_name and edge_a is None:
                    edge_a = edge
                if fn_b in fn_name and edge_b is None:
                    edge_b = edge

            if edge_a is None or edge_b is None:
                continue

            # Check if B's source state is reachable without passing through
            # A's destination state
            a_dest = edge_a.dest_state
            b_src = edge_b.source_state

            if b_src == "__ANY__":
                # B is callable from any state — ordering not enforced
                findings.append(self._make_finding(
                    f"Temporal ordering violation: '{edge_b.function_name}' "
                    f"is callable from any state, so the expected ordering "
                    f"'{edge_a.function_name}' → '{edge_b.function_name}' "
                    f"is not enforced",
                    evidence={
                        "expected_before": edge_a.function_name,
                        "expected_after": edge_b.function_name,
                        "reason": "B has no state guard",
                    },
                    affected_functions=[edge_a.function_name,
                                       edge_b.function_name],
                    confidence=0.70,
                ))
                continue

            # Check if b_src is reachable from initial WITHOUT going through a_dest
            if stg.initial_state and stg.initial_state in G.nodes:
                if b_src in G.nodes:
                    # Remove a_dest and check if b_src is still reachable
                    G_without_a = G.copy()
                    if a_dest in G_without_a and a_dest != stg.initial_state:
                        G_without_a.remove_node(a_dest)
                        if (b_src in G_without_a and
                                nx.has_path(G_without_a, stg.initial_state, b_src)):
                            findings.append(self._make_finding(
                                f"Temporal ordering violation: "
                                f"'{edge_b.function_name}' source state "
                                f"'{b_src}' is reachable without passing "
                                f"through '{a_dest}' "
                                f"(set by '{edge_a.function_name}')",
                                evidence={
                                    "expected_before": edge_a.function_name,
                                    "expected_after": edge_b.function_name,
                                    "bypassed_state": a_dest,
                                },
                                affected_states=[a_dest, b_src],
                                affected_functions=[
                                    edge_a.function_name,
                                    edge_b.function_name,
                                ],
                                confidence=0.60,
                            ))

        return findings


# Detector Registry

ALL_DETECTORS = {
    "FG1": FG1_BypassDetector,
    "FG2": FG2_GuardInconsistencyDetector,
    "FG3": FG3_MissingResetDetector,
    "FG4": FG4_ImplicitDependencyDetector,
    "FG5": FG5_DeadStateDetector,
    "FG6": FG6_DeadlockDetector,
    "FG7": FG7_PrivilegeEscalationDetector,
    "FG8": FG8_TemporalOrderingDetector,
}


def run_all_detectors(
    stg: STGResult,
    parse_result=None,
    enabled: List[str] = None,
) -> List[Finding]:
    """Run all (or selected) detectors on a single STG."""
    findings = []
    for key, DetectorClass in ALL_DETECTORS.items():
        if enabled and f"{key}_*" not in enabled and key not in [
            e.split("_")[0] for e in enabled
        ]:
            # Simple enable filter — matches FG1, FG2, etc.
            pass

        if key == "FG3" and parse_result is not None:
            detector = FG3_MissingResetDetectorFull(parse_result)
        else:
            detector = DetectorClass()

        try:
            results = detector.detect(stg)
            findings.extend(results)
        except Exception as e:
            logger.warning(f"Detector {key} failed: {e}")

    return findings
