"""
FlowGuard — State-Transition Graph (STG) Builder
Constructs directed graph from parsed contract, encoding states, transitions,
guards, and roles.  All graph operations use NetworkX.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.parsers.solidity_parser import (
    ParseResult, FunctionInfo, EnumDefinition, GuardPredicate
)

logger = logging.getLogger("flowguard.stg")


# Data Structures

@dataclass
class STGEdge:
    source_state: str
    dest_state: str
    function_name: str
    guards: List[GuardPredicate]
    required_roles: List[str]
    is_payable: bool = False
    has_external_call: bool = False


@dataclass
class STGResult:
    graph: nx.DiGraph
    states: List[str]
    edges: List[STGEdge]
    initial_state: Optional[str]
    terminal_states: List[str]
    wsv_name: str                  # primary workflow state variable
    enum_name: Optional[str]
    contract_name: str
    complexity: int                # number of states
    has_valid_workflow: bool        # ≥2 states and ≥1 transition


#  STG Builder

class STGBuilder:

    def build(self, parse_result: ParseResult) -> Optional[STGResult]:

        # Step 1: Identify primary WSV and its enum
        wsv, enum_def = self._find_primary_wsv(parse_result)
        if wsv is None:
            logger.debug(
                f"{parse_result.contract_name}: no enum-based WSV found"
            )
            return None

        enum_name = enum_def.name if enum_def else None
        states = enum_def.values if enum_def else []
        if len(states) < 2:
            return None

        # Step 2: Build graph
        G = nx.DiGraph()
        for s in states:
            G.add_node(s)

        edges: List[STGEdge] = []

        for fn in parse_result.functions:
            if fn.visibility in ("internal", "private"):
                continue  # only externally callable transitions matter

            src_state = self._get_source_state(fn, wsv, enum_name, states)
            dst_state = self._get_dest_state(fn, wsv, enum_name, states)

            if dst_state is None:
                continue  # function doesn't modify the WSV

            if src_state is None:
                # No guard on current state — transition from ANY state
                # This itself is a potential finding (FG-2 territory)
                src_state = "__ANY__"
                if "__ANY__" not in G:
                    G.add_node("__ANY__")

            # Collect role requirements
            roles = self._extract_roles(fn, parse_result.modifiers)

            edge = STGEdge(
                source_state=src_state,
                dest_state=dst_state,
                function_name=fn.name,
                guards=fn.guards,
                required_roles=roles,
                is_payable=fn.has_payable,
                has_external_call=fn.calls_external,
            )
            edges.append(edge)

            G.add_edge(
                src_state, dst_state,
                function=fn.name,
                guards=[g.raw for g in fn.guards],
                roles=roles,
                payable=fn.has_payable,
                external_call=fn.calls_external,
            )

        if len(edges) == 0:
            return None

        # Step 3: Identify initial and terminal states
        initial = self._infer_initial_state(states, G)
        terminals = self._infer_terminal_states(states, G)

        return STGResult(
            graph=G,
            states=states,
            edges=edges,
            initial_state=initial,
            terminal_states=terminals,
            wsv_name=wsv,
            enum_name=enum_name,
            contract_name=parse_result.contract_name,
            complexity=len(states),
            has_valid_workflow=len(states) >= 2 and len(edges) >= 1,
        )

    # Internal helpers

    def _find_primary_wsv(
        self, pr: ParseResult
    ) -> Tuple[Optional[str], Optional[EnumDefinition]]:
        enum_map = {e.name: e for e in pr.enums}

        # Priority 1: enum-typed state variables
        for sv in pr.state_variables:
            if sv.is_wsv and sv.enum_type and sv.enum_type in enum_map:
                return sv.name, enum_map[sv.enum_type]

        # Priority 2: any variable named 'state', 'status', 'phase', 'stage'
        workflow_keywords = {"state", "status", "phase", "stage", "step",
                             "currentstate", "currentstatus"}
        for sv in pr.state_variables:
            if sv.name.lower().replace("_", "") in workflow_keywords:
                if sv.enum_type and sv.enum_type in enum_map:
                    return sv.name, enum_map[sv.enum_type]

        return None, None

    def _get_source_state(
        self, fn: FunctionInfo, wsv: str, enum_name: str, states: List[str]
    ) -> Optional[str]:
        for g in fn.guards:
            if g.is_state_check and g.variable and g.operator == "==":
                value = self._normalize_state_value(g.value, enum_name, states)
                if value:
                    return value
            # Also check raw for enum patterns
            if g.raw and wsv in g.raw and "==" in g.raw:
                for s in states:
                    if s in g.raw or f"{enum_name}.{s}" in g.raw:
                        return s
        # Check state_reads dict
        if wsv in fn.state_reads:
            val = fn.state_reads[wsv]
            normalized = self._normalize_state_value(val, enum_name, states)
            if normalized:
                return normalized
        return None

    def _get_dest_state(
        self, fn: FunctionInfo, wsv: str, enum_name: str, states: List[str]
    ) -> Optional[str]:
        if wsv in fn.state_writes:
            val = fn.state_writes[wsv]
            return self._normalize_state_value(val, enum_name, states)
        return None

    def _normalize_state_value(
        self, value: str, enum_name: Optional[str], states: List[str]
    ) -> Optional[str]:
        if value is None:
            return None
        # Handle EnumName.Value format
        if "." in value:
            value = value.split(".")[-1]
        if value in states:
            return value
        # Case-insensitive fallback
        lower_map = {s.lower(): s for s in states}
        if value.lower() in lower_map:
            return lower_map[value.lower()]
        return None

    def _extract_roles(
        self, fn: FunctionInfo, modifiers: list
    ) -> List[str]:
        roles = []
        modifier_map = {m.name: m for m in modifiers}
        for mod_name in fn.modifiers:
            if mod_name in modifier_map:
                mod = modifier_map[mod_name]
                if mod.role_variable:
                    roles.append(mod.name)
            # Common role modifier names
            if any(kw in mod_name.lower() for kw in
                   ("only", "admin", "owner", "auth", "restricted")):
                roles.append(mod_name)

        # Check direct msg.sender comparisons in guards
        for g in fn.guards:
            if g.is_role_check:
                roles.append("msg.sender_check")

        return roles

    def _infer_initial_state(
        self, states: List[str], G: nx.DiGraph
    ) -> Optional[str]:
        if not states:
            return None
        # Convention: first enum value is initial
        first = states[0]
        if first in G.nodes:
            return first
        # Fallback: node with no incoming edges
        for s in states:
            if s in G.nodes and G.in_degree(s) == 0:
                return s
        return states[0]

    def _infer_terminal_states(
        self, states: List[str], G: nx.DiGraph
    ) -> List[str]:
        terminals = []
        for s in states:
            if s in G.nodes and G.out_degree(s) == 0:
                terminals.append(s)
        # Also check for semantic terminal names
        terminal_keywords = {"completed", "closed", "finished", "cancelled",
                             "terminated", "refunded", "settled", "finalized"}
        for s in states:
            if s.lower() in terminal_keywords and s not in terminals:
                terminals.append(s)
        return terminals

    # ── Utilities ──────────────────────────────────────────────

    @staticmethod
    def stg_to_dict(stg: STGResult) -> Dict:
        return {
            "contract_name": stg.contract_name,
            "wsv": stg.wsv_name,
            "enum": stg.enum_name,
            "states": stg.states,
            "initial_state": stg.initial_state,
            "terminal_states": stg.terminal_states,
            "complexity": stg.complexity,
            "edges": [
                {
                    "source": e.source_state,
                    "dest": e.dest_state,
                    "function": e.function_name,
                    "guards": [g.raw for g in e.guards],
                    "roles": e.required_roles,
                    "payable": e.is_payable,
                }
                for e in stg.edges
            ],
            "adjacency": dict(nx.to_dict_of_lists(stg.graph)),
        }

    @staticmethod
    def stg_summary(stg: STGResult) -> str:
        return (
            f"{stg.contract_name}: {len(stg.states)} states, "
            f"{len(stg.edges)} transitions, "
            f"init={stg.initial_state}, "
            f"terminals={stg.terminal_states}"
        )
