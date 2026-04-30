"""
FlowGuard — Solidity Contract Parser
Primary: Slither-based AST analysis (if available)
Fallback: Regex-based extraction for state vars, functions, guards, modifiers.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import concurrent.futures


logger = logging.getLogger("flowguard.parser")


#  Data Structures

@dataclass
class EnumDefinition:
    """A Solidity enum declaration."""
    name: str
    values: List[str]
    line_number: int = 0


@dataclass
class StateVariable:
    name: str
    var_type: str                  # "enum", "uint256", "address", "bool", etc.
    enum_type: Optional[str] = None  # if var_type == "enum", which enum
    visibility: str = "internal"
    is_wsv: bool = False           # Workflow State Variable
    is_acv: bool = False           # Access Control Variable


@dataclass
class GuardPredicate:
    raw: str                       # full require() string
    variable: Optional[str] = None
    operator: Optional[str] = None # "==", "!=", ">=", etc.
    value: Optional[str] = None
    is_state_check: bool = False   # references a WSV
    is_role_check: bool = False    # references msg.sender / role var


@dataclass
class Modifier:
    name: str
    body: str
    guards: List[GuardPredicate] = field(default_factory=list)
    role_variable: Optional[str] = None


@dataclass
class FunctionInfo:
    name: str
    visibility: str                # "public", "external", "internal", "private"
    modifiers: List[str] = field(default_factory=list)
    guards: List[GuardPredicate] = field(default_factory=list)
    state_reads: Dict[str, str] = field(default_factory=dict)   # var → value read
    state_writes: Dict[str, str] = field(default_factory=dict)  # var → value written
    parameters: List[str] = field(default_factory=list)
    body: str = ""
    line_number: int = 0
    has_payable: bool = False
    has_delegatecall: bool = False
    calls_external: bool = False


@dataclass
class ParseResult:
    contract_name: str
    enums: List[EnumDefinition]
    state_variables: List[StateVariable]
    functions: List[FunctionInfo]
    modifiers: List[Modifier]
    wsv_names: Set[str]             # Workflow State Variable names
    acv_names: Set[str]             # Access Control Variable names
    raw_source: str = ""
    parse_method: str = "regex"     # "regex" or "slither"


# Regex Patterns

class SolidityPatterns:

    # Contract / interface declaration
    CONTRACT = re.compile(
        r"(?:contract|library|interface)\s+(\w+)(?:\s+is\s+[^{]+)?\s*\{",
        re.MULTILINE
    )

    # Enum declaration
    ENUM = re.compile(
        r"enum\s+(\w+)\s*\{([^}]+)\}",
        re.MULTILINE
    )

    # State variable declaration (simplified but effective)
    STATE_VAR = re.compile(
        r"^\s*(\w+(?:\s*\[\s*\w*\s*\])?)\s+"
        r"(public|private|internal|external)?\s*"
        r"(\w+)\s*(?:=\s*[^;]+)?;",
        re.MULTILINE
    )

    # Function declaration (captures name, visibility, modifiers, body)
    FUNCTION = re.compile(
        r"function\s+(\w+)\s*\(([^)]*)\)\s+"
        r"((?:public|external|internal|private|view|pure|payable|virtual|override"
        r"|returns\s*\([^)]*\)|\w+\s*(?:\([^)]*\))?[\s,]*)+)\s*\{",
        re.MULTILINE | re.DOTALL
    )

    # Modifier declaration
    MODIFIER = re.compile(
        r"modifier\s+(\w+)\s*(?:\([^)]*\))?\s*\{([\s\S]*?_;\s*\})",
        re.MULTILINE
    )

    # require() calls
    REQUIRE = re.compile(
        r"require\s*\(\s*(.*?)(?:\s*,\s*\"[^\"]*\")?\s*\)\s*;",
        re.MULTILINE | re.DOTALL
    )

    # State variable assignment: varName = value;
    STATE_ASSIGN = re.compile(
        r"(\w+)\s*=\s*(?:(\w+)\.)?(\w+)\s*;",
        re.MULTILINE
    )

    # msg.sender comparison
    MSG_SENDER_CHECK = re.compile(
        r"(msg\.sender\s*==\s*(\w+)|(\w+)\s*==\s*msg\.sender)",
        re.MULTILINE
    )

    # Enum-typed variable declaration
    ENUM_VAR = re.compile(
        r"(\w+)\s+(?:public\s+|private\s+|internal\s+)?(\w+)\s*(?:=\s*\w+\.\w+)?\s*;",
        re.MULTILINE
    )

    # delegatecall
    DELEGATECALL = re.compile(r"\.delegatecall\s*\(", re.MULTILINE)

    # External call patterns
    EXTERNAL_CALL = re.compile(
        r"(?:\.call\s*\{|\.transfer\s*\(|\.send\s*\()",
        re.MULTILINE
    )


# Regex-Based Parser

class RegexParser:

    def parse(self, source: str) -> ParseResult:
        """Main entry: parse a Solidity source file."""
        contract_name = self._extract_contract_name(source)
        enums = self._extract_enums(source)
        modifiers = self._extract_modifiers(source)
        state_vars = self._extract_state_variables(source, enums)
        functions = self._extract_functions(source, enums, state_vars, modifiers)

        # Classify WSV and ACV
        wsv_names = set()
        acv_names = set()
        for sv in state_vars:
            if sv.is_wsv:
                wsv_names.add(sv.name)
            if sv.is_acv:
                acv_names.add(sv.name)

        return ParseResult(
            contract_name=contract_name,
            enums=enums,
            state_variables=state_vars,
            functions=functions,
            modifiers=modifiers,
            wsv_names=wsv_names,
            acv_names=acv_names,
            raw_source=source,
            parse_method="regex",
        )

    def _extract_contract_name(self, source: str) -> str:
        m = SolidityPatterns.CONTRACT.search(source)
        return m.group(1) if m else "UnknownContract"

    def _extract_enums(self, source: str) -> List[EnumDefinition]:
        enums = []
        for m in SolidityPatterns.ENUM.finditer(source):
            name = m.group(1)
            values = [v.strip() for v in m.group(2).split(",") if v.strip()]
            line_no = source[:m.start()].count("\n") + 1
            enums.append(EnumDefinition(name=name, values=values, line_number=line_no))
        return enums

    def _extract_modifiers(self, source: str) -> List[Modifier]:
        modifiers = []
        for m in SolidityPatterns.MODIFIER.finditer(source):
            name = m.group(1)
            body = m.group(2)
            guards = self._extract_guards(body)

            # Check if this modifier is role-based
            role_var = None
            sender_match = SolidityPatterns.MSG_SENDER_CHECK.search(body)
            if sender_match:
                role_var = sender_match.group(2) or sender_match.group(3)
                for g in guards:
                    g.is_role_check = True

            modifiers.append(Modifier(
                name=name, body=body, guards=guards, role_variable=role_var
            ))
        return modifiers

    def _extract_state_variables(
        self, source: str, enums: List[EnumDefinition]
    ) -> List[StateVariable]:
        state_vars = []
        enum_names = {e.name for e in enums}

        # Find enum-typed state variables
        for m in SolidityPatterns.ENUM_VAR.finditer(source):
            type_name = m.group(1)
            var_name = m.group(2)
            if type_name in enum_names:
                sv = StateVariable(
                    name=var_name,
                    var_type="enum",
                    enum_type=type_name,
                    is_wsv=True,  # enum vars are WSVs by default
                )
                state_vars.append(sv)

        # Find address-typed variables used in access control
        addr_pattern = re.compile(
            r"address\s+(?:public\s+|private\s+|internal\s+)?(\w+)",
            re.MULTILINE
        )
        for m in addr_pattern.finditer(source):
            var_name = m.group(1)
            # Check if used in msg.sender comparison
            check_pattern = re.compile(
                rf"(?:msg\.sender\s*==\s*{var_name}|{var_name}\s*==\s*msg\.sender)"
            )
            is_acv = bool(check_pattern.search(source))
            sv = StateVariable(
                name=var_name,
                var_type="address",
                is_acv=is_acv,
            )
            state_vars.append(sv)

        # Find uint/bool variables used in require guards AND assigned in functions
        guard_vars = set()
        for m in SolidityPatterns.REQUIRE.finditer(source):
            cond = m.group(1)
            # Extract variable names from condition
            for tok in re.findall(r"\b(\w+)\b", cond):
                if tok not in ("require", "msg", "sender", "true", "false",
                               "block", "timestamp", "now", "this"):
                    guard_vars.add(tok)

        assign_vars = set()
        for m in SolidityPatterns.STATE_ASSIGN.finditer(source):
            assign_vars.add(m.group(1))

        # Variables that appear in BOTH guards and assignments → potential WSV
        potential_wsv = guard_vars & assign_vars
        existing_names = {sv.name for sv in state_vars}

        for var_name in potential_wsv:
            if var_name not in existing_names:
                # Try to determine type from declaration
                type_match = re.search(
                    rf"(uint\d*|bool|bytes\d*|int\d*)\s+(?:public\s+|private\s+|internal\s+)?{var_name}\b",
                    source
                )
                if type_match:
                    sv = StateVariable(
                        name=var_name,
                        var_type=type_match.group(1),
                        is_wsv=True,
                    )
                    state_vars.append(sv)

        return state_vars

    def _extract_functions(
        self,
        source: str,
        enums: List[EnumDefinition],
        state_vars: List[StateVariable],
        modifiers: List[Modifier],
    ) -> List[FunctionInfo]:
        functions = []
        wsv_names = {sv.name for sv in state_vars if sv.is_wsv}
        enum_values = {}
        for e in enums:
            for v in e.values:
                enum_values[v] = e.name
                enum_values[f"{e.name}.{v}"] = e.name

        for m in SolidityPatterns.FUNCTION.finditer(source):
            fn_name = m.group(1)
            params = m.group(2).strip()
            qualifiers = m.group(3).strip()

            # Extract function body (brace-matched)
            body_start = m.end() - 1  # position of opening brace
            body = self._extract_brace_block(source, body_start)

            # Parse visibility
            visibility = "public"
            for vis in ("public", "external", "internal", "private"):
                if vis in qualifiers:
                    visibility = vis
                    break

            # Parse modifier names from qualifiers
            modifier_names = []
            mod_name_set = {mod.name for mod in modifiers}
            for tok in re.findall(r"\b(\w+)\b", qualifiers):
                if tok in mod_name_set:
                    modifier_names.append(tok)

            # Extract guards from function body
            guards = self._extract_guards(body)

            # Classify guard predicates
            for g in guards:
                if g.variable and g.variable in wsv_names:
                    g.is_state_check = True
                if "msg.sender" in g.raw:
                    g.is_role_check = True

            # Extract state reads (from require conditions)
            state_reads = {}
            for g in guards:
                if g.is_state_check and g.variable and g.value:
                    state_reads[g.variable] = g.value

            # Extract state writes (assignments to WSVs)
            state_writes = {}
            for assign_m in SolidityPatterns.STATE_ASSIGN.finditer(body):
                var = assign_m.group(1)
                if var in wsv_names:
                    # Value might be EnumType.Value or just Value
                    enum_prefix = assign_m.group(2)
                    value = assign_m.group(3)
                    if enum_prefix:
                        state_writes[var] = f"{enum_prefix}.{value}"
                    else:
                        state_writes[var] = value

            fn_info = FunctionInfo(
                name=fn_name,
                visibility=visibility,
                modifiers=modifier_names,
                guards=guards,
                state_reads=state_reads,
                state_writes=state_writes,
                parameters=[p.strip() for p in params.split(",") if p.strip()],
                body=body,
                line_number=source[:m.start()].count("\n") + 1,
                has_payable="payable" in qualifiers,
                has_delegatecall=bool(SolidityPatterns.DELEGATECALL.search(body)),
                calls_external=bool(SolidityPatterns.EXTERNAL_CALL.search(body)),
            )
            functions.append(fn_info)

        return functions

    def _extract_guards(self, body: str) -> List[GuardPredicate]:
        guards = []
        for m in SolidityPatterns.REQUIRE.finditer(body):
            raw = m.group(1).strip()
            variable, operator, value = None, None, None

            # Try to decompose: variable op value
            cmp_match = re.match(
                r"(\w+(?:\.\w+)?)\s*(==|!=|>=|<=|>|<)\s*(\w+(?:\.\w+)?)",
                raw
            )
            if cmp_match:
                variable = cmp_match.group(1)
                operator = cmp_match.group(2)
                value = cmp_match.group(3)

            guards.append(GuardPredicate(
                raw=raw, variable=variable, operator=operator, value=value
            ))
        return guards

    def _extract_brace_block(self, source: str, start: int) -> str:
        """Extract text between matching braces starting at position start."""
        if start >= len(source) or source[start] != "{":
            return ""
        depth = 0
        i = start
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return source[start + 1 : i]
            i += 1
        return source[start + 1 :]  # unmatched — return rest


# Slither Integration (optional)

class SlitherParser:

    def __init__(self):
        self.available = False
        try:
            from slither.slither import Slither
            self.Slither = Slither
            self.available = True
            logger.info("Slither available — using as primary parser")
        except ImportError:
            logger.info("Slither not installed — regex parser will be used")

    def parse(self, source: str, filepath: str = None) -> Optional[ParseResult]:
        """Attempt Slither-based parse. Returns None on failure."""
        if not self.available or filepath is None:
            return None

        try:
            slither = self.Slither(filepath)
        except Exception as e:
            logger.debug(f"Slither failed on {filepath}: {e}")
            return None

        enums = []
        state_vars = []
        functions = []
        modifiers_list = []
        wsv_names = set()
        acv_names = set()

        for contract in slither.contracts_derived:
            # Extract enums
            for enum in contract.enums:
                ed = EnumDefinition(
                    name=enum.name,
                    values=[str(v) for v in enum.values],
                )
                enums.append(ed)

            enum_names = {e.name for e in enums}

            # Extract state variables
            for sv in contract.state_variables:
                is_enum = str(sv.type) in enum_names
                is_acv = str(sv.type) == "address"
                state_var = StateVariable(
                    name=sv.name,
                    var_type="enum" if is_enum else str(sv.type),
                    enum_type=str(sv.type) if is_enum else None,
                    visibility=str(sv.visibility),
                    is_wsv=is_enum,
                    is_acv=is_acv,
                )
                state_vars.append(state_var)
                if is_enum:
                    wsv_names.add(sv.name)
                if is_acv:
                    acv_names.add(sv.name)

            # Extract functions
            for fn in contract.functions:
                if fn.is_constructor:
                    continue
                guards = []
                state_reads = {}
                state_writes = {}

                # Extract require conditions from Slither's nodes
                for node in fn.nodes:
                    for ir in node.irs:
                        ir_str = str(ir)
                        if "SOLIDITY_CALL" in ir_str and "require" in ir_str:
                            guards.append(GuardPredicate(raw=ir_str))

                # State variable reads and writes
                for sv_read in fn.state_variables_read:
                    if sv_read.name in wsv_names:
                        state_reads[sv_read.name] = "read"

                for sv_write in fn.state_variables_written:
                    if sv_write.name in wsv_names:
                        state_writes[sv_write.name] = "written"

                fn_info = FunctionInfo(
                    name=fn.name,
                    visibility=str(fn.visibility),
                    modifiers=[str(m) for m in fn.modifiers],
                    guards=guards,
                    state_reads=state_reads,
                    state_writes=state_writes,
                    body="",  # not needed when Slither provides IR
                )
                functions.append(fn_info)

            # Extract modifiers
            for mod in contract.modifiers:
                mod_guards = []
                role_var = None
                for node in mod.nodes:
                    for ir in node.irs:
                        ir_str = str(ir)
                        if "require" in ir_str.lower():
                            mod_guards.append(GuardPredicate(raw=ir_str))
                        if "msg.sender" in ir_str:
                            role_var = mod.name
                modifiers_list.append(Modifier(
                    name=mod.name, body="", guards=mod_guards, role_variable=role_var
                ))

            break  # analyze first derived contract only

        return ParseResult(
            contract_name=contract.name if slither.contracts_derived else "Unknown",
            enums=enums,
            state_variables=state_vars,
            functions=functions,
            modifiers=modifiers_list,
            wsv_names=wsv_names,
            acv_names=acv_names,
            raw_source=source,
            parse_method="slither",
        )


# Unified Parser Entry Point

class ContractParser:
    def __init__(self, parse_timeout: int = 5):
        self.regex_parser = RegexParser()
        self.parse_timeout = parse_timeout

    def parse(self, source: str, filepath: str = None) -> ParseResult:
        return self.regex_parser.parse(source)

    def parse_batch(self, records, max_workers: int = 1):
        results = {}
        for rec in records:
            try:
                results[rec.contract_id] = self.parse(rec.source_code, rec.file_path)
            except Exception:
                pass
        return results
