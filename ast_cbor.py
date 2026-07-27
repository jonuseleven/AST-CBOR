"""
AST-based code normalization, graph extraction, and similarity retrieval.

Core capabilities:
    1. UnifiedNormalizer — rename all user-defined identifiers to canonical forms
       (e.g., OBJ_0, VAR_1, TMP_2) to enable structural comparison across solutions
    2. ASTGraphBuilder — parse normalized code into a typed node+edge graph
    3. GraphEnhancer — enrich the graph with sibling, control, data-dependency,
       and API-semantic edges
    4. Similarity engine — multi-dimensional cosine similarity over token streams,
       AST node types, edge types, API calls, and structural patterns
    5. Top-K retrieval — search a case library for the most similar code files
"""

import ast
import os
import re
import json
import math
import hashlib
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Step 1: Unified Variable Normalization
# =============================================================================

class UnifiedNormalizer(ast.NodeTransformer):
    """Rename all user-defined identifiers to canonical prefix-index forms.

    Assignments (Assign, AnnAssign) → infer prefix from RHS value type
    Functions → FUNC_N
    For-loop targets → IDX_N
    Lambda args → ARG_N
    """

    def __init__(self):
        super().__init__()
        self.name_map: Dict[str, str] = {}
        self.counters: Dict[str, int] = defaultdict(int)

    def _alloc(self, prefix: str) -> str:
        idx = self.counters[prefix]
        self.counters[prefix] += 1
        return f"{prefix}_{idx}"

    def _set_name(self, old_name: str, prefix: str) -> str:
        if old_name not in self.name_map:
            self.name_map[old_name] = self._alloc(prefix)
        return self.name_map[old_name]

    def _infer_prefix(self, value) -> str:
        if isinstance(value, ast.Call):
            try:
                call_name = ast.unparse(value.func)
            except Exception:
                call_name = getattr(value.func, "attr", "unknown")
            low = call_name.lower()
            if ".model" in low or low == "model":
                return "OBJ"
            if ".addvar" in low or ".addvars" in low:
                return "VAR"
            return "TMP"
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return "COL"
        if isinstance(value, ast.Dict):
            return "MAP"
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return "COL"
        if isinstance(value, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            return "TMP"
        if isinstance(value, ast.Subscript):
            return "TMP"
        return "TMP"

    def _rename_target(self, target, prefix: str):
        if isinstance(target, ast.Name):
            new = self._set_name(target.id, prefix)
            return ast.copy_location(ast.Name(id=new, ctx=target.ctx), target)
        elif isinstance(target, (ast.Tuple, ast.List)):
            target.elts = [
                ast.copy_location(ast.Name(id=self._set_name(e.id, prefix), ctx=e.ctx), e)
                if isinstance(e, ast.Name) else self.visit(e)
                for e in target.elts
            ]
            return target
        return self.visit(target)

    def visit_Assign(self, node: ast.Assign):
        node.value = self.visit(node.value)
        prefix = self._infer_prefix(node.value)
        node.targets = [self._rename_target(t, prefix) for t in node.targets]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None:
            node.value = self.visit(node.value)
            prefix = self._infer_prefix(node.value)
        else:
            prefix = "TMP"
        node.target = self._rename_target(node.target, prefix)
        if node.annotation:
            node.annotation = self.visit(node.annotation)
        return node

    def visit_AugAssign(self, node: ast.AugAssign):
        node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        return node

    def visit_For(self, node: ast.For):
        node.iter = self.visit(node.iter)
        node.target = self._rename_target(node.target, "IDX")
        node.body = [self.visit(n) for n in node.body]
        node.orelse = [self.visit(n) for n in node.orelse]
        return node

    def visit_comprehension(self, node: ast.comprehension):
        node.iter = self.visit(node.iter)
        node.target = self._rename_target(node.target, "IDX")
        node.ifs = [self.visit(n) for n in node.ifs]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self._set_name(node.name, "FUNC")
        for arg in node.args.args:
            arg.arg = self._set_name(arg.arg, "ARG")
        if node.args.vararg:
            node.args.vararg.arg = self._set_name(node.args.vararg.arg, "ARG")
        if node.args.kwarg:
            node.args.kwarg.arg = self._set_name(node.args.kwarg.arg, "ARG")
        for arg in node.args.kwonlyargs:
            arg.arg = self._set_name(arg.arg, "ARG")
        node.body = [self.visit(n) for n in node.body]
        node.decorator_list = [self.visit(n) for n in node.decorator_list]
        if node.returns:
            node.returns = self.visit(node.returns)
        return node

    def visit_Lambda(self, node: ast.Lambda):
        for arg in node.args.args:
            arg.arg = self._set_name(arg.arg, "ARG")
        node.body = self.visit(node.body)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self.name_map:
            return ast.copy_location(
                ast.Name(id=self.name_map[node.id], ctx=node.ctx), node,
            )
        return node

    def normalize(self, source_code: str) -> Tuple[str, Dict[str, str]]:
        """Normalize source code. Returns (normalized_code, name_map)."""
        tree = ast.parse(source_code)
        new_tree = self.visit(tree)
        ast.fix_missing_locations(new_tree)
        try:
            normalized_code = ast.unparse(new_tree)
        except Exception:
            normalized_code = source_code
        return normalized_code, dict(self.name_map)


# =============================================================================
# Step 2: AST Graph Builder
# =============================================================================

class ASTGraphBuilder(ast.NodeVisitor):
    """Walk an AST and build a node + edge graph."""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.nodes: List[dict] = []
        self.edges: List[dict] = []
        self.node_id = 0
        self.parent_stack: List[int] = []
        self.ast_node_to_id: Dict[int, int] = {}

    def _new_node(self, node, node_type: str, extra: dict = None) -> int:
        nid = self.node_id
        self.node_id += 1
        code_snippet = ast.get_source_segment(self.source_code, node)
        item = {
            "id": nid,
            "type": node_type,
            "lineno": getattr(node, "lineno", None),
            "end_lineno": getattr(node, "end_lineno", None),
            "col_offset": getattr(node, "col_offset", None),
            "end_col_offset": getattr(node, "end_col_offset", None),
            "code": code_snippet,
        }
        if extra:
            item.update(extra)
        self.nodes.append(item)
        self.ast_node_to_id[id(node)] = nid
        return nid

    def generic_visit(self, node):
        node_type = type(node).__name__
        extra = {}
        if isinstance(node, ast.Name):
            extra["name"] = node.id
        elif isinstance(node, ast.Attribute):
            extra["attr"] = node.attr
        elif isinstance(node, ast.Call):
            extra["call_name"] = self._get_call_name(node)
        elif isinstance(node, ast.Constant):
            extra["value"] = repr(node.value)

        current_id = self._new_node(node, node_type, extra)
        if self.parent_stack:
            self._add_edge(self.parent_stack[-1], current_id, "ast")
        self.parent_stack.append(current_id)
        super().generic_visit(node)
        self.parent_stack.pop()

    def _add_edge(self, src: int, dst: int, etype: str):
        self.edges.append({"source": src, "target": dst, "type": etype})

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Attribute):
            try:
                return f"{ast.unparse(func.value)}.{func.attr}"
            except Exception:
                return func.attr
        elif isinstance(func, ast.Name):
            return func.id
        return "unknown_call"

    def build(self, tree: ast.AST) -> Tuple[List[dict], List[dict]]:
        self.visit(tree)
        return self.nodes, self.edges


# =============================================================================
# Step 3: Graph Enhancement
# =============================================================================

class GraphEnhancer:
    """Add semantic edges to the AST graph."""

    def __init__(self, source_code: str, nodes: List[dict],
                 edges: List[dict], tree: ast.AST):
        self.source_code = source_code
        self.nodes = nodes
        self.edges = list(edges)  # copy
        self.enhanced_edges = list(edges)
        self.tree = tree
        self.children_map: Dict[int, List[int]] = defaultdict(list)
        for e in edges:
            if e["type"] == "ast":
                self.children_map[e["source"]].append(e["target"])
        self.node_lookup = {n["id"]: n for n in nodes}
        self.ast_node_to_id: Dict[tuple, int] = {}
        for n in nodes:
            key = (n.get("lineno"), n.get("col_offset"), n.get("type"), n.get("code"))
            self.ast_node_to_id[key] = n["id"]

    def enhance(self) -> List[dict]:
        self._add_sibling_edges()
        defs, uses, api_calls = self._collect_def_use_api(self.tree)
        self._add_data_dependency_edges(defs, uses)
        self._add_control_edges(self.tree)
        self._add_api_semantic_edges(api_calls)
        return self.enhanced_edges

    def _add_sibling_edges(self):
        for parent, children in self.children_map.items():
            sorted_children = sorted(
                children,
                key=lambda cid: (
                    self.node_lookup[cid].get("lineno") or -1,
                    self.node_lookup[cid].get("col_offset") or -1,
                ),
            )
            for i in range(len(sorted_children) - 1):
                self.enhanced_edges.append({
                    "source": sorted_children[i],
                    "target": sorted_children[i + 1],
                    "type": "sibling",
                })

    def _collect_def_use_api(self, node):
        """Collect variable definitions, uses, and API call nodes."""
        defs = defaultdict(list)
        uses = defaultdict(list)
        api_calls = defaultdict(list)
        current_id = self._find_node_id(node)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in self._extract_names(target):
                    if current_id is not None:
                        defs[name].append(current_id)
        elif isinstance(node, ast.AnnAssign):
            for name in self._extract_names(node.target):
                if current_id is not None:
                    defs[name].append(current_id)
        elif isinstance(node, ast.For):
            for name in self._extract_names(node.target):
                if current_id is not None:
                    defs[name].append(current_id)
        elif isinstance(node, ast.comprehension):
            for name in self._extract_names(node.target):
                if current_id is not None:
                    defs[name].append(current_id)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and current_id is not None:
                uses[node.id].append(current_id)
        elif isinstance(node, ast.Call):
            call_name = ASTGraphBuilder._get_call_name(node)
            if current_id is not None:
                api_calls[call_name].append(current_id)

        for child in ast.iter_child_nodes(node):
            cd, cu, ca = self._collect_def_use_api(child)
            for k, v in cd.items():
                defs[k].extend(v)
            for k, v in cu.items():
                uses[k].extend(v)
            for k, v in ca.items():
                api_calls[k].extend(v)

        return defs, uses, api_calls

    def _add_data_dependency_edges(self, defs, uses):
        for name, use_ids in uses.items():
            def_ids = defs.get(name, [])
            if not def_ids:
                continue
            for uid in use_ids:
                u_lineno = self.node_lookup[uid].get("lineno")
                if u_lineno is None:
                    continue
                candidates = [
                    did for did in def_ids
                    if self.node_lookup[did].get("lineno") is not None
                    and self.node_lookup[did].get("lineno") <= u_lineno
                ]
                if candidates:
                    nearest = max(candidates, key=lambda did: self.node_lookup[did].get("lineno") or 0)
                    self.enhanced_edges.append({
                        "source": nearest, "target": uid,
                        "type": "data_dependency", "var": name,
                    })

    def _add_control_edges(self, node, controller=None):
        node_id = self._find_node_id(node)
        new_controller = controller
        if isinstance(node, (ast.If, ast.For, ast.While)):
            new_controller = node_id
        if controller is not None and node_id is not None and node_id != controller:
            self.enhanced_edges.append({
                "source": controller, "target": node_id, "type": "control",
            })
        for child in ast.iter_child_nodes(node):
            self._add_control_edges(child, new_controller)

    def _add_api_semantic_edges(self, api_calls):
        for call_name, ids in api_calls.items():
            uniq = sorted(set(ids))
            if len(uniq) > 1:
                for i in range(len(uniq) - 1):
                    self.enhanced_edges.append({
                        "source": uniq[i], "target": uniq[i + 1],
                        "type": "api_semantic", "api": call_name,
                    })

    @staticmethod
    def _extract_names(node) -> List[str]:
        names = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
        return names

    def _find_node_id(self, node) -> Optional[int]:
        key = (
            getattr(node, "lineno", None),
            getattr(node, "col_offset", None),
            type(node).__name__,
            ast.get_source_segment(self.source_code, node),
        )
        return self.ast_node_to_id.get(key)


# =============================================================================
# Pipeline entry point
# =============================================================================

def build_output_paths(input_path: str) -> Tuple[str, str]:
    """Return (normalized_path, ast_json_path) for an input file."""
    directory = os.path.dirname(os.path.abspath(input_path))
    stem, _ext = os.path.splitext(os.path.basename(input_path))
    normalized_path = os.path.join(directory, f"{stem}.normalized.py")
    ast_json_path = os.path.join(directory, f"{stem}.ast.json")
    return normalized_path, ast_json_path


def process_single_file(input_file: str) -> Tuple[str, str]:
    """Run the full AST pipeline on a single Python file.

    Returns (normalized_path, ast_json_path).
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    normalized_path, ast_json_path = build_output_paths(input_file)

    with open(input_file, "r", encoding="utf-8") as f:
        source_code = f.read()

    # Normalize
    normalizer = UnifiedNormalizer()
    normalized_code, name_map = normalizer.normalize(source_code)

    with open(normalized_path, "w", encoding="utf-8") as f:
        f.write(normalized_code)

    # Build AST graph
    tree = ast.parse(normalized_code)
    builder = ASTGraphBuilder(normalized_code)
    nodes, ast_edges = builder.build(tree)

    # Enhance
    enhancer = GraphEnhancer(normalized_code, nodes, ast_edges, tree)
    enhanced_edges = enhancer.enhance()

    # Save
    result = {
        "input_file": os.path.abspath(input_file),
        "normalized_file": normalized_path,
        "normalized_code": normalized_code,
        "name_map": name_map,
        "nodes": nodes,
        "edges": enhanced_edges,
        "summary": {
            "num_nodes": len(nodes),
            "num_edges": len(enhanced_edges),
            "num_ast_edges": sum(1 for e in enhanced_edges if e["type"] == "ast"),
            "num_sibling_edges": sum(1 for e in enhanced_edges if e["type"] == "sibling"),
            "num_control_edges": sum(1 for e in enhanced_edges if e["type"] == "control"),
            "num_data_dependency_edges": sum(1 for e in enhanced_edges if e["type"] == "data_dependency"),
            "num_api_semantic_edges": sum(1 for e in enhanced_edges if e["type"] == "api_semantic"),
        },
    }
    with open(ast_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("AST pipeline complete: %s -> %s", input_file, ast_json_path)
    return normalized_path, ast_json_path


def batch_process(dataset_dir: str, filename: str = "gurobi_solver.py") -> Dict[str, int]:
    """Process all files matching `filename` under `dataset_dir`.

    Returns dict with ok, failed, total counts.
    """
    solver_files = []
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        if filename in files:
            solver_files.append(os.path.join(root, filename))

    solver_files.sort()
    ok, failed = 0, 0
    for path in solver_files:
        try:
            process_single_file(path)
            ok += 1
        except Exception as exc:
            failed += 1
            logger.error("[ERR] %s: %s", path, exc)

    logger.info(
        "Batch AST done: dataset=%s files=%d ok=%d failed=%d",
        dataset_dir, len(solver_files), ok, failed,
    )
    return {"ok": ok, "failed": failed, "total": len(solver_files)}


# =============================================================================
# Similarity & Retrieval
# =============================================================================

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|>=|<=|!=|[-+*/%=<>()\[\]{},.:]")


def tokenize_code(code: str) -> List[str]:
    """Tokenize Python code into a list of tokens."""
    return TOKEN_PATTERN.findall(code)


def cosine_similarity(c1: Counter, c2: Counter) -> float:
    """Cosine similarity between two Counters."""
    if not c1 or not c2:
        return 0.0
    common = set(c1.keys()) & set(c2.keys())
    dot = sum(c1[k] * c2[k] for k in common)
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def extract_features(normalized_code: str, ast_data: dict) -> Dict[str, Counter]:
    """Extract multi-dimensional feature vectors from code + AST."""
    tokens = tokenize_code(normalized_code)
    token_counter = Counter(tokens)

    node_type_counter = Counter()
    edge_type_counter = Counter()
    for node in ast_data.get("nodes", []):
        ntype = node.get("type")
        if ntype:
            node_type_counter[ntype] += 1
    for edge in ast_data.get("edges", []):
        etype = edge.get("type")
        if etype:
            edge_type_counter[etype] += 1

    api_counter = Counter()
    for node in ast_data.get("nodes", []):
        if node.get("type") == "Call":
            name = node.get("call_name", "")
            if name:
                api_counter[name.split(".")[-1]] += 1

    pattern_counter = Counter()
    for api in ("Model", "addVar", "addVars", "addConstr", "addConstrs",
                "setObjective", "optimize", "getVars"):
        pattern_counter[f"api::{api}"] = api_counter.get(api, 0)
    if "OPTIMAL" in normalized_code:
        pattern_counter["flag::OPTIMAL"] = 1
    if "ObjVal" in normalized_code:
        pattern_counter["flag::ObjVal"] = 1
    if "MINIMIZE" in normalized_code:
        pattern_counter["flag::MINIMIZE"] = 1
    if "MAXIMIZE" in normalized_code:
        pattern_counter["flag::MAXIMIZE"] = 1

    return {
        "token": token_counter,
        "node_type": node_type_counter,
        "edge_type": edge_type_counter,
        "api": api_counter,
        "pattern": pattern_counter,
    }


def compute_similarity(
    query_feat: Dict[str, Counter],
    cand_feat: Dict[str, Counter],
    w_token: float = 0.35,
    w_node: float = 0.20,
    w_edge: float = 0.15,
    w_api: float = 0.20,
    w_pattern: float = 0.10,
) -> Tuple[float, Dict[str, float]]:
    """Compute weighted multi-dimensional similarity."""
    scores = {
        "token": cosine_similarity(query_feat["token"], cand_feat["token"]),
        "node_type": cosine_similarity(query_feat["node_type"], cand_feat["node_type"]),
        "edge_type": cosine_similarity(query_feat["edge_type"], cand_feat["edge_type"]),
        "api": cosine_similarity(query_feat["api"], cand_feat["api"]),
        "pattern": cosine_similarity(query_feat["pattern"], cand_feat["pattern"]),
    }
    final = (
        w_token * scores["token"]
        + w_node * scores["node_type"]
        + w_edge * scores["edge_type"]
        + w_api * scores["api"]
        + w_pattern * scores["pattern"]
    )
    return final, scores


def retrieve_topk(
    root_dir: str,
    case_ids: List[int],
    query_code_file: str,
    topk: int = 5,
) -> Dict[str, Any]:
    """Retrieve the top-K most similar cases from a case library.

    Args:
        root_dir: Root directory containing case subdirectories.
        case_ids: List of case IDs to search over.
        query_code_file: Path to the query .py file.
        topk: Number of results to return.

    Returns:
        Dict with query info and ranked results list.
    """
    # Always refresh the query so stale AST artifacts cannot affect retrieval.
    normalized_path, ast_json_path = process_single_file(query_code_file)

    # Load query features
    with open(ast_json_path, "r", encoding="utf-8") as f:
        query_ast = json.load(f)
    with open(normalized_path, "r", encoding="utf-8") as f:
        query_code = f.read()
    query_feat = extract_features(query_code, query_ast)

    # Collect candidate cases
    candidates = []
    query_abs = os.path.abspath(query_code_file)
    seen_hashes = set()

    for cid in case_ids:
        case_dir = os.path.join(root_dir, str(cid))
        if not os.path.isdir(case_dir):
            continue
        raw_py = os.path.join(case_dir, "gurobi_solver.py")
        if not os.path.isfile(raw_py):
            raw_candidates = sorted(
                os.path.join(case_dir, fname)
                for fname in os.listdir(case_dir)
                if fname.endswith(".py")
                and not fname.endswith(".normalized.py")
                and not fname.endswith(".initial.py")
            )
            raw_py = raw_candidates[0] if raw_candidates else None
        if raw_py is None:
            continue
        if os.path.abspath(raw_py) == query_abs:
            continue

        normalized_py, ast_json = build_output_paths(raw_py)
        raw_mtime = os.path.getmtime(raw_py)
        artifacts_stale = (
            not os.path.isfile(normalized_py)
            or not os.path.isfile(ast_json)
            or os.path.getmtime(normalized_py) < raw_mtime
            or os.path.getmtime(ast_json) < raw_mtime
        )
        if artifacts_stale:
            try:
                normalized_py, ast_json = process_single_file(raw_py)
            except (OSError, SyntaxError, ValueError) as exc:
                logger.warning("Skipping invalid case %s: %s", cid, exc)
                continue

        try:
            with open(ast_json, "r", encoding="utf-8") as f:
                cand_ast = json.load(f)
            with open(normalized_py, "r", encoding="utf-8") as f:
                cand_code = f.read()
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable case %s: %s", cid, exc)
            continue

        code_hash = hashlib.sha256(cand_code.encode()).hexdigest()
        if code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)

        cand_feat = extract_features(cand_code, cand_ast)
        sim, detail = compute_similarity(query_feat, cand_feat)

        candidates.append({
            "case_id": str(cid),
            "case_dir": case_dir,
            "raw_py": raw_py,
            "normalized_py": normalized_py,
            "ast_json": ast_json,
            "similarity": sim,
            "detail_scores": detail,
            "summary": cand_ast.get("summary", {}),
        })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "query_code_file": os.path.abspath(query_code_file),
        "query_normalized_file": normalized_path,
        "query_ast_json": ast_json_path,
        "searched_case_ids": [str(x) for x in case_ids],
        "topk": topk,
        "results": candidates[:topk],
    }


def discover_case_ids(root_dir: str, valid_only: bool = False) -> List[int]:
    """Return numeric case IDs containing a generated solver file."""
    if not os.path.isdir(root_dir):
        return []
    case_ids = []
    for name in os.listdir(root_dir):
        case_dir = os.path.join(root_dir, name)
        if not name.isdigit() or not os.path.isfile(os.path.join(case_dir, "gurobi_solver.py")):
            continue
        if valid_only:
            valid = False
            result_path = os.path.join(case_dir, "solver_result.json")
            if os.path.isfile(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        valid = json.load(f).get("solver_status") == "OPTIMAL"
                except (OSError, json.JSONDecodeError):
                    valid = False
            else:
                objective_path = os.path.join(case_dir, "objective_value.txt")
                if os.path.isfile(objective_path):
                    try:
                        with open(objective_path, "r", encoding="utf-8") as f:
                            valid = math.isfinite(float(f.read().strip()))
                    except (OSError, ValueError):
                        valid = False
            if not valid:
                continue
        case_ids.append(int(name))
    return sorted(case_ids)


def parse_case_ids(spec: str) -> List[int]:
    """Parse a case ID specification string.

    Supports: "1,2,3"  "[1,2,3]"  "1 2 3"
    """
    s = spec.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    parts = re.split(r"[,\s]+", s)
    parts = [p.strip() for p in parts if p.strip()]
    result = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid case ID: '{p}'")
        result.append(int(p))
    return result
