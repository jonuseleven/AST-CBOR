"""
Gurobi solver code generation from extracted problem representations.

Three-step pipeline:
    1. JSON → Mathematical Model   (natural language math formulation)
    2. Math Model → Core Code       (Gurobi Python variable/constraint definitions)
    3. Core Code → Full Solver      (complete runnable script with imports and output)

The generated code is executed and objective values are saved.
"""

import os
import sys
import json
import subprocess
import threading
import logging
import re
import ast
import math
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import (
    GUROBI_MATH_MODEL_PROMPT,
    GUROBI_CORE_PROMPT,
    GUROBI_FULL_PROMPT,
    GUROBI_CBR_REFINE_PROMPT,
)
from llm import DEFAULT_MODEL, LLMClient

logger = logging.getLogger(__name__)

RESULT_LOCK = threading.Lock()
DEFAULT_MAX_PARALLEL = 30


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def build_math_model_prompt(detailed_result: dict) -> str:
    """Build the math model prompt from a detailed_result JSON."""
    # Format detailed attributes for the prompt
    detailed_attrs = detailed_result.get("detailed_attributes", {})
    formatted_attrs = ""
    if isinstance(detailed_attrs, dict):
        for entity, attrs in detailed_attrs.items():
            formatted_attrs += f"\n{entity}:\n"
            if isinstance(attrs, list):
                for attr in attrs:
                    if isinstance(attr, dict):
                        name = attr.get("name", "Unknown")
                        explanation = attr.get("explanation", "")
                        value_range = attr.get("value_range", "Unknown")
                        formatted_attrs += (
                            f"  - {name}: {explanation} "
                            f"(value: {value_range})\n"
                        )
                    elif isinstance(attr, str):
                        formatted_attrs += f"  - {attr}\n"
            elif isinstance(attrs, dict):
                for attr_name, attr_info in attrs.items():
                    formatted_attrs += f"  - {attr_name}: {attr_info}\n"
    else:
        formatted_attrs = str(detailed_attrs)

    return GUROBI_MATH_MODEL_PROMPT.format(
        json_input=json.dumps(detailed_result, ensure_ascii=False, indent=2),
        detailed_attributes=formatted_attrs,
    )


def clean_gurobi_code(code: str) -> str:
    """Remove markdown fences and fix common LLM API typos."""
    lines = code.split("\n")
    # Strip leading/trailing markdown fences
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    # Strip leading blank lines
    while lines and lines[0].strip() == "":
        lines = lines[1:]
    code = "\n".join(lines)

    # Fix common gurobipy API typos
    replacements = [
        ("model.Status", "model.status"),
        ("model.Objval", "model.ObjVal"),
        ("model.objval", "model.ObjVal"),
        ("model.objVal", "model.ObjVal"),
        ("grb.GRB.optimal", "grb.GRB.OPTIMAL"),
        ("grb.GRB.Optimal", "grb.GRB.OPTIMAL"),
    ]
    for old, new in replacements:
        code = code.replace(old, new)

    return code


def json_to_gurobi_solver(
    detailed_result_path: str,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    return_metadata: bool = False,
):
    """End-to-end: JSON → Math Model → Gurobi Core → Full Solver.

    Args:
        detailed_result_path: Path to detailed_result.json.
        model_name: LLM model alias.
        max_tokens: Max tokens per generation step.

    Returns:
        Complete Gurobi solver Python code as a string.
    """
    with open(detailed_result_path, "r", encoding="utf-8") as f:
        problem_json = json.load(f)

    client = LLMClient(model=model_name)
    import time

    # Step 1: JSON → Mathematical model
    prompt1 = build_math_model_prompt(problem_json)
    result1 = client.generate(prompt1, max_tokens=max_tokens)
    math_model = result1.get("output", "") if isinstance(result1, dict) else str(result1)

    # Step 2: Mathematical model → Gurobi core code
    prompt2 = GUROBI_CORE_PROMPT.format(math_model=math_model)
    result2 = client.generate(prompt2, max_tokens=max_tokens)
    gurobi_core = result2.get("output", "") if isinstance(result2, dict) else str(result2)

    # Brief delay to avoid rate limits
    time.sleep(0.5)

    # Step 3: Gurobi core → Full solver script
    prompt3 = GUROBI_FULL_PROMPT.format(core_code=gurobi_core)
    result3 = client.generate(prompt3, max_tokens=max_tokens)
    full_code = result3.get("output", "") if isinstance(result3, dict) else str(result3)

    code = clean_gurobi_code(full_code)
    metadata = {
        "steps": {
            "math_model": {
                "model": result1.get("model"), "usage": result1.get("usage", {})
            },
            "core_code": {
                "model": result2.get("model"), "usage": result2.get("usage", {})
            },
            "full_solver": {
                "model": result3.get("model"), "usage": result3.get("usage", {})
            },
        }
    }
    return (code, metadata) if return_metadata else code


def refine_with_retrieved_cases(
    detailed_result_path: str,
    initial_code: str,
    retrieval: Dict[str, Any],
    model_name: str,
    max_tokens: int = 8192,
) -> Tuple[str, Dict[str, Any]]:
    """Refine initial solver code using AST-retrieved structural examples."""
    results = retrieval.get("results", [])
    if not results:
        raise ValueError("AST-CBOR requested but the case library yielded no usable cases")

    case_blocks = []
    for item in results:
        raw_path = item.get("raw_py")
        if not raw_path or not os.path.isfile(raw_path):
            continue
        with open(raw_path, "r", encoding="utf-8") as f:
            case_code = f.read(16000)
        case_blocks.append(
            f"Case {item['case_id']} (similarity={item['similarity']:.6f}):\n{case_code}"
        )
    if not case_blocks:
        raise ValueError("AST-CBOR retrieved cases without readable solver code")

    with open(detailed_result_path, "r", encoding="utf-8") as f:
        problem_json = json.load(f)
    prompt = GUROBI_CBR_REFINE_PROMPT.format(
        json_input=json.dumps(problem_json, ensure_ascii=False, indent=2),
        initial_code=initial_code,
        retrieved_cases="\n\n".join(case_blocks),
    )
    response = LLMClient(model=model_name).generate(prompt, max_tokens=max_tokens)
    refined = clean_gurobi_code(response.get("output", ""))
    validate_generated_code(refined)
    return refined, {
        "model": response.get("model"),
        "usage": response.get("usage", {}),
        "case_ids": [item["case_id"] for item in results],
        "similarities": [item["similarity"] for item in results],
    }


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

VALID_NO_SOLUTION_STATUSES = {"INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD"}
VALID_TERMINAL_STATUSES = {"OPTIMAL", *VALID_NO_SOLUTION_STATUSES}
ALLOWED_IMPORTS = {"gurobipy", "math", "itertools", "collections"}
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "input",
    "system", "popen", "remove", "unlink", "rmtree", "socket",
}


def _directories_overlap(first: str, second: str) -> bool:
    first_real = os.path.realpath(first)
    second_real = os.path.realpath(second)
    try:
        common = os.path.commonpath((first_real, second_real))
    except ValueError:
        return False
    return common in (first_real, second_real)


def validate_generated_code(code: str) -> None:
    """Reject syntax errors and common unsafe operations before execution."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots <= ALLOWED_IMPORTS:
                raise ValueError(f"Generated code imports disallowed modules: {sorted(roots - ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise ValueError(f"Generated code imports disallowed module: {root}")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in FORBIDDEN_CALLS:
                raise ValueError(f"Generated code uses disallowed call: {name}")


def _write_solver_result(
    dir_path: str,
    solver_status: str,
    objective: Optional[str] = None,
    returncode: Optional[int] = None,
    stdout: str = "",
    stderr: str = "",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist structured and legacy-compatible solver output files."""
    payload = {
        "solver_status": solver_status,
        "objective": objective,
        "returncode": returncode,
        "error": error,
    }
    with RESULT_LOCK:
        with open(os.path.join(dir_path, "solver_result.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(os.path.join(dir_path, "objective_result.txt"), "w", encoding="utf-8") as f:
            f.write(f"Solver status: {solver_status}\n")
            f.write(f"Return code: {returncode}\n")
            if objective is not None:
                f.write(f"Optimal objective value: {objective}\n")
            if error:
                f.write(f"Error: {error}\n")
            if stdout:
                f.write(f"Standard output:\n{stdout}\n")
            if stderr:
                f.write(f"Standard error:\n{stderr}\n")
        legacy_value = "Solver Error"
        if solver_status == "OPTIMAL" and objective is not None:
            legacy_value = objective
        elif solver_status in VALID_NO_SOLUTION_STATUSES:
            legacy_value = "No Best Solution"
        with open(os.path.join(dir_path, "objective_value.txt"), "w", encoding="utf-8") as f:
            f.write(legacy_value)
    return payload

def run_gurobi(gurobi_file_path: str, timeout: int = 60) -> Dict[str, Any]:
    """Execute a Gurobi solver script and save results.

    Returns dict with status, message, and objective value.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-I", gurobi_file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(gurobi_file_path)),
        )
        output_dir = os.path.dirname(gurobi_file_path)
        status_match = re.search(
            r"^Solver status:\s*([A-Z_]+)\s*$", result.stdout, re.MULTILINE
        )
        objective_match = re.search(
            r"^Optimal objective value\s*[:：]\s*(\S+)\s*$",
            result.stdout,
            re.MULTILINE,
        )
        solver_status = status_match.group(1) if status_match else None
        objective_value = objective_match.group(1) if objective_match else None

        if result.returncode != 0:
            _write_solver_result(
                output_dir,
                "EXECUTION_ERROR",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                error="Generated solver exited with a nonzero return code",
            )
            return {"status": "failed", "message": "Generated solver execution failed"}

        # Backward compatibility for existing scripts that only print an objective.
        if solver_status is None and objective_value is not None:
            solver_status = "OPTIMAL"
        if solver_status == "OPTIMAL":
            try:
                numeric_objective = float(objective_value) if objective_value is not None else math.nan
            except ValueError:
                numeric_objective = math.nan
            if not math.isfinite(numeric_objective):
                _write_solver_result(
                    output_dir,
                    "INVALID_OUTPUT",
                    returncode=0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error="OPTIMAL status did not include a finite objective value",
                )
                return {"status": "failed", "message": "Invalid optimal objective output"}
            _write_solver_result(
                output_dir,
                "OPTIMAL",
                objective=objective_value,
                returncode=0,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            return {
                "status": "success",
                "message": f"Objective value: {objective_value}",
                "objective": objective_value,
                "solver_status": "OPTIMAL",
            }
        if solver_status in VALID_NO_SOLUTION_STATUSES:
            _write_solver_result(
                output_dir,
                solver_status,
                returncode=0,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            return {"status": "success", "message": solver_status, "solver_status": solver_status}

        _write_solver_result(
            output_dir,
            "UNKNOWN",
            returncode=0,
            stdout=result.stdout,
            stderr=result.stderr,
            error="Solver did not emit a recognized terminal status",
        )
        return {"status": "failed", "message": "No recognized solver status"}

    except subprocess.TimeoutExpired:
        _save_no_solution(os.path.dirname(gurobi_file_path), "Execution timed out", "TIMEOUT")
        return {"status": "failed", "message": "Execution timed out"}
    except Exception as e:
        _save_no_solution(os.path.dirname(gurobi_file_path), str(e), "EXECUTION_ERROR")
        return {"status": "failed", "message": str(e)}


def _save_no_solution(
    dir_path: str,
    error_msg: str,
    solver_status: str = "EXECUTION_ERROR",
):
    """Persist a failed generation/execution without labeling it infeasible."""
    _write_solver_result(dir_path, solver_status, error=error_msg)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def should_skip(objective_value_path: str, expected_cbr: Optional[bool] = None) -> bool:
    """Return True if a problem has a successful result and can be skipped."""
    if expected_cbr is not None:
        metadata_path = os.path.join(
            os.path.dirname(objective_value_path), "generation_metadata.json"
        )
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                if bool(json.load(f).get("use_cbr")) != expected_cbr:
                    return False
        except (OSError, json.JSONDecodeError):
            return False
    result_path = os.path.join(os.path.dirname(objective_value_path), "solver_result.json")
    if os.path.isfile(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            return result.get("solver_status") in VALID_TERMINAL_STATUSES
        except (OSError, json.JSONDecodeError):
            return False
    if not os.path.exists(objective_value_path):
        return False
    try:
        with open(objective_value_path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        numeric = float(value)
        return math.isfinite(numeric)
    except Exception:
        return False


def process_single_case(
    case_path: str,
    model_name: str = DEFAULT_MODEL,
    skip: bool = True,
    case_library: Optional[str] = None,
    cbr_topk: int = 3,
    solver_timeout: int = 60,
) -> Tuple[str, Dict[str, Any]]:
    """Process a single case directory: generate and run Gurobi solver.

    Returns (path, result_dict).
    """
    if not os.path.isdir(case_path):
        return case_path, {"status": "failed", "reason": "not a directory"}

    detailed_result_path = os.path.join(case_path, "detailed_result.json")
    if not os.path.exists(detailed_result_path):
        return case_path, {"status": "skipped", "reason": "no detailed_result.json"}

    objective_value_path = os.path.join(case_path, "objective_value.txt")
    use_cbr = case_library is not None
    if use_cbr and _directories_overlap(case_path, case_library):
        return case_path, {
            "status": "failed",
            "reason": "The target case and AST-CBOR case library must be separate",
        }
    if skip and should_skip(objective_value_path, expected_cbr=use_cbr):
        try:
            with open(objective_value_path, "r", encoding="utf-8") as f:
                existing = f.read().strip()
        except Exception:
            existing = "unknown"
        return case_path, {"status": "skipped", "message": f"already processed: {existing}"}

    try:
        # Generate solver code
        initial_code, generation_metadata = json_to_gurobi_solver(
            detailed_result_path,
            model_name=model_name,
            max_tokens=4096,
            return_metadata=True,
        )
        validate_generated_code(initial_code)
        gurobi_code = initial_code
        generation_metadata.update({
            "requested_model": model_name,
            "use_cbr": use_cbr,
            "case_library": os.path.abspath(case_library) if case_library else None,
            "cbr_topk": cbr_topk if use_cbr else 0,
        })

        if use_cbr:
            from ast_cbor import discover_case_ids, retrieve_topk

            initial_path = os.path.join(case_path, "gurobi_solver.initial.py")
            with RESULT_LOCK:
                with open(initial_path, "w", encoding="utf-8") as f:
                    f.write(initial_code)
            case_ids = discover_case_ids(case_library, valid_only=True)
            retrieval = retrieve_topk(
                root_dir=case_library,
                case_ids=case_ids,
                query_code_file=initial_path,
                topk=cbr_topk,
            )
            with RESULT_LOCK:
                with open(os.path.join(case_path, "retrieval.json"), "w", encoding="utf-8") as f:
                    json.dump(retrieval, f, ensure_ascii=False, indent=2)
            gurobi_code, refinement_metadata = refine_with_retrieved_cases(
                detailed_result_path,
                initial_code,
                retrieval,
                model_name,
            )
            generation_metadata["cbr_refinement"] = refinement_metadata

        validate_generated_code(gurobi_code)
        gurobi_file_path = os.path.join(case_path, "gurobi_solver.py")
        with RESULT_LOCK:
            with open(gurobi_file_path, "w", encoding="utf-8") as f:
                f.write(gurobi_code)
            with open(
                os.path.join(case_path, "generation_metadata.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(generation_metadata, f, ensure_ascii=False, indent=2)

        # Execute
        result = run_gurobi(gurobi_file_path, timeout=solver_timeout)
        return case_path, result

    except Exception as e:
        _save_no_solution(case_path, str(e), "GENERATION_ERROR")
        return case_path, {"status": "failed", "reason": str(e)}


def generate_for_dataset(
    output_dir: str,
    model_name: str = DEFAULT_MODEL,
    start_from: int = 0,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    skip: bool = True,
    case_library: Optional[str] = None,
    cbr_topk: int = 3,
    solver_timeout: int = 60,
) -> Dict[str, int]:
    """Generate and run Gurobi solvers for all cases in an output directory.

    Returns dict with success, skipped, failed counts.
    """
    if not os.path.isdir(output_dir):
        logger.error("Directory not found: %s", output_dir)
        return {"success": 0, "skipped": 0, "failed": 0}
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    if solver_timeout < 1:
        raise ValueError("solver_timeout must be at least 1 second")
    if case_library is not None:
        if not os.path.isdir(case_library):
            raise FileNotFoundError(f"Case library not found: {case_library}")
        if _directories_overlap(case_library, output_dir):
            raise ValueError("The AST-CBOR case library must be separate from the target output directory")
        if cbr_topk < 1:
            raise ValueError("cbr_topk must be at least 1")
        from ast_cbor import batch_process, discover_case_ids
        if not discover_case_ids(case_library, valid_only=True):
            raise ValueError("The AST-CBOR case library has no valid OPTIMAL solver cases")
        # Build candidate artifacts once before parallel query retrieval begins.
        batch_process(case_library, "gurobi_solver.py")

    subdirs = sorted(
        [d for d in os.listdir(output_dir)
         if os.path.isdir(os.path.join(output_dir, d))],
        key=lambda x: (
            int(x.split("_")[0]) if x.split("_")[0].isdigit() else float("inf"),
            x,
        ),
    )

    paths = []
    previously_skipped = 0
    for subdir in subdirs:
        if start_from > 0:
            try:
                num = int(subdir.split("_")[0])
                if num < start_from:
                    continue
            except ValueError:
                continue
        subdir_path = os.path.join(output_dir, subdir)
        detailed_path = os.path.join(subdir_path, "detailed_result.json")
        if not os.path.exists(detailed_path):
            continue
        if skip and should_skip(
            os.path.join(subdir_path, "objective_value.txt"),
            expected_cbr=case_library is not None,
        ):
            previously_skipped += 1
            continue
        paths.append(subdir_path)

    logger.info(
        "Processing %d cases in %s (max_parallel=%d)",
        len(paths), output_dir, max_parallel,
    )

    success, skipped, failed = 0, previously_skipped, 0

    if not paths:
        return {"success": 0, "skipped": skipped, "failed": 0}

    # Validate credentials once before worker threads or paid calls begin.
    LLMClient(model=model_name)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                process_single_case,
                p,
                model_name,
                skip,
                case_library,
                cbr_topk,
                solver_timeout,
            ): p
            for p in paths
        }
        for future in as_completed(futures):
            try:
                path, result = future.result()
            except Exception as exc:
                path = futures[future]
                result = {"status": "failed", "reason": str(exc)}
            problem_id = os.path.basename(path)
            status = result.get("status", "failed")
            if status == "success":
                logger.info("  OK %s: %s", problem_id, result.get("message", ""))
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                logger.info("  FAIL %s: %s", problem_id, result.get("reason", result.get("message", "")))
                failed += 1

    logger.info(
        "Results for %s - success=%d skipped=%d failed=%d",
        output_dir, success, skipped, failed,
    )
    return {"success": success, "skipped": skipped, "failed": failed}


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Gurobi solver code")
    parser.add_argument("output_dir", help="Directory containing detailed_result.json files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model alias")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--skip", action="store_true", default=True)
    parser.add_argument("--no-skip", dest="skip", action="store_false")
    parser.add_argument("--case-library", default=None)
    parser.add_argument("--cbr-topk", type=int, default=3)
    parser.add_argument("--solver-timeout", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_for_dataset(
        args.output_dir, args.model, args.start_from,
        args.max_parallel,
        args.skip,
        args.case_library,
        args.cbr_topk,
        args.solver_timeout,
    )
