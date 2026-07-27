"""
Full pipeline orchestration: extract → generate → evaluate.

Provides a single entry point that runs the complete modeling pipeline
on one or more datasets, with progress reporting and summary statistics.
"""

import os
import json
import logging
import math
import shutil
import platform
import importlib.metadata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from datasets import DATASET_META
from extract import process_dataset
from generate import generate_for_dataset
from ast_cbor import batch_process as ast_batch_process
from llm import DEFAULT_MODEL

logger = logging.getLogger(__name__)


def _runtime_metadata() -> Dict[str, Any]:
    versions = {}
    for package in ("openai", "anthropic", "gurobipy", "python-dotenv"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "llm_temperature": 0.0,
        "gurobi_seed": 0,
    }


def run_pipeline(
    dataset_key: str,
    model: str = DEFAULT_MODEL,
    max_parallel_extract: int = 50,
    max_parallel_generate: int = 30,
    start_from: int = 0,
    single: Optional[int] = None,
    output_dir: Optional[str] = None,
    skip_extract: bool = False,
    skip_generate: bool = False,
    skip_ast: bool = False,
    skip_evaluate: bool = False,
    case_library: Optional[str] = None,
    cbr_topk: int = 3,
    solver_timeout: int = 60,
) -> Dict[str, Any]:
    """Run the full pipeline on a single dataset.

    Args:
        dataset_key: Dataset name (nl4opt, nlp4lp, industryor, etc.).
        model: LLM model alias.
        max_parallel_extract: Max parallel LLM calls during extraction.
        max_parallel_generate: Max parallel calls during code generation.
        start_from: Skip problems with index < start_from.
        single: Process only this problem index.
        output_dir: Custom output directory.
        skip_extract: Skip the extraction phase.
        skip_generate: Skip the Gurobi generation phase.
        skip_ast: Skip the AST analysis phase.

    Returns:
        Summary dict with phase results.
    """
    if output_dir is None:
        output_dir = f"output_{dataset_key}"

    summary = {
        "dataset": dataset_key,
        "model": model,
        "output_dir": output_dir,
        "extraction": None,
        "generation": None,
        "ast": None,
        "evaluation": None,
    }

    # ---- Phase 1: Extraction ----
    if not skip_extract:
        logger.info("=" * 60)
        logger.info("Phase 1: Extraction - %s", dataset_key)
        logger.info("=" * 60)

        summary["extraction"] = process_dataset(
            dataset_key=dataset_key,
            model_name=model,
            start_from=start_from,
            single=single,
            max_parallel=max_parallel_extract,
            output_dir=output_dir,
        )
    else:
        logger.info("Phase 1: Extraction SKIPPED")
        summary["extraction"] = {"skipped": True}

    # ---- Phase 2: Gurobi Code Generation ----
    if not skip_generate:
        logger.info("=" * 60)
        logger.info("Phase 2: Gurobi Code Generation - %s", dataset_key)
        logger.info("=" * 60)

        summary["generation"] = generate_for_dataset(
            output_dir=output_dir,
            model_name=model,
            start_from=start_from,
            max_parallel=max_parallel_generate,
            skip=True,
            case_library=case_library,
            cbr_topk=cbr_topk,
            solver_timeout=solver_timeout,
        )
    else:
        logger.info("Phase 2: Gurobi Generation SKIPPED")
        summary["generation"] = {"skipped": True}

    # ---- Phase 3: AST Analysis ----
    if not skip_ast:
        logger.info("=" * 60)
        logger.info("Phase 3: AST Analysis - %s", dataset_key)
        logger.info("=" * 60)

        summary["ast"] = ast_batch_process(output_dir, "gurobi_solver.py")
    else:
        logger.info("Phase 3: AST Analysis SKIPPED")
        summary["ast"] = {"skipped": True}

    # ---- Phase 4: Evaluation ----
    if not skip_evaluate:
        logger.info("=" * 60)
        logger.info("Phase 4: Evaluation - %s", dataset_key)
        logger.info("=" * 60)
        summary["evaluation"] = evaluate_dataset(output_dir, dataset_key)
    else:
        summary["evaluation"] = {"skipped": True}

    return summary


def run_all_datasets(
    model: str = DEFAULT_MODEL,
    datasets: Optional[List[str]] = None,
    max_parallel_extract: int = 50,
    max_parallel_generate: int = 30,
    skip_extract: bool = False,
    skip_generate: bool = False,
    skip_ast: bool = False,
    skip_evaluate: bool = False,
    case_library: Optional[str] = None,
    cbr_topk: int = 3,
    solver_timeout: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """Run the full pipeline across multiple datasets.

    Args:
        model: LLM model alias.
        datasets: List of dataset keys; defaults to all supported.
        max_parallel_extract: Max parallel extraction calls.
        max_parallel_generate: Max parallel generation calls.
        skip_extract: Skip extraction across all datasets.
        skip_generate: Skip generation across all datasets.
        skip_ast: Skip AST analysis across all datasets.

    Returns:
        {dataset_key: summary_dict} mapping.
    """
    if datasets is None:
        datasets = list(DATASET_META.keys())

    logger.info("=" * 70)
    logger.info("  AST_CBOR Pipeline - %d datasets", len(datasets))
    logger.info("  Model: %s  Extract workers: %d  Generate workers: %d",
                model, max_parallel_extract, max_parallel_generate)
    logger.info("=" * 70)

    results = {}
    for ds in datasets:
        logger.info("\n>>> Processing: %s", ds)
        results[ds] = run_pipeline(
            dataset_key=ds,
            model=model,
            max_parallel_extract=max_parallel_extract,
            max_parallel_generate=max_parallel_generate,
            skip_extract=skip_extract,
            skip_generate=skip_generate,
            skip_ast=skip_ast,
            skip_evaluate=skip_evaluate,
            case_library=case_library,
            cbr_topk=cbr_topk,
            solver_timeout=solver_timeout,
        )

    # Print overall summary
    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline Complete")
    logger.info("=" * 70)
    for ds, summary in results.items():
        ext = summary.get("extraction", {}) or {}
        gen = summary.get("generation", {}) or {}
        ast = summary.get("ast", {}) or {}
        evaluation = summary.get("evaluation", {}) or {}
        logger.info(
            "  %s: extract=%s gen=%s ast=%s accuracy=%s",
            ds,
            ext.get("success", ext.get("skipped", "?")),
            gen.get("success", gen.get("skipped", "?")),
            ast.get("ok", ast.get("skipped", "?")),
            evaluation.get("accuracy_percent", evaluation.get("skipped", "?")),
        )

    return results


def _prepare_generation_workspace(source_dir: str, target_dir: str) -> int:
    """Copy only extraction artifacts needed for an isolated generation run."""
    os.makedirs(target_dir, exist_ok=True)
    copied = 0
    for name in sorted(os.listdir(source_dir)):
        source_case = os.path.join(source_dir, name)
        if not name.isdigit() or not os.path.isdir(source_case):
            continue
        detailed = os.path.join(source_case, "detailed_result.json")
        if not os.path.isfile(detailed):
            continue
        target_case = os.path.join(target_dir, name)
        os.makedirs(target_case, exist_ok=True)
        for filename in ("detailed_result.json", "result.json", "modules_summary.json"):
            source_file = os.path.join(source_case, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, os.path.join(target_case, filename))
        copied += 1
    return copied


def _aggregate_generation_usage(output_dir: str) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for root, _, files in os.walk(output_dir):
        if "generation_metadata.json" not in files:
            continue
        try:
            with open(os.path.join(root, "generation_metadata.json"), "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        usages = [
            step.get("usage", {})
            for step in metadata.get("steps", {}).values()
            if isinstance(step, dict)
        ]
        refinement = metadata.get("cbr_refinement", {})
        if isinstance(refinement, dict):
            usages.append(refinement.get("usage", {}))
        for usage in usages:
            if not isinstance(usage, dict):
                continue
            for key, value in usage.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
    return totals


def run_experiment(
    extracted_dir: str,
    dataset_key: str,
    model: str,
    case_library: str,
    experiment_dir: Optional[str] = None,
    max_parallel_generate: int = 10,
    cbr_topk: int = 3,
    solver_timeout: int = 60,
) -> Dict[str, Any]:
    """Run baseline and AST-CBOR generation on the same extracted cases."""
    if not os.path.isdir(extracted_dir):
        raise FileNotFoundError(f"Extraction directory not found: {extracted_dir}")
    if not os.path.isdir(case_library):
        raise FileNotFoundError(f"Case library not found: {case_library}")
    if os.path.realpath(extracted_dir) == os.path.realpath(case_library):
        raise ValueError("Use a separate training case library to avoid test-set leakage")

    if experiment_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_dir = f"experiment_{dataset_key}_{stamp}"
    elif os.path.isdir(experiment_dir) and os.listdir(experiment_dir):
        raise FileExistsError(f"Experiment directory is not empty: {experiment_dir}")
    baseline_dir = os.path.join(experiment_dir, "baseline")
    cbr_dir = os.path.join(experiment_dir, "ast_cbor")
    baseline_count = _prepare_generation_workspace(extracted_dir, baseline_dir)
    cbr_count = _prepare_generation_workspace(extracted_dir, cbr_dir)
    if baseline_count == 0 or cbr_count == 0:
        raise ValueError("No completed detailed_result.json cases were found")

    baseline_generation = generate_for_dataset(
        baseline_dir,
        model_name=model,
        max_parallel=max_parallel_generate,
        skip=False,
        case_library=None,
        solver_timeout=solver_timeout,
    )
    cbr_generation = generate_for_dataset(
        cbr_dir,
        model_name=model,
        max_parallel=max_parallel_generate,
        skip=False,
        case_library=case_library,
        cbr_topk=cbr_topk,
        solver_timeout=solver_timeout,
    )
    baseline_ast = ast_batch_process(baseline_dir, "gurobi_solver.py")
    cbr_ast = ast_batch_process(cbr_dir, "gurobi_solver.py")
    baseline_evaluation = evaluate_dataset(baseline_dir, dataset_key)
    cbr_evaluation = evaluate_dataset(cbr_dir, dataset_key)

    summary = {
        "dataset": dataset_key,
        "model": model,
        "case_library": os.path.abspath(case_library),
        "cbr_topk": cbr_topk,
        "case_count": baseline_count,
        "runtime": _runtime_metadata(),
        "baseline": {
            "output_dir": os.path.abspath(baseline_dir),
            "generation": baseline_generation,
            "token_usage": _aggregate_generation_usage(baseline_dir),
            "ast": baseline_ast,
            "evaluation": baseline_evaluation,
        },
        "ast_cbor": {
            "output_dir": os.path.abspath(cbr_dir),
            "generation": cbr_generation,
            "token_usage": _aggregate_generation_usage(cbr_dir),
            "ast": cbr_ast,
            "evaluation": cbr_evaluation,
        },
    }
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def evaluate_dataset(
    output_dir: str,
    dataset_key: str,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Compare generated objective values against ground truth.

    Returns {correct, total, accuracy_percent, per_problem: {pid: match_bool}}.
    """
    if not os.path.isdir(output_dir):
        return {"error": f"Output directory not found: {output_dir}"}

    from datasets import get_dataset_path, read_json_records

    # Collect structured solver outcomes. Legacy numeric outputs remain supported,
    # but a legacy "No Best Solution" marker is not trusted as proof of infeasibility.
    generated = {}
    for d in os.listdir(output_dir):
        sp = os.path.join(output_dir, d)
        if not os.path.isdir(sp) or not d.isdigit():
            continue
        pid = int(d)
        result_file = os.path.join(sp, "solver_result.json")
        obj_file = os.path.join(sp, "objective_value.txt")
        if os.path.isfile(result_file):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    generated[pid] = json.load(f)
            except (OSError, json.JSONDecodeError):
                generated[pid] = {"solver_status": "INVALID_RESULT", "objective": None}
        elif os.path.isfile(obj_file):
            with open(obj_file, "r", encoding="utf-8") as f:
                value = f.read().strip()
            try:
                numeric = float(value)
                finite = math.isfinite(numeric)
            except ValueError:
                finite = False
            generated[pid] = {
                "solver_status": "OPTIMAL" if finite else "LEGACY_UNKNOWN",
                "objective": value if finite else None,
            }

    if not generated:
        return {"error": "No generated results found"}

    no_solution_answers = {"No Best Solution", "-99999", "-99999.0", "-99999.00"}
    valid_no_solution_statuses = {"INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD"}

    def extract_answer(item):
        for key in (
            "en_answer", "answer", "objective", "objective_value",
            "obj_val", "optimal_value",
        ):
            if key in item and item[key] not in (None, ""):
                return str(item[key])
        solution = item.get("solution")
        if isinstance(solution, dict):
            return extract_answer(solution)
        if isinstance(solution, str) and solution.strip():
            try:
                parsed = json.loads(solution)
            except json.JSONDecodeError:
                return solution.strip()
            if isinstance(parsed, dict):
                return extract_answer(parsed)
        return None

    # Load ground truth based on dataset type
    ds_path = get_dataset_path(dataset_key)
    answers = {}
    meta = DATASET_META.get(dataset_key, {})
    filename = meta.get("files", [""])[0]
    filepath = os.path.join(ds_path, filename)
    if os.path.isfile(filepath):
        for idx, item in enumerate(read_json_records(filepath)):
            answer = extract_answer(item)
            if answer is None:
                continue
            raw_pid = item.get("id", item.get("problem_id", idx))
            pid = int(raw_pid) if str(raw_pid).isdigit() else idx
            answers[pid] = answer
            answers.setdefault(idx, answer)
    elif dataset_key == "nlp4lp":
        for root, _, files in os.walk(ds_path):
            if "solution.json" not in files:
                continue
            dirname = os.path.basename(root)
            if not dirname.isdigit():
                continue
            try:
                with open(os.path.join(root, "solution.json"), "r", encoding="utf-8") as f:
                    answer = extract_answer(json.load(f))
                if answer is not None:
                    answers[int(dirname)] = answer
            except (OSError, json.JSONDecodeError):
                continue

    if not answers:
        return {"error": f"No ground-truth answers found for dataset: {dataset_key}"}

    # Compare
    correct, total, valid, failed = 0, 0, 0, 0
    per_problem = {}
    solver_statuses = {}
    for pid, outcome in generated.items():
        ev = answers.get(pid)
        if ev is None:
            continue
        solver_status = str(outcome.get("solver_status", "UNKNOWN")).upper()
        solver_statuses[pid] = solver_status
        gv = outcome.get("objective")
        match = False
        if solver_status in valid_no_solution_statuses:
            valid += 1
            match = ev in no_solution_answers
        elif solver_status == "OPTIMAL" and gv is not None:
            valid += 1
            if ev not in no_solution_answers:
                try:
                    gvf, evf = float(gv), float(ev)
                    if math.isfinite(gvf) and math.isfinite(evf):
                        match = abs(gvf - evf) < absolute_tolerance or (
                            evf != 0 and abs(gvf - evf) / abs(evf) < relative_tolerance
                        )
                except (TypeError, ValueError):
                    pass
        else:
            failed += 1
        if match:
            correct += 1
        total += 1
        per_problem[pid] = match

    acc = correct / total * 100 if total > 0 else 0.0
    if total == 0:
        return {
            "error": "Generated case IDs did not match any ground-truth IDs",
            "generated_ids": sorted(generated),
        }
    return {
        "correct": correct,
        "total": total,
        "valid": valid,
        "failed": failed,
        "accuracy_percent": round(acc, 1),
        "per_problem": per_problem,
        "solver_statuses": solver_statuses,
    }
