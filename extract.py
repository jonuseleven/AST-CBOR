"""
Six-step extraction pipeline for optimization problem modeling.

Pipeline steps:
    0. Normalization — rewrite the problem in structured, unambiguous language
    1. Entities     — identify physical objects / decision-variable entities
    2. Attributes   — extract essential attributes of each entity
    3. Objectives   — identify the single optimization objective
    4. Variables    — determine decision variables
    5. Constraints  — list all constraints with explanations and formulas

Each step calls an LLM with a specialized prompt, parses the output,
and feeds structured results into downstream steps.
"""

import os
import re
import json
import threading
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import (
    NORMALIZER_PROMPT,
    ENTITY_EXTRACTOR_PROMPT,
    ATTRIBUTE_EXTRACTOR_PROMPT,
    OBJECTIVE_EXTRACTOR_PROMPT,
    VARIABLE_EXTRACTOR_PROMPT,
    CONSTRAINT_EXTRACTOR_PROMPT,
    EXAMPLE_PROBLEM,
    EXAMPLE_ENTITIES,
    EXAMPLE_ATTRIBUTES,
    EXAMPLE_CONSTRAINTS,
    EXAMPLE_CONSTRAINT_EXPLANATIONS,
    EXAMPLE_OBJECTIVE,
    EXAMPLE_OBJECTIVE_CALCULATION,
    EXAMPLE_DECISION_VARIABLES,
    EXAMPLE_DECISION_VARIABLES_CALCULATION,
)
from llm import DEFAULT_MODEL, LLMClient
from datasets import load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_PARALLEL = 50
RESULT_LOCK = threading.Lock()

# Token configuration per dataset type: (base_max_tokens, token_increment, max_attempts)
TOKEN_CONFIGS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "nl4opt": {
        "normalization": (512, 512, 6),
        "entities": (256, 256, 6),
        "attributes": (1024, 1024, 3),
        "objectives": (512, 1024, 6),
        "decision_variables": (512, 1024, 6),
        "constraints": (1024, 1024, 3),
    },
    "nlp4lp": {
        "normalization": (512, 512, 6),
        "entities": (256, 256, 6),
        "attributes": (1024, 1024, 3),
        "objectives": (512, 512, 6),
        "decision_variables": (512, 512, 6),
        "constraints": (1024, 1024, 3),
    },
    "mamo_easy": {
        "normalization": (512, 512, 6),
        "entities": (256, 256, 6),
        "attributes": (1024, 1024, 3),
        "objectives": (512, 1024, 6),
        "decision_variables": (512, 1024, 6),
        "constraints": (1024, 1024, 3),
    },
    "mamo_complex": {
        "normalization": (2048, 512, 6),
        "entities": (1024, 256, 6),
        "attributes": (4096, 1024, 3),
        "objectives": (2048, 1024, 3),
        "decision_variables": (2048, 1024, 3),
        "constraints": (4096, 4096, 3),
    },
    "industryor": {
        "normalization": (2048, 512, 6),
        "entities": (1024, 256, 6),
        "attributes": (4096, 1024, 3),
        "objectives": (2048, 1024, 3),
        "decision_variables": (2048, 1024, 3),
        "constraints": (4096, 4096, 3),
    },
}


def get_token_config(dataset_type: str) -> Dict[str, Tuple[int, int, int]]:
    """Resolve token configuration for a dataset type."""
    key = dataset_type.lower() if dataset_type else "nlp4lp"
    if key.startswith("mamo_") and key not in TOKEN_CONFIGS:
        key = "mamo_easy"
    return TOKEN_CONFIGS.get(key, TOKEN_CONFIGS["nlp4lp"])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def clean_llm_output(text: str) -> str:
    """Remove common LLM response prefixes."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    for prefix in ("Answer:", "Answer: ", "Entities:", "Entities: "):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned


def _extract_labeled_value(text: str, label: str) -> str:
    """Extract a non-empty `Label: value` line from an LLM response."""
    pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return ""


def _load_completed_modules(
    case_dir: str,
    final_result: Dict[str, Any],
    detailed_result: Dict[str, Any],
) -> set:
    """Rehydrate valid module outputs so resumed runs preserve prior work."""
    mapping = {
        "00_normalization": ("normalized_description", "normalized_desc", "normalization"),
        "01_entities": ("entities", "entities", "entities"),
        "02_attributes": ("attributes", "attributes", "attributes"),
        "03_objectives": ("objectives", "objectives", "objectives"),
        "04_decision_variables": (
            "decision_variables", "decision_variables", "decision_variables"
        ),
        "05_constraints": ("constraints", "constraints", "constraints"),
    }
    completed = set()
    for module_name, (cached_key, result_key, detail_key) in mapping.items():
        output_path = os.path.join(case_dir, module_name, "output.json")
        if not os.path.isfile(output_path):
            continue
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            value = cached.get(cached_key)
            if not value or (isinstance(value, str) and not value.strip()):
                continue
            final_result[result_key] = value
            detailed_result[detail_key] = cached.get("full_output") or value
            completed.add(module_name)
            logger.info("  OK %s already completed, reusing output...", module_name)
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("  Ignoring invalid cached module: %s", output_path)
    return completed


def _is_permanent_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "api key", "authentication", "unauthorized", "invalid model",
        "model_not_found", "permission", "401", "403", "404",
    )
    return any(marker in text for marker in markers)


def validate_result_format(result: Dict[str, Any]) -> bool:
    """Check that a final result dict has all required fields."""
    required = [
        "original_question", "normalized_desc", "entities",
        "attributes", "objectives", "decision_variables", "constraints",
    ]
    for field in required:
        if field not in result:
            return False
        value = result[field]
        if not value or (isinstance(value, str) and not value.strip()):
            return False
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in ("Answer:", "Answer: ", "Entities:", "Entities:: "):
                return False

    for field, min_count in [
        ("objectives", 1), ("decision_variables", 1),
        ("entities", 1), ("constraints", 1),
    ]:
        val = result[field]
        if isinstance(val, str):
            items = [v.strip() for v in val.split(",") if v.strip()]
            if len(items) < min_count:
                return False
    return True


# ---------------------------------------------------------------------------
# Module runner (generic retry logic)
# ---------------------------------------------------------------------------

def _run_module(
    module_name: str,
    case_dir: str,
    completed_modules: set,
    extract_fn: Callable,
    extract_kwargs: dict,
    token_cfg: Tuple[int, int, int],
    parse_fn: Callable,
    save_fn: Callable,
) -> Tuple[bool, str, dict]:
    """Generic helper: run one extraction module with retry logic.

    Returns (ok, status, metadata).
    """
    if module_name in completed_modules:
        return True, "skipped", {"reason": "already completed"}

    module_dir = os.path.join(case_dir, module_name)
    base_max_tokens, token_increment, max_attempts = token_cfg

    for attempt in range(max_attempts):
        try:
            os.makedirs(module_dir, exist_ok=True)
            current_max = base_max_tokens + attempt * token_increment
            extract_kwargs["max_tokens"] = current_max
            output = extract_fn(**extract_kwargs)

            success, parsed = parse_fn(output)
            if success:
                save_fn(module_dir, parsed, output)
                metadata = {
                    "attempts": attempt + 1,
                    "max_tokens": current_max,
                }
                if isinstance(output, dict):
                    if output.get("model"):
                        metadata["model"] = output["model"]
                    if output.get("usage"):
                        metadata["usage"] = output["usage"]
                return True, "success", metadata

            if attempt + 1 < max_attempts:
                logger.info(
                    "  Retry %s attempt %d, tokens %d...",
                    module_name, attempt + 1, current_max + token_increment,
                )
        except Exception as e:
            if _is_permanent_llm_error(e):
                return False, "failed", {"reason": str(e), "attempts": attempt + 1}
            if attempt + 1 >= max_attempts:
                return False, "failed", {
                    "reason": str(e),
                    "attempts": max_attempts,
                }

    return False, "failed", {
        "reason": "Invalid output after max attempts",
        "attempts": max_attempts,
    }


# ---------------------------------------------------------------------------
# Per-problem processor
# ---------------------------------------------------------------------------

def process_single_problem(
    problem_index: int,
    target_question: str,
    output_dir: str,
    model_name: str,
    dataset_type: str,
) -> Tuple[int, dict]:
    """Run the full 6-step extraction pipeline on a single problem."""
    case_dir = os.path.join(output_dir, str(problem_index))
    result_file = os.path.join(case_dir, "result.json")

    # Check if already processed
    if os.path.exists(result_file):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if validate_result_format(existing):
                return problem_index, {"status": "skipped"}
        except Exception:
            pass

    logger.info("Processing problem %d...", problem_index)
    os.makedirs(case_dir, exist_ok=True)

    final_result: Dict[str, Any] = {"original_question": target_question}
    detailed_result: Dict[str, Any] = {
        "original_question": target_question,
        "model": model_name,
        "dataset_type": dataset_type,
    }
    module_outputs: Dict[str, Any] = {}
    token_cfg = get_token_config(dataset_type)

    completed_modules = _load_completed_modules(case_dir, final_result, detailed_result)
    client = None if len(completed_modules) == 6 else LLMClient(model=model_name)

    # ---- Step 0: Normalization ----
    def _call_normalizer(**kw):
        example_block = ""
        if EXAMPLE_PROBLEM:
            example_block = (
                "Example:\nProblem:\n" + EXAMPLE_PROBLEM + "\n\n"
            )
        prompt = NORMALIZER_PROMPT.format(
            example_block=example_block,
            problem_description=target_question,
        )
        return client.generate(prompt, max_tokens=kw["max_tokens"])

    def _parse_norm(output):
        text = output.get("output", "").strip() if isinstance(output, dict) else str(output)
        if text and text.strip() not in ("Answer:", "Answer: "):
            return True, clean_llm_output(text)
        return False, None

    def _save_norm(mod_dir, parsed, output):
        full = output.get("output", "") if isinstance(output, dict) else str(output)
        with open(os.path.join(mod_dir, "input.txt"), "w", encoding="utf-8") as f:
            f.write(target_question)
        with open(os.path.join(mod_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump({"normalized_description": parsed, "full_output": full},
                      f, ensure_ascii=False, indent=2)
        final_result["normalized_desc"] = parsed
        detailed_result["normalization"] = full

    ok, status, meta = _run_module(
        "00_normalization", case_dir, completed_modules,
        _call_normalizer, {}, token_cfg["normalization"],
        _parse_norm, _save_norm,
    )
    module_outputs["normalization"] = {"status": status, **meta}
    if not ok:
        _save_partial(case_dir, problem_index, final_result, detailed_result, module_outputs)
        return problem_index, {"status": "partial", "modules_processed": 0}

    norm_text = final_result.get("normalized_desc", target_question)

    # ---- Step 1: Entities ----
    def _build_example_block(is_example=False):
        if not is_example and not EXAMPLE_PROBLEM:
            return ""
        problem = EXAMPLE_PROBLEM if EXAMPLE_PROBLEM else ""
        answer = EXAMPLE_ENTITIES if EXAMPLE_ENTITIES else ""
        if problem and answer:
            return f"Example:\nProblem:\n{problem}\nAnswer: {answer}\n\n"
        return ""

    def _call_entities(**kw):
        example = _build_example_block()
        prompt = ENTITY_EXTRACTOR_PROMPT.format(
            example_block=example, target_problem=norm_text,
        )
        return client.generate(prompt, max_tokens=kw["max_tokens"])

    def _parse_entities(output):
        text = output.get("output", "") if isinstance(output, dict) else str(output)
        text = clean_llm_output(text)
        # Extract comma-separated entities from the answer
        items = [x.strip() for x in text.split(",") if x.strip()]
        return bool(items), ", ".join(items) if items else ""

    def _save_entities(mod_dir, parsed, output):
        full = output.get("output", "") if isinstance(output, dict) else str(output)
        with open(os.path.join(mod_dir, "input.json"), "w", encoding="utf-8") as f:
            json.dump({"problem": norm_text}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(mod_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump({"entities": parsed, "full_output": full},
                      f, ensure_ascii=False, indent=2)
        final_result["entities"] = parsed
        detailed_result["entities"] = full

    ok, status, meta = _run_module(
        "01_entities", case_dir, completed_modules,
        _call_entities, {}, token_cfg["entities"],
        _parse_entities, _save_entities,
    )
    module_outputs["entities"] = {"status": status, **meta}
    if not ok:
        _save_partial(case_dir, problem_index, final_result, detailed_result, module_outputs)
        return problem_index, {"status": "partial", "modules_processed": 1}
    entities_text = final_result.get("entities", "")

    # ---- Step 2: Attributes ----
    def _call_attrs(**kw):
        example = _build_example_block()
        # Use the detailed attributes example for the example block
        example_attrs = (
            f"Problem:\n{EXAMPLE_PROBLEM}\n"
            f"Answer: {EXAMPLE_ATTRIBUTES}\n\n"
        ) if EXAMPLE_PROBLEM else ""
        prompt = ATTRIBUTE_EXTRACTOR_PROMPT.format(
            example_block=example_attrs,
            target_problem=norm_text,
            target_entities=entities_text,
        )
        return client.generate(prompt, max_tokens=kw["max_tokens"])

    def _parse_attrs(output):
        text = output.get("output", "") if isinstance(output, dict) else str(output)
        text = clean_llm_output(text)
        return bool(text.strip()), text

    def _save_attrs(mod_dir, parsed, output):
        full = output.get("output", "") if isinstance(output, dict) else str(output)
        with open(os.path.join(mod_dir, "input.json"), "w", encoding="utf-8") as f:
            json.dump({"problem": norm_text, "entities": entities_text},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(mod_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump({"attributes": parsed, "full_output": full},
                      f, ensure_ascii=False, indent=2)
        final_result["attributes"] = parsed
        detailed_result["attributes"] = full

    ok, status, meta = _run_module(
        "02_attributes", case_dir, completed_modules,
        _call_attrs, {}, token_cfg["attributes"],
        _parse_attrs, _save_attrs,
    )
    module_outputs["attributes"] = {"status": status, **meta}
    attrs_text = final_result.get("attributes", "")

    if not ok or not attrs_text:
        _save_partial(case_dir, problem_index, final_result, detailed_result, module_outputs)
        return problem_index, {
            "status": "partial", "modules_processed": 2,
            "errors": {"attributes": "Failed"},
        }

    # ---- Step 3: Objectives ----
    def _call_obj(**kw):
        example = (
            f"Problem:\n{EXAMPLE_PROBLEM}\n"
            f"Answer: {EXAMPLE_OBJECTIVE}\n"
            f"Calculation: {EXAMPLE_OBJECTIVE_CALCULATION}\n\n"
        ) if EXAMPLE_PROBLEM else ""
        prompt = OBJECTIVE_EXTRACTOR_PROMPT.format(
            example_block=example,
            target_problem=norm_text,
            target_entities=entities_text,
            target_attributes=attrs_text,
        )
        return client.generate(prompt, max_tokens=kw["max_tokens"])

    def _parse_obj(output):
        text = output.get("output", "") if isinstance(output, dict) else str(output)
        return bool(_extract_labeled_value(text, "Answer")), text

    def _save_obj(mod_dir, parsed, output):
        full = output.get("output", "") if isinstance(output, dict) else str(output)
        # Extract answer and calculation lines
        answer = _extract_labeled_value(parsed, "Answer")
        calc = _extract_labeled_value(parsed, "Calculation")
        with open(os.path.join(mod_dir, "input.json"), "w", encoding="utf-8") as f:
            json.dump({"problem": norm_text, "entities": entities_text,
                       "attributes": attrs_text}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(mod_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump({"objectives": answer, "calculation": calc, "full_output": full},
                      f, ensure_ascii=False, indent=2)
        final_result["objectives"] = answer
        detailed_result["objectives"] = full

    ok, status, meta = _run_module(
        "03_objectives", case_dir, completed_modules,
        _call_obj, {}, token_cfg["objectives"],
        _parse_obj, _save_obj,
    )
    module_outputs["objectives"] = {"status": status, **meta}
    if not ok:
        _save_partial(case_dir, problem_index, final_result, detailed_result, module_outputs)
        return problem_index, {"status": "partial", "modules_processed": 3}

    # ---- Step 4: Decision Variables ----
    def _call_dev(**kw):
        example = (
            f"Problem:\n{EXAMPLE_PROBLEM}\n"
            f"Answer: {EXAMPLE_DECISION_VARIABLES}\n"
            f"Calculation: {EXAMPLE_DECISION_VARIABLES_CALCULATION}\n\n"
        ) if EXAMPLE_PROBLEM else ""
        prompt = VARIABLE_EXTRACTOR_PROMPT.format(
            example_block=example,
            target_problem=norm_text,
            target_entities=entities_text,
            target_attributes=attrs_text,
        )
        return client.generate(prompt, max_tokens=kw["max_tokens"])

    def _parse_dev(output):
        text = output.get("output", "") if isinstance(output, dict) else str(output)
        return bool(_extract_labeled_value(text, "Answer")), text

    def _save_dev(mod_dir, parsed, output):
        full = output.get("output", "") if isinstance(output, dict) else str(output)
        answer = _extract_labeled_value(parsed, "Answer")
        calc = _extract_labeled_value(parsed, "Calculation")
        with open(os.path.join(mod_dir, "input.json"), "w", encoding="utf-8") as f:
            json.dump({"problem": norm_text, "entities": entities_text,
                       "attributes": attrs_text}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(mod_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump({"decision_variables": answer, "calculation": calc,
                       "full_output": full}, f, ensure_ascii=False, indent=2)
        final_result["decision_variables"] = answer
        detailed_result["decision_variables"] = full

    ok, status, meta = _run_module(
        "04_decision_variables", case_dir, completed_modules,
        _call_dev, {}, token_cfg["decision_variables"],
        _parse_dev, _save_dev,
    )
    module_outputs["decision_variables"] = {"status": status, **meta}
    if not ok:
        _save_partial(case_dir, problem_index, final_result, detailed_result, module_outputs)
        return problem_index, {"status": "partial", "modules_processed": 4}

    # ---- Step 5: Constraints ----
    def _call_cons(**kw):
        example = (
            f"Problem:\n{EXAMPLE_PROBLEM}\n"
            f"Answer: {EXAMPLE_CONSTRAINTS}\n"
            f"{EXAMPLE_CONSTRAINT_EXPLANATIONS}\n\n"
        ) if EXAMPLE_PROBLEM else ""
        prompt = CONSTRAINT_EXTRACTOR_PROMPT.format(
            example_block=example,
            target_problem=norm_text,
            target_entities=entities_text,
            target_attributes=attrs_text,
        )
        return client.generate(prompt, max_tokens=kw["max_tokens"])

    def _parse_cons(output):
        text = output.get("output", "") if isinstance(output, dict) else str(output)
        return bool(_extract_labeled_value(text, "Answer")), text

    def _save_cons(mod_dir, parsed, output):
        full = output.get("output", "") if isinstance(output, dict) else str(output)
        answer = _extract_labeled_value(parsed, "Answer")
        with open(os.path.join(mod_dir, "input.json"), "w", encoding="utf-8") as f:
            json.dump({"problem": norm_text, "entities": entities_text,
                       "attributes": attrs_text}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(mod_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump({"constraints": answer, "full_output": full},
                      f, ensure_ascii=False, indent=2)
        final_result["constraints"] = answer
        detailed_result["constraints"] = full

    ok, status, meta = _run_module(
        "05_constraints", case_dir, completed_modules,
        _call_cons, {}, token_cfg["constraints"],
        _parse_cons, _save_cons,
    )
    module_outputs["constraints"] = {"status": status, **meta}
    if not ok:
        _save_partial(case_dir, problem_index, final_result, detailed_result, module_outputs)
        return problem_index, {"status": "partial", "modules_processed": 5}

    # ---- Finalize ----
    completed_count = sum(
        1 for v in module_outputs.values()
        if v["status"] in ("success", "skipped")
    )

    if completed_count == 6 and validate_result_format(final_result):
        detailed_result["module_outputs"] = module_outputs
        with RESULT_LOCK:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            with open(os.path.join(case_dir, "detailed_result.json"), "w", encoding="utf-8") as f:
                json.dump(detailed_result, f, ensure_ascii=False, indent=2)
            with open(os.path.join(case_dir, "modules_summary.json"), "w", encoding="utf-8") as f:
                json.dump(module_outputs, f, ensure_ascii=False, indent=2)
        logger.info("  OK Problem %d completed (%d/6 modules)", problem_index, completed_count)
        return problem_index, {"status": "success", "modules_processed": completed_count}

    _save_partial(case_dir, problem_index, final_result, detailed_result, module_outputs)
    return problem_index, {
        "status": "partial",
        "modules_processed": completed_count,
    }


def _save_partial(
    case_dir: str,
    problem_index: int,
    final_result: dict,
    detailed_result: dict,
    module_outputs: dict,
):
    """Save partial results for a failed/incomplete problem."""
    error_dir = os.path.join(os.path.dirname(case_dir), f"{problem_index}_error")
    os.makedirs(error_dir, exist_ok=True)
    with open(os.path.join(error_dir, "partial_result.json"), "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(error_dir, "detailed_partial_result.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(error_dir, "modules_summary.json"), "w", encoding="utf-8") as f:
        json.dump(module_outputs, f, ensure_ascii=False, indent=2)
    logger.warning(
        "  WARN Problem %d partial (%d/6 modules)",
        problem_index,
        sum(1 for v in module_outputs.values() if v["status"] in ("success", "skipped")),
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _delete_failed_dirs(output_dir: str, problem_index: int):
    """Remove case and error directories for a failed problem."""
    import shutil
    for sub in (str(problem_index), f"{problem_index}_error"):
        path = os.path.join(output_dir, sub)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def process_dataset(
    dataset_key: str,
    model_name: str = DEFAULT_MODEL,
    start_from: int = 0,
    single: Optional[int] = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    output_dir: Optional[str] = None,
    dataset_type: Optional[str] = None,
) -> dict:
    """Run the extraction pipeline on an entire dataset.

    Args:
        dataset_key: Dataset key (nl4opt, nlp4lp, industryor, mamo_easy, mamo_complex).
        model_name: LLM model alias.
        start_from: Skip problems with index < start_from.
        single: Process only the problem with this index.
        max_parallel: Maximum concurrent LLM requests.
        output_dir: Custom output directory (default: output_<dataset_key>).
        dataset_type: Override dataset type for token config.

    Returns:
        Summary dict with success, partial, error, skipped counts.
    """
    if dataset_type is None:
        dataset_type = dataset_key
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")

    # Determine output directory
    if output_dir is None:
        output_dir = f"output_{dataset_key}"
    os.makedirs(output_dir, exist_ok=True)

    # Load problems
    problems = load_dataset(dataset_key, start_from=start_from, single=single)
    if not problems:
        logger.error("No problems found for dataset '%s'", dataset_key)
        return {"success": 0, "partial": 0, "error": 0, "skipped": 0}

    # Fail once with a clear configuration error instead of once per worker.
    needs_api = False
    for pid, _ in problems:
        result_path = os.path.join(output_dir, str(pid), "result.json")
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                if validate_result_format(json.load(f)):
                    continue
        except (OSError, json.JSONDecodeError):
            pass
        needs_api = True
        break
    if needs_api:
        LLMClient(model=model_name)

    logger.info(
        "Processing %d problems (max_parallel=%d, model=%s, dataset_type=%s)",
        len(problems), max_parallel, model_name, dataset_type,
    )

    # Pass 1: process in parallel
    stats = _run_one_pass(problems, output_dir, model_name, dataset_type, max_parallel)
    success, partial, error, skipped, failed_list = stats

    logger.info(
        "Pass 1 complete - success=%d partial=%d failed=%d skipped=%d",
        success, partial, error, skipped,
    )

    # Regeneration passes (up to 2)
    for regen_round in range(1, 3):
        if not failed_list:
            break
        logger.info(
            "Regeneration round %d: %d failed/partial problems to retry...",
            regen_round, len(failed_list),
        )
        for idx, _ in failed_list:
            _delete_failed_dirs(output_dir, idx)

        r_success, r_partial, r_error, r_skipped, failed_list = _run_one_pass(
            failed_list, output_dir, model_name, dataset_type, max_parallel,
        )
        success += r_success
        partial = r_partial
        error = r_error
        skipped += r_skipped

    summary = {
        "success": success,
        "partial": partial,
        "error": error,
        "skipped": skipped,
        "total": len(problems),
    }
    logger.info(
        "Final - success=%d partial=%d error=%d total=%d",
        success, partial, error, len(problems),
    )
    return summary


def _run_one_pass(
    problems: List[Tuple[int, str]],
    output_dir: str,
    model_name: str,
    dataset_type: str,
    max_parallel: int,
) -> Tuple[int, int, int, int, List[Tuple[int, str]]]:
    """Execute one pass over the problem list. Returns (success, partial, error, skipped, failed_list)."""
    idx_to_question = dict(problems)
    success, partial, error, skipped = 0, 0, 0, 0
    failed_list: List[Tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                process_single_problem, pid, q, output_dir, model_name, dataset_type,
            ): pid
            for pid, q in problems
        }
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                status = result.get("status", "error")
                if status == "success":
                    success += 1
                elif status == "partial":
                    partial += 1
                    failed_list.append((idx, idx_to_question.get(idx, "")))
                elif status == "skipped":
                    skipped += 1
                else:
                    error += 1
                    failed_list.append((idx, idx_to_question.get(idx, "")))
            except Exception as exc:
                error += 1
                pid = futures[future]
                failed_list.append((pid, idx_to_question.get(pid, "")))
                logger.error("Problem %d crashed: %s", pid, exc)

    return success, partial, error, skipped, failed_list


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the extraction pipeline on a dataset.")
    parser.add_argument(
        "dataset",
        choices=["nl4opt", "nlp4lp", "industryor", "mamo_easy", "mamo_complex"],
        help="Dataset to process",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model alias")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--single", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-type", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summary = process_dataset(
        args.dataset,
        model_name=args.model,
        start_from=args.start_from,
        single=args.single,
        max_parallel=args.max_parallel,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
    )
    print(f"\nExtraction complete: {summary}")
