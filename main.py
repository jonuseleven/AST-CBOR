#!/usr/bin/env python3
"""
AST_CBOR — Abstract Syntax Tree enhanced Case-Based Reasoning for
Optimization Modeling and Solving.

Unified CLI for the LLM-driven optimization modeling pipeline:
    extract   — Run the 6-step extraction pipeline
    generate  — Generate and execute Gurobi solver code
    run       — Run the full pipeline (extract + generate + ast)
    download  — Download datasets
    ast       — Run AST normalization and graph extraction
    retrieve  — Retrieve top-K similar cases via AST similarity
    evaluate  — Compare generated results against ground truth

Examples:
    python main.py download nl4opt
    python main.py extract nl4opt --model gpt-4o
    python main.py generate output_nl4opt --model claude-sonnet-5
    python main.py run nl4opt --model deepseek
    python main.py run --all --model gpt-4o
    python main.py ast output_nl4opt
    python main.py retrieve --root-dir output_nl4opt --case-ids "1,2,3" --query-file solver.py
    python main.py evaluate output_nl4opt nl4opt
"""

import argparse
import logging
import sys

from llm import DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_download(args):
    """Download one or all datasets."""
    from datasets import download_dataset, download_all, list_datasets

    if args.all:
        results = download_all(force=args.force)
        for key, ok in results.items():
            status = "OK" if ok else "FAILED"
            print(f"  {key}: {status}")
        if not all(results.values()):
            raise SystemExit(1)
    elif args.dataset:
        ok = download_dataset(args.dataset, force=args.force)
        print(f"{'Downloaded' if ok else 'Failed to download'}: {args.dataset}")
        if not ok:
            raise SystemExit(1)
    else:
        print("Available datasets:")
        for key in list_datasets():
            from datasets import is_downloaded, DATASET_META
            cached = " [cached]" if is_downloaded(key) else ""
            print(f"  {key}: {DATASET_META[key]['description']}{cached}")
        print("\nUsage: python main.py download <dataset>  or  --all")


def cmd_extract(args):
    """Run the extraction pipeline."""
    from extract import process_dataset

    summary = process_dataset(
        dataset_key=args.dataset,
        model_name=args.model,
        start_from=args.start_from,
        single=args.single,
        max_parallel=args.max_parallel,
        output_dir=args.output_dir,
        dataset_type=args.type,
    )
    print(f"\nExtraction complete: {summary}")
    if summary.get("partial", 0) or summary.get("error", 0):
        raise SystemExit(1)


def cmd_generate(args):
    """Generate and execute Gurobi solver code."""
    from generate import generate_for_dataset

    failed = 0
    for output_dir in args.output_dirs:
        print(f"\n--- Processing: {output_dir} ---")
        result = generate_for_dataset(
            output_dir=output_dir,
            model_name=args.model,
            start_from=args.start_from,
            max_parallel=args.max_parallel,
            skip=args.skip,
            case_library=args.case_library,
            cbr_topk=args.cbr_topk,
            solver_timeout=args.solver_timeout,
        )
        failed += result.get("failed", 0)
    print("\nGurobi generation complete.")
    if failed:
        raise SystemExit(1)


def cmd_run(args):
    """Run the full pipeline."""
    from pipeline import run_all_datasets, run_pipeline

    if args.all:
        datasets = None  # all
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = None

    if datasets is not None and len(datasets) == 1:
        results = {datasets[0]: run_pipeline(
            dataset_key=datasets[0],
            model=args.model,
            max_parallel_extract=args.max_parallel_extract,
            max_parallel_generate=args.max_parallel_generate,
            start_from=args.start_from,
            single=args.single,
            output_dir=args.output_dir,
            skip_extract=args.skip_extract,
            skip_generate=args.skip_generate,
            skip_ast=args.skip_ast,
            skip_evaluate=args.skip_evaluate,
            case_library=args.case_library,
            cbr_topk=args.cbr_topk,
            solver_timeout=args.solver_timeout,
        )}
    else:
        if args.start_from or args.single is not None or args.output_dir:
            raise SystemExit("--start-from, --single, and --output-dir require exactly one dataset")
        results = run_all_datasets(
            model=args.model,
            datasets=datasets,
            max_parallel_extract=args.max_parallel_extract,
            max_parallel_generate=args.max_parallel_generate,
            skip_extract=args.skip_extract,
            skip_generate=args.skip_generate,
            skip_ast=args.skip_ast,
            skip_evaluate=args.skip_evaluate,
            case_library=args.case_library,
            cbr_topk=args.cbr_topk,
            solver_timeout=args.solver_timeout,
        )
    print("\nFull pipeline complete.")
    for result in results.values():
        extraction = result.get("extraction") or {}
        generation = result.get("generation") or {}
        evaluation = result.get("evaluation") or {}
        if extraction.get("partial", 0) or extraction.get("error", 0):
            raise SystemExit(1)
        if generation.get("failed", 0) or "error" in evaluation:
            raise SystemExit(1)


def cmd_ast(args):
    """Run AST analysis."""
    from ast_cbor import batch_process
    import os as _os

    target = args.dataset
    if not _os.path.isdir(target):
        candidate = _os.path.join(_os.getcwd(), f"output_{target}")
        if _os.path.isdir(candidate):
            target = candidate
        else:
            print(f"Error: directory not found: {target} (also tried {candidate})")
            sys.exit(1)

    result = batch_process(target, args.filename)
    if result.get("failed", 0):
        raise SystemExit(1)


def cmd_retrieve(args):
    """Retrieve top-K similar cases."""
    from ast_cbor import retrieve_topk, parse_case_ids

    case_ids = parse_case_ids(args.case_ids)
    result = retrieve_topk(
        root_dir=args.root_dir,
        case_ids=case_ids,
        query_code_file=args.query_file,
        topk=args.topk,
    )

    # Save result
    import json as _json
    import os as _os

    if args.save_json:
        save_path = args.save_json
    else:
        query_dir = _os.path.dirname(_os.path.abspath(args.query_file))
        query_name = _os.path.splitext(_os.path.basename(args.query_file))[0]
        save_path = _os.path.join(query_dir, f"{query_name}.top{args.topk}.retrieval.json")

    with open(save_path, "w", encoding="utf-8") as f:
        _json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Retrieval result saved to: {save_path}")
    print("\nTop results:")
    for rank, item in enumerate(result.get("results", []), 1):
        print(f"  {rank:>2}. case_id={item['case_id']}, similarity={item['similarity']:.6f}")


def cmd_evaluate(args):
    """Evaluate results against ground truth."""
    from pipeline import evaluate_dataset

    result = evaluate_dataset(args.output_dir, args.dataset)
    if "error" in result:
        print(f"Error: {result['error']}")
        raise SystemExit(1)

    print(f"\nEvaluation: {args.dataset}")
    print(f"  Correct: {result['correct']}/{result['total']}")
    print(f"  Valid solver outcomes: {result['valid']}")
    print(f"  Failed solver outcomes: {result['failed']}")
    print(f"  Accuracy: {result['accuracy_percent']}%")
    print()


def cmd_experiment(args):
    """Run an isolated baseline versus AST-CBOR comparison."""
    from pipeline import run_experiment

    summary = run_experiment(
        extracted_dir=args.extracted_dir,
        dataset_key=args.dataset,
        model=args.model,
        case_library=args.case_library,
        experiment_dir=args.experiment_dir,
        max_parallel_generate=args.max_parallel_generate,
        cbr_topk=args.cbr_topk,
        solver_timeout=args.solver_timeout,
    )
    baseline = summary["baseline"]["evaluation"]
    cbr = summary["ast_cbor"]["evaluation"]
    print(f"\nExperiment complete: {args.dataset}")
    print(f"  Baseline accuracy: {baseline.get('accuracy_percent', 0)}%")
    print(f"  AST-CBOR accuracy: {cbr.get('accuracy_percent', 0)}%")


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AST_CBOR - Optimization Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download nl4opt
  python main.py extract nl4opt --model gpt-4o
  python main.py generate output_nl4opt --model claude-sonnet-5
  python main.py run nl4opt --model deepseek
  python main.py run --all --model gpt-4o
  python main.py ast output_nl4opt
  python main.py retrieve --root-dir output_nl4opt --case-ids "1,2,3" --query-file solver.py
  python main.py evaluate output_nl4opt nl4opt
""",
    )
    sub = parser.add_subparsers(dest="command", help="Available subcommands")
    sub.required = True

    # ---- download ----
    p_dl = sub.add_parser("download", help="Download datasets")
    p_dl.add_argument("dataset", nargs="?", default=None,
                      help="Dataset name or leave empty to list available")
    p_dl.add_argument("--all", action="store_true", default=False,
                      help="Download all datasets")
    p_dl.add_argument("--force", action="store_true", default=False,
                      help="Re-download even if cached")
    p_dl.set_defaults(func=cmd_download)

    # ---- extract ----
    p_ext = sub.add_parser("extract", help="Run the 6-step extraction pipeline")
    p_ext.add_argument("dataset", choices=[
        "nl4opt", "nlp4lp", "industryor", "mamo_easy", "mamo_complex",
    ])
    p_ext.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"LLM model (default: {DEFAULT_MODEL})")
    p_ext.add_argument("--type", default=None,
                       help="Dataset type override for token config")
    p_ext.add_argument("--start-from", type=int, default=0)
    p_ext.add_argument("--single", type=int, default=None)
    p_ext.add_argument("--max-parallel", type=int, default=50)
    p_ext.add_argument("--output-dir", default=None)
    p_ext.set_defaults(func=cmd_extract)

    # ---- generate ----
    p_gen = sub.add_parser("generate", help="Generate and execute Gurobi solver code")
    p_gen.add_argument("output_dirs", nargs="+", help="Output directories to process")
    p_gen.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"LLM model (default: {DEFAULT_MODEL})")
    p_gen.add_argument("--start-from", type=int, default=0)
    p_gen.add_argument("--max-parallel", type=int, default=30)
    p_gen.add_argument("--skip", action="store_true", default=True,
                       help="Skip already-processed problems")
    p_gen.add_argument("--no-skip", dest="skip", action="store_false",
                       help="Reprocess all problems")
    p_gen.add_argument("--case-library", default=None,
                       help="Separate solved-case directory; enables AST-CBOR refinement")
    p_gen.add_argument("--cbr-topk", type=int, default=3)
    p_gen.add_argument("--solver-timeout", type=int, default=60)
    p_gen.set_defaults(func=cmd_generate)

    # ---- run ----
    p_run = sub.add_parser("run", help="Run the full pipeline on dataset(s)")
    p_run.add_argument("datasets", nargs="?", default=None,
                       help="Comma-separated dataset names (default: all)")
    p_run.add_argument("--all", action="store_true", default=False,
                       help="Run on all datasets")
    p_run.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"LLM model (default: {DEFAULT_MODEL})")
    p_run.add_argument("--max-parallel-extract", type=int, default=50)
    p_run.add_argument("--max-parallel-generate", type=int, default=30)
    p_run.add_argument("--skip-extract", action="store_true", default=False)
    p_run.add_argument("--skip-generate", action="store_true", default=False)
    p_run.add_argument("--skip-ast", action="store_true", default=False)
    p_run.add_argument("--skip-evaluate", action="store_true", default=False)
    p_run.add_argument("--case-library", default=None,
                       help="Separate solved-case directory; enables AST-CBOR refinement")
    p_run.add_argument("--cbr-topk", type=int, default=3)
    p_run.add_argument("--solver-timeout", type=int, default=60)
    p_run.add_argument("--start-from", type=int, default=0)
    p_run.add_argument("--single", type=int, default=None)
    p_run.add_argument("--output-dir", default=None)
    p_run.set_defaults(func=cmd_run)

    # ---- ast ----
    p_ast = sub.add_parser("ast", help="Run AST normalization and graph extraction")
    p_ast.add_argument("dataset", help="Dataset path or name")
    p_ast.add_argument("--filename", default="gurobi_solver.py",
                       help="Target filename (default: gurobi_solver.py)")
    p_ast.set_defaults(func=cmd_ast)

    # ---- retrieve ----
    p_ret = sub.add_parser("retrieve", help="Retrieve top-K similar cases")
    p_ret.add_argument("--root-dir", required=True, help="Case library root directory")
    p_ret.add_argument("--case-ids", required=True,
                       help='Case IDs, e.g. "1,2,3,8"')
    p_ret.add_argument("--query-file", required=True, help="Query code file path")
    p_ret.add_argument("--topk", type=int, default=5)
    p_ret.add_argument("--save-json", default=None, help="Output JSON path")
    p_ret.set_defaults(func=cmd_retrieve)

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="Compare results against ground truth")
    p_eval.add_argument("output_dir", help="Directory containing generated results")
    p_eval.add_argument("dataset", choices=[
        "nl4opt", "nlp4lp", "industryor", "mamo_easy", "mamo_complex",
    ], help="Dataset name for ground truth")
    p_eval.set_defaults(func=cmd_evaluate)

    # ---- experiment ----
    p_exp = sub.add_parser("experiment", help="Compare baseline and AST-CBOR generation")
    p_exp.add_argument("extracted_dir", help="Directory containing completed extraction results")
    p_exp.add_argument("dataset", choices=[
        "nl4opt", "nlp4lp", "industryor", "mamo_easy", "mamo_complex",
    ])
    p_exp.add_argument("--case-library", required=True,
                       help="Separate training case library with solved Gurobi scripts")
    p_exp.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"LLM model (default: {DEFAULT_MODEL})")
    p_exp.add_argument("--experiment-dir", default=None)
    p_exp.add_argument("--max-parallel-generate", type=int, default=10)
    p_exp.add_argument("--cbr-topk", type=int, default=3)
    p_exp.add_argument("--solver-timeout", type=int, default=60)
    p_exp.set_defaults(func=cmd_experiment)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
