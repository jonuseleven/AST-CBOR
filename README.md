# AST_CBOR — AST-Enhanced Case-Based Reasoning for Optimization

An LLM-driven pipeline that transforms natural language optimization problem descriptions into executable Gurobi solver code, using AST-based structural analysis and case-based reasoning to improve generation quality.

## Overview

Given a natural language description of an optimization problem (e.g., *"A factory produces chairs and tables..."*), AST_CBOR:

1. **Extracts** a structured representation via a 6-step LLM pipeline: normalization → entities → attributes → objectives → decision variables → constraints
2. **Generates** executable Gurobi Python solver code through a 3-step cascade: mathematical model → core code → full solver script
3. **Executes** the solver and records objective values
4. **Analyzes** generated code via AST normalization and multi-dimensional graph similarity
5. **Retrieves** similar solved cases via AST structure and uses them to refine generated solver code

## Supported Models

| Provider  | Models                                          | Alias Examples                     |
|-----------|-------------------------------------------------|------------------------------------|
| OpenAI    | GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4        | `gpt-4o`, `gpt-4-turbo`           |
| Anthropic | Claude Sonnet 5, Opus 5, Sonnet 4, Haiku 4.5   | `claude-sonnet-5`, `claude-opus-5`|
| DeepSeek  | DeepSeek V4 Flash and V4 Pro                    | `deepseek`, `deepseek-v4-pro`      |

Use the alias (e.g., `--model gpt-4o`) or the raw `provider:model_id` format (e.g., `--model openai:gpt-4o`).

## Supported Datasets

| Dataset         | Description                                            |
|-----------------|--------------------------------------------------------|
| NL4OPT          | Natural language optimization problems with solutions  |
| NLP4LP          | Natural language descriptions for linear programming   |
| IndustryOR      | Industry operations research benchmark                 |
| MAMO_EasyLP     | Easy linear programming problems (MAMO benchmark)      |
| MAMO_ComplexLP  | Complex linear programming problems (MAMO benchmark)   |

## Installation

```bash
# Clone and enter the project directory
cd ast_cbor_core

# Create an isolated environment and install the pinned direct dependencies
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

# Set up API keys (copy .env.example to .env and edit)
cp .env.example .env
```

**Requirements:**
- Python 3.10+
- Gurobi license (for solver execution; `gurobipy` is required)
- At least one API key (OpenAI, Anthropic, or DeepSeek)

## Quick Start

```bash
# 1. Download a dataset
python main.py download nl4opt

# 2. Run extraction (LLM pipeline)
python main.py extract nl4opt --model gpt-4o

# 3. Generate and execute Gurobi solvers
python main.py generate output_nl4opt --model gpt-4o

# 4. Run AST analysis on generated code
python main.py ast output_nl4opt

# 5. Evaluate accuracy against ground truth
python main.py evaluate output_nl4opt nl4opt

# 6. Enable AST-CBOR refinement with a separate solved training case library
python main.py generate output_nl4opt --case-library output_training_cases

# Or run everything in one command:
python main.py run nl4opt --model gpt-4o

# Run on all datasets:
python main.py run --all --model claude-sonnet-5

# Controlled baseline versus AST-CBOR comparison on the same extracted cases
python main.py experiment output_nl4opt nl4opt --case-library output_training_cases --model gpt-4o-mini
```

## CLI Reference

```
python main.py <command> [options]
```

### Commands

| Command       | Description                                            |
|---------------|--------------------------------------------------------|
| `download`    | Download datasets to local cache (`./datasets/`)      |
| `extract`     | Run 6-step extraction pipeline on a dataset           |
| `generate`    | Generate Gurobi solver code from extraction results    |
| `run`         | Run full pipeline (extract → generate → AST → evaluate)|
| `ast`         | Run AST normalization and graph extraction             |
| `retrieve`    | Retrieve top-K similar cases via AST similarity        |
| `evaluate`    | Compare generated objective values against ground truth|
| `experiment`  | Run isolated baseline and AST-CBOR generation arms     |

### Common Options

| Option              | Description                              |
|---------------------|------------------------------------------|
| `--model`           | LLM alias (auto-selected from configured keys) |
| `--max-parallel`    | Max parallel API calls (default: 50)     |
| `--start-from`      | Start processing from this problem index |
| `--single`          | Process only one specific problem        |
| `--case-library`    | Separate solved training cases; enables AST-CBOR |

## Project Structure

```
ast_cbor_core/
├── main.py           # CLI entry point
├── llm.py            # Multi-model LLM client (OpenAI, Anthropic, DeepSeek)
├── extract.py        # 6-step extraction pipeline
├── generate.py       # Gurobi solver code generation and execution
├── ast_cbor.py       # AST normalization, graph construction, similarity retrieval
├── pipeline.py       # Full pipeline orchestration and evaluation
├── prompts.py        # All prompt templates (inline)
├── datasets.py       # Dataset download and loading
├── requirements.txt  # Python dependencies
└── .env.example      # API key template
```

## Pipeline Details

### Extraction (6 steps)

Each step calls an LLM with a specialized prompt and feeds output to downstream steps:

1. **Normalization** — Rewrite the problem in structured, unambiguous language
2. **Entities** — Identify physical objects and decision-variable entities
3. **Attributes** — Extract essential attributes of each entity (type, value range, units)
4. **Objectives** — Identify the single optimization objective with direction
5. **Decision Variables** — Determine variables to be solved, with types (integer/continuous/binary)
6. **Constraints** — List all constraints with natural language explanations and mathematical formulas

### Gurobi Generation (3 steps)

1. **JSON → Math Model** — Convert structured JSON into a formal mathematical formulation
2. **Math Model → Core Code** — Generate Gurobi Python code (variables, objective, constraints)
3. **Core Code → Full Solver** — Wrap in a complete, executable Python script

### AST Analysis

1. **Normalization** — Rename all user identifiers to canonical forms (`OBJ_0`, `VAR_1`, etc.)
2. **Graph Construction** — Build a typed node + edge graph from the AST
3. **Graph Enhancement** — Add sibling, control-flow, data-dependency, and API-semantic edges
4. **Similarity** — Multi-dimensional cosine similarity over token, node type, edge type, API, and pattern features

### AST-CBOR Refinement

When `--case-library` is supplied, generation uses a separate solved training
case library. It first generates an initial solver, normalizes its AST, retrieves
the top-K structurally similar cases, and performs one constrained refinement
call. Retrieved code is treated only as a structural reference; the target JSON
remains authoritative. The target output directory cannot also be the case
library, which prevents direct test-case leakage.

The `experiment` command creates isolated `baseline/` and `ast_cbor/` arms from
the same extraction artifacts, runs both, evaluates both, and writes
`experiment_summary.json`.

## Configuration

API keys are configured via environment variables or a `.env` file:

| Variable            | Provider    |
|---------------------|-------------|
| `OPENAI_API_KEY`    | OpenAI      |
| `ANTHROPIC_API_KEY` | Anthropic   |
| `DEEPSEEK_API_KEY`  | DeepSeek    |

`.env` is loaded automatically from this project directory. If `AST_CBOR_MODEL`
is not set, the code selects the first provider with a usable key in this order:
DeepSeek, OpenAI, Anthropic. Missing or placeholder keys fail before worker
threads or paid API calls start.

NLP4LP is gated on Hugging Face. Accept the dataset terms and set `HF_TOKEN`
before running `python main.py download nlp4lp`.

Optional: `AST_CBOR_DATA_DIR` overrides the default dataset cache location (`./datasets/`).

## Output Structure

After running the pipeline, each output directory contains:

```
output_nl4opt/
├── 0/
│   ├── 00_normalization/   # Normalized problem description
│   ├── 01_entities/        # Extracted entities
│   ├── 02_attributes/      # Extracted attributes
│   ├── 03_objectives/      # Optimization objective
│   ├── 04_decision_variables/
│   ├── 05_constraints/     # Constraints with explanations
│   ├── result.json         # Condensed extraction result
│   ├── detailed_result.json # Full extraction details
│   ├── gurobi_solver.py    # Generated solver code
│   ├── gurobi_solver.normalized.py  # AST-normalized code
│   ├── gurobi_solver.ast.json       # AST graph data
│   ├── objective_value.txt # Solved objective value
│   └── objective_result.txt # Full solver output
├── 1/
│   └── ...
└── ...
```

Each generated case also contains:

- `solver_result.json`: structured solver status, objective, return code, error
- `generation_metadata.json`: model, token usage, and CBR configuration
- `retrieval.json`: retrieved case IDs and similarity scores (CBR runs only)
- `gurobi_solver.initial.py`: pre-refinement solver (CBR runs only)

Only `OPTIMAL`, `INFEASIBLE`, `UNBOUNDED`, and `INF_OR_UNBD` are valid terminal
outcomes. Timeouts, generation errors, and execution errors are counted as
failed cases and can never match a ground-truth no-solution label.

## Verification

Run the local regression suite without making API calls:

```bash
python -m unittest discover -s tests -v
```

## License

This project is provided for research purposes. See the original datasets for their respective licenses.
