"""
Dataset download and loading utilities.

Supported datasets:
    - NL4OPT:    Natural language optimization problem descriptions with solutions
    - NLP4LP:    Natural language processing for linear programming
    - IndustryOR: Industry operations research problems
    - MAMO_EasyLP: Easy linear programming problems from MAMO benchmark
    - MAMO_ComplexLP: Complex linear programming problems from MAMO benchmark

Data is cached locally under the ./datasets/ directory by default.
"""

import os
import json
import logging
import zipfile
import shutil
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default local cache directory
DATA_DIR = os.environ.get(
    "AST_CBOR_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"),
)

# Dataset metadata registry
DATASET_META: Dict[str, Dict[str, Any]] = {
    "nl4opt": {
        "name": "NL4OPT",
        "description": "Natural language optimization problem descriptions with optimal solutions",
        "files": ["NL4OPT_with_optimal_solution.json"],
        "format": "jsonl",
        "question_key": "en_question",
        "source_urls": [
            "https://huggingface.co/datasets/CardinalOperations/NL4OPT/resolve/main/NL4OPT_with_optimal_solution.json",
        ],
    },
    "nlp4lp": {
        "name": "NLP4LP",
        "description": "Natural language processing for linear programming",
        "files": ["train.jsonl"],
        "format": "nlp4lp",
        "question_key": "description",
        "source_urls": [
            "https://huggingface.co/datasets/udell-lab/NLP4LP/resolve/main/train.jsonl",
        ],
    },
    "industryor": {
        "name": "IndustryOR",
        "description": "Industry operations research benchmark",
        "files": ["IndustryOR.json"],
        "format": "jsonl",
        "question_key": "en_question",
        "source_urls": [
            "https://huggingface.co/datasets/CardinalOperations/IndustryOR/resolve/main/IndustryOR.json",
        ],
    },
    "mamo_easy": {
        "name": "MAMO_EasyLP",
        "description": "Easy LP problems from the MAMO benchmark",
        "files": ["MAMO_EasyLP.json"],
        "format": "jsonl",
        "question_key": "en_question",
        "source_urls": [
            "https://huggingface.co/datasets/CardinalOperations/MAMO/resolve/main/MAMO_EasyLP.json",
        ],
    },
    "mamo_complex": {
        "name": "MAMO_ComplexLP",
        "description": "Complex LP problems from the MAMO benchmark",
        "files": ["MAMO_ComplexLP.json"],
        "format": "jsonl",
        "question_key": "en_question",
        "source_urls": [
            "https://huggingface.co/datasets/CardinalOperations/MAMO/resolve/main/MAMO_ComplexLP.json",
        ],
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: str) -> bool:
    """Download a single file from url to dest. Returns True on success."""
    import urllib.request

    logger.info("Downloading: %s -> %s", url, dest)
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        headers = {"User-Agent": "ast-cbor-core/1.0"}
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        logger.info("Downloaded: %s (%d bytes)", dest, os.path.getsize(dest))
        return True
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        return False


def _extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract a zip file, moving contents up one level if nested."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            root = os.path.realpath(extract_dir)
            for member in zf.infolist():
                target = os.path.realpath(os.path.join(extract_dir, member.filename))
                if os.path.commonpath((root, target)) != root:
                    raise ValueError(f"Unsafe path in archive: {member.filename}")
            zf.extractall(extract_dir)
        os.remove(zip_path)
        # If extraction created a single subdirectory, flatten it
        items = os.listdir(extract_dir)
        if len(items) == 1:
            inner = os.path.join(extract_dir, items[0])
            if os.path.isdir(inner):
                for fname in os.listdir(inner):
                    shutil.move(os.path.join(inner, fname), extract_dir)
                os.rmdir(inner)
        return True
    except Exception as e:
        logger.error("Extraction failed for %s: %s", zip_path, e)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_path(dataset_key: str) -> str:
    """Return the local filesystem path for a dataset."""
    meta = DATASET_META.get(dataset_key)
    if not meta:
        raise KeyError(f"Unknown dataset: {dataset_key}. Valid: {list(DATASET_META.keys())}")
    return os.path.join(DATA_DIR, meta["name"])


def is_downloaded(dataset_key: str) -> bool:
    """Check if a dataset has been downloaded to the local cache."""
    path = get_dataset_path(dataset_key)
    if not os.path.isdir(path):
        return False

    meta = DATASET_META[dataset_key]
    fmt = meta.get("format", "jsonl")

    if fmt == "jsonl":
        for fname in meta.get("files", []):
            if not os.path.isfile(os.path.join(path, fname)):
                return False
        return len(meta.get("files", [])) > 0
    elif fmt == "nlp4lp":
        jsonl_path = os.path.join(path, "train.jsonl")
        if os.path.isfile(jsonl_path) and os.path.getsize(jsonl_path) > 0:
            return True
        return any("description.txt" in files for _, _, files in os.walk(path))
    return False


def download_dataset(dataset_key: str, force: bool = False) -> bool:
    """Download a single dataset to the local cache.

    Args:
        dataset_key: One of the keys in DATASET_META.
        force: Re-download even if already cached.

    Returns:
        True if the dataset is available locally after this call.
    """
    meta = DATASET_META.get(dataset_key)
    if not meta:
        logger.error("Unknown dataset key: %s", dataset_key)
        return False

    if not force and is_downloaded(dataset_key):
        logger.info("Dataset '%s' is already downloaded at %s", dataset_key, get_dataset_path(dataset_key))
        return True

    dest_dir = get_dataset_path(dataset_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    urls = meta.get("source_urls", [])
    downloaded_any = False

    for url in urls:
        filename = os.path.basename(url.split("?")[0])  # strip query params
        dest = os.path.join(dest_dir, filename)

        if _download_file(url, dest):
            downloaded_any = True
            # Handle zip files (e.g., NLP4LP GitHub archive)
            if filename.endswith(".zip"):
                _extract_zip(dest, dest_dir)
            break
        else:
            logger.warning("Failed URL %s, trying next...", url)

    if not downloaded_any:
        logger.error("All download URLs failed for dataset '%s'.", dataset_key)
        # Check if directory exists with any files as fallback
        if os.path.isdir(dest_dir) and os.listdir(dest_dir):
            if is_downloaded(dataset_key):
                logger.info("Using existing verified files in %s as fallback.", dest_dir)
                return True
            logger.error("Existing files in %s are incomplete or invalid.", dest_dir)
        return False

    if not is_downloaded(dataset_key):
        logger.error("Downloaded files do not match the expected layout for '%s'.", dataset_key)
        return False
    return True


def download_all(force: bool = False) -> Dict[str, bool]:
    """Download all known datasets. Returns {dataset_key: success} dict."""
    results = {}
    for key in DATASET_META:
        logger.info("--- Downloading %s ---", key)
        results[key] = download_dataset(key, force=force)
    return results


def list_datasets() -> List[str]:
    """List all available dataset keys."""
    return sorted(DATASET_META.keys())


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_key: str,
    start_from: int = 0,
    single: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """Load problems from a dataset.

    Args:
        dataset_key: Dataset key (nl4opt, nlp4lp, etc.).
        start_from: Skip problems with id < start_from.
        single: Load only the problem with this id.

    Returns:
        List of (problem_id, question_text) tuples.
    """
    meta = DATASET_META.get(dataset_key)
    if not meta:
        raise KeyError(f"Unknown dataset: {dataset_key}")

    fmt = meta.get("format", "jsonl")

    if fmt == "jsonl":
        return _load_jsonl(dataset_key, start_from, single)
    elif fmt == "nlp4lp":
        jsonl_path = os.path.join(get_dataset_path(dataset_key), "train.jsonl")
        if os.path.isfile(jsonl_path):
            return _load_jsonl(dataset_key, start_from, single)
        return _load_directory(dataset_key, start_from, single)
    raise ValueError(f"Unknown dataset format: {fmt}")


def _load_jsonl(
    dataset_key: str,
    start_from: int,
    single: Optional[int],
) -> List[Tuple[int, str]]:
    """Load problems from a JSONL file."""
    meta = DATASET_META[dataset_key]
    path = get_dataset_path(dataset_key)
    key = meta.get("question_key", "en_question")

    filename = meta.get("files", [""])[0]
    filepath = os.path.join(path, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}. "
            f"Run download_dataset('{dataset_key}') first."
        )

    records = read_json_records(filepath)
    logger.info("Found %d problems in %s", len(records), filepath)
    problems = []
    for i, data in enumerate(records):
        raw_pid = data.get("id", data.get("problem_id", i))
        pid = int(raw_pid) if str(raw_pid).isdigit() else i
        question = data.get(key, "")
        if not isinstance(question, str) or not question.strip():
            continue
        if single is not None:
            if i == single or pid == single or str(raw_pid) == str(single):
                problems.append((pid, question.strip()))
                break
        elif pid >= start_from:
            problems.append((pid, question.strip()))

    return problems


def read_json_records(filepath: str) -> List[Dict[str, Any]]:
    """Read either a JSON array or a JSON-lines file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {filepath}")
        return [item for item in data if isinstance(item, dict)]

    records = []
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping invalid JSON at %s:%d: %s", filepath, line_number, exc)
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _load_directory(
    dataset_key: str,
    start_from: int,
    single: Optional[int],
) -> List[Tuple[int, str]]:
    """Load problems from a directory structure (e.g., NLP4LP)."""
    path = get_dataset_path(dataset_key)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. "
            f"Run download_dataset('{dataset_key}') first."
        )

    def _numeric_prefix(s: str) -> int:
        """Extract leading integer from a directory name."""
        import re
        m = re.match(r"^(\d+)", s)
        return int(m.group(1)) if m else float("inf")

    description_paths = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if "description.txt" in files:
            description_paths.append(os.path.join(root, "description.txt"))
    description_paths.sort(
        key=lambda p: (_numeric_prefix(os.path.basename(os.path.dirname(p))), p)
    )
    logger.info("Found %d problem directories in %s", len(description_paths), path)

    problems = []
    used_ids = set()
    for fallback_id, desc_path in enumerate(description_paths):
        dirname = os.path.basename(os.path.dirname(desc_path))
        num = _numeric_prefix(dirname)
        if isinstance(num, float):  # no numeric prefix
            num = fallback_id
        while num in used_ids:
            num += 1
        used_ids.add(num)
        if single is not None:
            if num == single:
                with open(desc_path, "r", encoding="utf-8") as f:
                    problems.append((num, f.read().strip()))
                break
        elif num >= start_from:
            with open(desc_path, "r", encoding="utf-8") as f:
                problems.append((num, f.read().strip()))

    return problems
