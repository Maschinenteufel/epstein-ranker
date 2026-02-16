#!/usr/bin/env python3
"""Rank Epstein file rows by querying a local GPT server."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import requests


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


DEFAULT_SYSTEM_PROMPT = (
    "You analyze primary documents related to court and investigative filings.\n"
    "Focus on whether the passage offers potential leads—even if unverified—that\n"
    "connect influential actors (presidents, cabinet officials, foreign leaders,\n"
    "billionaires, intelligence agencies) to controversial actions, financial flows, or\n"
    "possible misconduct.\n"
    "Score each passage on:\n"
    "  1. Investigative usefulness: Does it suggest concrete follow-up steps, names,\n"
    "     transactions, dates, or relationships worth pursuing?\n"
    "  2. Controversy / sensitivity: Would the lead cause public outcry or legal risk if\n"
    "     validated?\n"
    "  3. Novelty: Is this information new or rarely reported, versus already known?\n"
    "  4. Power linkage: Does it implicate high-ranking officials or major power\n"
    "     centers? Leads tying unknown individuals only to minor issues should score\n"
    "     lower.\n"
    "Assign an importance_score from 0 (no meaningful lead) to 100 (blockbuster lead\n"
"linking powerful actors to fresh controversy). Use the full range—avoid clustering\n"
"around a few favorite numbers so similar passages can still be differentiated.\n"
"Use the scale consistently:\n"
"  • 0–10  : noise, duplicates, previously published facts, or gossip with no actors.\n"
"  • 10–30 : low-value context; speculative or weak leads lacking specifics.\n"
"  • 30–50 : moderate leads with partial details or missing novelty.\n"
"  • 50–70 : strong leads with actionable info or notable controversy.\n"
"  • 70–85 : high-impact, new revelations tying powerful actors to clear misconduct.\n"
"  • 85–100: blockbuster revelations demanding immediate follow-up.\n"
"Reserve 70+ for claims that, if true, would represent major revelations or\n"
"next-step investigations.\n"
    "Return strict JSON with the following fields:\n"
    "  - headline (string)\n"
    "  - importance_score (0-100 number)\n"
    "  - reason (string explaining score)\n"
    "  - key_insights (array of short bullet strings)\n"
    "  - tags (array of topical strings)\n"
    "  - power_mentions (array listing high-profile people or institutions mentioned;\n"
    "    include titles or roles if possible)\n"
    "  - agency_involvement (array naming government, intelligence, or law-enforcement\n"
    "    bodies involved or implicated)\n"
    "  - lead_types (array describing lead categories such as 'financial flow',\n"
    "    'legal exposure', 'foreign influence', 'sexual misconduct', etc.)\n"
    "If a category has no data, return an empty array for it."
)

LEAD_TYPE_CANONICAL_MAP = {
    "financial flow": {
        "financial flow",
        "financial flows",
        "financial transaction",
        "money trail",
        "money movement",
        "money laundering",
        "overpriced sale",
        "payment chain",
    },
    "foreign influence": {
        "foreign influence",
        "foreign interference",
        "foreign collusion",
    },
    "legal exposure": {"legal exposure", "legal risk", "criminal liability"},
    "political corruption": {"political corruption", "corruption", "quid pro quo"},
    "sexual misconduct": {"sexual misconduct", "sexual abuse", "sex trafficking"},
    "intelligence operation": {
        "intelligence operation",
        "intelligence activity",
        "spycraft",
        "espionage",
    },
    "national security": {"national security", "security breach"},
    "human trafficking": {
        "human trafficking",
        "trafficking",
        "exploitation",
    },
    "cover-up": {"cover-up", "cover up", "obstruction"},
    "financial fraud": {"financial fraud", "fraud"},
}

AGENCY_CANONICAL_MAP = {
    "NSA": {"nsa", "national security agency"},
    "CIA": {"cia", "central intelligence agency"},
    "FBI": {"fbi", "federal bureau of investigation"},
    "DOJ": {"doj", "department of justice", "u.s. department of justice"},
    "DHS": {"dhs", "department of homeland security"},
    "ODNI": {"odni", "office of the director of national intelligence"},
    "State Department": {"state department", "u.s. department of state", "dos"},
    "Treasury": {"treasury", "u.s. treasury", "department of the treasury"},
    "IRS": {"irs", "internal revenue service"},
    "SEC": {"sec", "securities and exchange commission"},
    "House Oversight Committee": {
        "house oversight committee",
        "house oversight",
        "house committee on oversight",
    },
    "Senate Judiciary Committee": {"senate judiciary committee", "senate judiciary"},
    "Congress": {"congress", "u.s. congress"},
    "FSB": {"fsb", "federal security service"},
    "GRU": {"gru", "main directorate"},
    "GCHQ": {"gchq", "government communications headquarters"},
    "MI6": {"mi6", "secret intelligence service", "sis"},
    "MI5": {"mi5"},
    "Mossad": {"mossad"},
    "Interpol": {"interpol"},
    "NYPD": {"nypd", "new york police department"},
}

DEFAULT_JUSTICE_FILES_BASE_URL = "https://www.justice.gov/epstein/files"


def explicit_cli_destinations(argv: List[str]) -> Set[str]:
    explicit: Set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        flag = token[2:]
        if "=" in flag:
            flag = flag.split("=", 1)[0]
        if flag.startswith("no-"):
            flag = flag[3:]
        explicit.add(flag.replace("-", "_"))
    return explicit


def sanitize_dataset_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "dataset"


def infer_dataset_tag(input_path: Path) -> str:
    if input_path.name:
        stem = input_path.stem if input_path.is_file() else input_path.name
        if stem:
            return sanitize_dataset_tag(stem)
    return "dataset"


def apply_dataset_workspace_defaults(
    args: argparse.Namespace,
    *,
    cli_explicit: Optional[Set[str]] = None,
) -> None:
    explicit = cli_explicit or set()
    if not args.dataset_workspace_root:
        return

    workspace_root = Path(args.dataset_workspace_root)
    tag = args.dataset_tag or infer_dataset_tag(args.input)
    tag = sanitize_dataset_tag(tag)
    args.dataset_tag = tag
    base = workspace_root / tag

    workspace_defaults = {
        "output": base / "results" / "epstein_ranked.csv",
        "json_output": base / "results" / "epstein_ranked.jsonl",
        "checkpoint": base / "state" / ".epstein_checkpoint",
        "chunk_dir": base / "chunks",
        "chunk_manifest": base / "metadata" / "chunks.json",
        "run_metadata_file": base / "metadata" / "run_metadata.json",
    }
    for key, value in workspace_defaults.items():
        if key not in explicit:
            setattr(args, key, value)

    # Prevent accidental overlap with other corpora when using isolated workspaces.
    if "known_json" not in explicit:
        args.known_json = []


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    cli_explicit: Optional[Set[str]] = None,
) -> None:
    explicit = cli_explicit or set()
    config_path: Path = args.config  # type: ignore[assignment]
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    for key, value in data.items():
        if not hasattr(args, key):
            continue
        if key in explicit:
            continue
        current = getattr(args, key)
        default = parser.get_default(key)
        if current == default:
            if isinstance(default, Path):
                setattr(args, key, Path(value))
            else:
                setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call a local OpenAI-compatible server (e.g. qwen/qwen3-coder-next) to extract "
            "useful information and rank each source row."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional TOML config file to supply defaults (see ranker_config.example.toml).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "EPS_FILES_20K_NOV2026.csv",
        help="Path to the source CSV (filename/text columns) or a directory of .txt files.",
    )
    parser.add_argument(
        "--input-glob",
        default="*.txt",
        help="Glob used when --input points to a directory (searched recursively).",
    )
    parser.add_argument(
        "--dataset-workspace-root",
        type=Path,
        default=None,
        help=(
            "If set, isolate outputs/checkpoint/chunks under "
            "<dataset-workspace-root>/<dataset-tag>/ to avoid mixing corpora."
        ),
    )
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help="Dataset identifier used with --dataset-workspace-root (defaults to input name).",
    )
    parser.add_argument(
        "--dataset-source-label",
        default=None,
        help="Optional human-readable source label for provenance metadata.",
    )
    parser.add_argument(
        "--dataset-source-url",
        default=None,
        help="Optional source URL for provenance metadata.",
    )
    parser.add_argument(
        "--dataset-metadata-file",
        type=Path,
        default=None,
        help="Optional JSON file containing dataset provenance/stats to attach to run metadata.",
    )
    parser.add_argument(
        "--run-metadata-file",
        type=Path,
        default=None,
        help="Optional JSON sidecar path for run metadata (auto-set in workspace mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "epstein_ranked.csv",
        help="Path to write the ranked CSV results.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data") / "epstein_ranked.jsonl",
        help="Path to append newline-delimited JSON records for each row.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5002/v1",
        help="Base URL of the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-coder-next",
        help="Model identifier exposed by the server (check via --list-models).",
    )
    parser.add_argument(
        "--justice-files-base-url",
        default=DEFAULT_JUSTICE_FILES_BASE_URL,
        help="Base URL for DOJ Epstein PDF files used to derive source document links.",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=4,
        help="Maximum concurrent model requests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model responses (0.0 = deterministic, higher = more random).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to a text file containing the system prompt (overrides default).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Inline system prompt string (overrides --prompt-file and default).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for servers that require authentication.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="If provided, passes reasoning effort hints supported by some models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the checkpoint or JSON output.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow truncating existing output/JSONL files (use with caution).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data") / ".epstein_checkpoint",
        help="Plain-text file storing processed filenames (used with --resume).",
    )
    parser.add_argument(
        "--known-json",
        action="append",
        default=[],
        help="Additional JSONL files containing already-processed rows to skip.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="If >0, rotate outputs every N source rows and store chunk files in --chunk-dir.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("contrib"),
        help="Directory to store chunked outputs when --chunk-size > 0.",
    )
    parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=Path("data") / "chunks.json",
        help="Manifest JSON file listing generated chunks (used by the viewer).",
    )
    parser.add_argument(
        "--include-action-items",
        action="store_true",
        help="Request action items from the model and include them in outputs.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based row index to start processing (useful for collaborative chunking).",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="1-based row index to stop processing (inclusive).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit processing to the first N rows (useful for smoke-tests).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests to avoid overwhelming the server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP request timeout in seconds (default: 600 = 10 minutes).",
    )
    parser.add_argument(
        "--skip-low-quality",
        dest="skip_low_quality",
        action="store_true",
        default=True,
        help="Skip low-signal rows (empty/too-short/noisy) before model calls.",
    )
    parser.add_argument(
        "--no-skip-low-quality",
        dest="skip_low_quality",
        action="store_false",
        help="Disable low-quality row skipping.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=60,
        help="Minimum non-whitespace character count required to process a row.",
    )
    parser.add_argument(
        "--min-text-words",
        type=int,
        default=12,
        help="Minimum token count required to process a row.",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.25,
        help="Minimum alphabetic-character ratio required to process a row.",
    )
    parser.add_argument(
        "--min-unique-word-ratio",
        type=float,
        default=0.15,
        help="Minimum unique-word ratio required to process a row.",
    )
    parser.add_argument(
        "--max-repeated-char-run",
        type=int,
        default=40,
        help="Maximum repeated-character run before a row is considered noisy.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models exposed by the endpoint and exit.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Scan chunk files and rebuild the manifest (data/chunks.json), then exit.",
    )
    parser.add_argument(
        "--power-watts",
        type=float,
        default=None,
        help="If provided, estimate energy usage (average watts).",
    )
    parser.add_argument(
        "--electric-rate",
        type=float,
        default=None,
        help="Electricity cost in USD per kWh for cost estimation.",
    )
    parser.add_argument(
        "--run-hours",
        type=float,
        default=None,
        help="Override elapsed hours for cost estimate (otherwise uses wall time).",
    )
    args = parser.parse_args()
    cli_explicit = explicit_cli_destinations(sys.argv[1:])
    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        for candidate in (Path("ranker_config.toml"), Path("ranker_config.example.toml")):
            if candidate.exists():
                config_path = candidate
                break
    if config_path:
        args.config = config_path
        apply_config_defaults(parser, args, cli_explicit=cli_explicit)
    apply_dataset_workspace_defaults(args, cli_explicit=cli_explicit)
    return args


def load_system_prompt(args: argparse.Namespace) -> Tuple[str, str]:
    """Load the system prompt from file or use inline/default prompt.

    Returns:
        Tuple of (prompt_text, prompt_source_description)
    """
    # Priority: inline --system-prompt > --prompt-file > default file > hardcoded default
    if args.system_prompt:
        return args.system_prompt, "inline (--system-prompt)"

    prompt_file = args.prompt_file
    if not prompt_file:
        # Try default prompt file location
        default_prompt_file = Path("prompts") / "default_system_prompt.txt"
        if default_prompt_file.exists():
            prompt_file = default_prompt_file
        else:
            # Fall back to hardcoded default
            return DEFAULT_SYSTEM_PROMPT, "default (hardcoded)"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with prompt_file.open("r", encoding="utf-8") as f:
        prompt = f.read().strip()

    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")

    return prompt, str(prompt_file)


def call_model(
    *,
    endpoint: str,
    model: str,
    filename: str,
    text: str,
    system_prompt: str,
    api_key: Optional[str],
    timeout: float,
    temperature: float,
    reasoning_effort: Optional[str],
    config_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send the document to the local GPT server and return parsed JSON."""
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Analyze the following document and respond with the JSON schema "
                    "described in the system prompt.\n"
                    f"Filename: {filename}\n"
                    "---------\n"
                    f"{text.strip()}\n"
                    "---------"
                ),
            },
        ],
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    # Include config metadata in the request if provided
    if config_metadata:
        payload["metadata"] = config_metadata

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{endpoint.rstrip('/')}/chat/completions"
    response = requests.post(url, json=payload, timeout=timeout, headers=headers)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # noqa: PERF203
        snippet = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {snippet}") from exc
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format: {data}") from exc

    return ensure_json_dict(content)


def ensure_json_dict(content: str) -> Dict[str, Any]:
    """Parse the model response into a dictionary, trimming extra text if needed."""
    candidate = content.strip()
    if not candidate:
        raise ValueError("Model returned an empty message.")

    # Try direct JSON parse first.
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        parsed = json.loads(candidate[start : end + 1])

    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a JSON object, received: {type(parsed)}")
    return parsed


def iter_rows(path: Path, *, input_glob: str = "*.txt", include_text: bool = True) -> Iterable[Dict[str, str]]:
    if path.is_dir():
        for file_path in sorted(path.rglob(input_glob)):
            if not file_path.is_file():
                continue
            row: Dict[str, str] = {
                "filename": file_path.relative_to(path).as_posix(),
                "text": "",
            }
            if include_text:
                row["text"] = file_path.read_text(encoding="utf-8", errors="replace")
            yield row
        return

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "text" not in row or "filename" not in row:
                raise ValueError("Input CSV must contain 'filename' and 'text' columns.")
            if include_text:
                yield row
            else:
                yield {"filename": row["filename"], "text": ""}


def load_checkpoint(path: Optional[Path]) -> Set[str]:
    completed: Set[str] = set()
    if not path or not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if entry:
                completed.add(entry)
    return completed


def load_jsonl_filenames(path: Optional[Path]) -> Set[str]:
    completed: Set[str] = set()
    if not path or not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = record.get("filename")
            if filename:
                completed.add(filename)
    return completed


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value)]


def canonicalize_from_map(value: str, mapping: Dict[str, Set[str]], *, title_case: bool = False, upper_case: bool = False) -> Optional[str]:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    lookup = cleaned.lower()
    for canonical, synonyms in mapping.items():
        if lookup in synonyms:
            return canonical if not upper_case else canonical.upper()
    if upper_case:
        return cleaned.upper()
    if title_case:
        return cleaned.title()
    return cleaned


def normalize_lead_types(values: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        canonical = canonicalize_from_map(value, LEAD_TYPE_CANONICAL_MAP, title_case=False)
        if not canonical:
            continue
        canonical = canonical.lower()
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def normalize_agencies(values: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        canonical = canonicalize_from_map(value, AGENCY_CANONICAL_MAP, upper_case=False)
        if not canonical:
            continue
        canonical = canonical.strip()
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def clean_entity_label(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s*\(.*?\)\s*", " ", cleaned)
    for delimiter in (" - ", " – ", " — "):
        if delimiter in cleaned:
            cleaned = cleaned.split(delimiter, 1)[0]
    return " ".join(cleaned.split())


def normalize_text_list(values: List[str], *, strip_descriptor: bool = False) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        cleaned = clean_entity_label(value) if strip_descriptor else " ".join(value.strip().split())
        if not cleaned:
            continue
        if cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized


def max_repeated_char_run(text: str) -> int:
    longest = 0
    current = 0
    prev = ""
    for ch in text:
        if ch == prev:
            current += 1
        else:
            current = 1
            prev = ch
        if current > longest:
            longest = current
    return longest


def assess_text_quality(text: str) -> Dict[str, Any]:
    compact = " ".join((text or "").split())
    char_count = len(compact)
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", compact)
    word_count = len(tokens)
    alpha_count = sum(1 for ch in compact if ch.isalpha())
    unique_word_count = len({token.lower() for token in tokens})
    alpha_ratio = (alpha_count / char_count) if char_count else 0.0
    unique_word_ratio = (unique_word_count / word_count) if word_count else 0.0
    repeated_run = max_repeated_char_run(compact)
    return {
        "char_count": char_count,
        "word_count": word_count,
        "alpha_ratio": round(alpha_ratio, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
        "max_repeated_char_run": repeated_run,
    }


def build_skip_reason(quality: Dict[str, Any], args: argparse.Namespace) -> Optional[str]:
    if quality["char_count"] == 0:
        return "empty text"
    if quality["char_count"] < args.min_text_chars:
        return (
            f"too short ({quality['char_count']} chars < minimum {args.min_text_chars})"
        )
    if quality["word_count"] < args.min_text_words:
        return (
            f"too few words ({quality['word_count']} words < minimum {args.min_text_words})"
        )
    if quality["alpha_ratio"] < args.min_alpha_ratio:
        return (
            f"low alphabetic density ({quality['alpha_ratio']:.2f} < minimum {args.min_alpha_ratio:.2f})"
        )
    if quality["word_count"] > 0 and quality["unique_word_ratio"] < args.min_unique_word_ratio:
        return (
            "high repetition/noise "
            f"({quality['unique_word_ratio']:.2f} unique ratio < minimum {args.min_unique_word_ratio:.2f})"
        )
    if quality["max_repeated_char_run"] > args.max_repeated_char_run:
        return (
            "contains long repeated-character sequences "
            f"({quality['max_repeated_char_run']} > max {args.max_repeated_char_run})"
        )
    return None


def build_skipped_model_result(skip_reason: str) -> Dict[str, Any]:
    return {
        "headline": "Skipped low-signal document",
        "importance_score": 0,
        "reason": f"Skipped before model call: {skip_reason}.",
        "key_insights": [],
        "tags": ["skipped"],
        "power_mentions": [],
        "agency_involvement": [],
        "lead_types": [],
    }


def derive_justice_pdf_url(filename: str, base_url: str = DEFAULT_JUSTICE_FILES_BASE_URL) -> Optional[str]:
    if not filename:
        return None
    dataset_match = re.search(r"DataSet\s*0*(\d+)", filename, flags=re.IGNORECASE)
    efta_match = re.search(r"(EFTA\d{8})", filename, flags=re.IGNORECASE)
    if not dataset_match or not efta_match:
        return None
    try:
        dataset_num = int(dataset_match.group(1))
    except ValueError:
        return None
    if dataset_num < 1:
        return None
    efta_id = efta_match.group(1).upper()
    base = base_url.rstrip("/")
    return f"{base}/DataSet%20{dataset_num}/{efta_id}.pdf"


def count_total_rows(path: Path, *, input_glob: str = "*.txt") -> int:
    """Count total number of source rows for CSV or directory input."""
    if path.is_dir():
        return sum(1 for _ in iter_rows(path, input_glob=input_glob, include_text=False))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def calculate_workload(
    path: Path,
    *,
    input_glob: str,
    max_rows: Optional[int],
    completed_filenames: Set[str],
    start_row: int,
    end_row: Optional[int],
) -> Dict[str, int]:
    total = 0
    already_done = 0
    for idx, row in enumerate(iter_rows(path, input_glob=input_glob, include_text=False), start=1):
        if idx < start_row:
            continue
        if end_row is not None and idx > end_row:
            break
        total += 1
        if completed_filenames and row.get("filename") in completed_filenames:
            already_done += 1
        if max_rows is not None and total >= max_rows:
            break
    workload = max(0, total - already_done)
    return {"total": total, "already_done": already_done, "workload": workload}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(
    start_time: float,
    processed: int,
    total: int,
    power_watts: Optional[float] = None,
    electric_rate: Optional[float] = None,
) -> str:
    if total <= 0:
        return ""
    if processed == 0:
        return "(ETA estimating...)"
    elapsed = time.monotonic() - start_time
    if elapsed <= 0:
        return "(ETA --:--:--)"
    rate = processed / elapsed
    if rate <= 0:
        return "(ETA --:--:--)"
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate else 0

    # Base ETA message
    eta_msg = f"(ETA {format_duration(eta_seconds)})"

    # Add energy/cost estimates if available
    if power_watts is not None and electric_rate is not None:
        total_estimated_hours = (elapsed + eta_seconds) / 3600
        energy_cost = calculate_energy_cost(power_watts, electric_rate, total_estimated_hours)
        if energy_cost:
            eta_msg += f" | Est. total: {energy_cost['energy_kwh']:.2f} kWh / ${energy_cost['cost_usd']:.2f}"

    return eta_msg


class OutputRouter:
    def __init__(self, args: argparse.Namespace, fieldnames: List[str]):
        self.args = args
        self.fieldnames = fieldnames
        self.chunk_size = max(0, args.chunk_size)
        self.mode = "chunk" if self.chunk_size > 0 else "single"
        self.include_action_items = args.include_action_items
        self.csv_handle = None
        self.json_handle = None
        self.csv_writer = None
        self.current_chunk: Optional[Tuple[int, int]] = None
        self.current_json_path: Optional[Path] = None
        if self.mode == "single":
            self._init_single()
        else:
            self._init_chunk_state()

    def _init_single(self) -> None:
        csv_mode = "a" if self.args.resume and self.args.output.exists() else "w"
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.csv_handle = self.args.output.open(csv_mode, newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=self.fieldnames)
        if csv_mode == "w":
            self.csv_writer.writeheader()
        json_mode = "a" if self.args.resume and self.args.json_output.exists() else "w"
        self.args.json_output.parent.mkdir(parents=True, exist_ok=True)
        self.json_handle = self.args.json_output.open(json_mode, encoding="utf-8")

    def _init_chunk_state(self) -> None:
        self.chunk_dir: Path = self.args.chunk_dir
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_manifest: Path = self.args.chunk_manifest
        self.manifest_entries = self._load_manifest()
        self.manifest_dirty = False
        self.total_dataset_rows = None  # Will be set by main()

    def _load_manifest(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        if not self.chunk_manifest.exists():
            return {}
        try:
            with self.chunk_manifest.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

        # Handle both old format (array) and new format (object with chunks)
        if isinstance(data, list):
            # Old format: array of chunks
            chunk_list = data
        elif isinstance(data, dict) and "chunks" in data:
            # New format: object with metadata and chunks
            chunk_list = data["chunks"]
        else:
            return {}

        entries: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for entry in chunk_list:
            key = (entry.get("start_row"), entry.get("end_row"))
            if not isinstance(key[0], int) or not isinstance(key[1], int):
                continue
            entries[key] = entry
        return entries

    def write(self, row_idx: int, csv_row: Dict[str, Any], json_record: Dict[str, Any]) -> None:
        if self.mode == "single":
            self.csv_writer.writerow(csv_row)
            self.csv_handle.flush()
            self.json_handle.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            self.json_handle.flush()
            return

        chunk_bounds = self._chunk_bounds(row_idx)
        self._ensure_chunk(chunk_bounds)
        self.json_handle.write(json.dumps(json_record, ensure_ascii=False) + "\n")
        self.json_handle.flush()

    def _chunk_bounds(self, row_idx: int) -> Tuple[int, int]:
        chunk_start = ((row_idx - 1) // self.chunk_size) * self.chunk_size + 1
        chunk_end = chunk_start + self.chunk_size - 1
        if self.args.end_row is not None:
            chunk_end = min(chunk_end, self.args.end_row)
        return (chunk_start, chunk_end)

    def _ensure_chunk(self, chunk_bounds: Tuple[int, int]) -> None:
        if self.current_chunk == chunk_bounds:
            return
        self._close_chunk()
        self.current_chunk = chunk_bounds
        chunk_start, chunk_end = chunk_bounds
        base = f"epstein_ranked_{chunk_start:05d}_{chunk_end:05d}"
        json_path = self.chunk_dir / f"{base}.jsonl"
        json_exists = json_path.exists()
        if json_exists and not self.args.resume and not self.args.overwrite_output:
            raise FileExistsError(
                f"Chunk JSON {json_path} exists. Use --resume or --overwrite-output."
            )
        json_mode = "a" if self.args.resume and json_exists else "w"
        self.json_handle = json_path.open(json_mode, encoding="utf-8")
        self.current_json_path = json_path

    def _close_chunk(self) -> None:
        if self.json_handle:
            self.json_handle.close()
            self.json_handle = None
        if self.current_chunk and self.current_json_path:
            chunk_start, chunk_end = self.current_chunk
            entry = {
                "start_row": chunk_start,
                "end_row": chunk_end,
                "json": str(self.current_json_path.as_posix()),
            }
            self.manifest_entries[self.current_chunk] = entry
            self.manifest_dirty = True
            # Write manifest immediately after each chunk closes
            self._write_manifest()
        self.current_chunk = None
        self.current_json_path = None

    def _write_manifest(self) -> None:
        """Write the manifest file to disk."""
        if not self.manifest_dirty:
            return
        entries = sorted(self.manifest_entries.values(), key=lambda e: e["start_row"])

        # Calculate total rows processed
        total_processed = 0
        for entry in entries:
            # Count actual lines in the chunk file
            chunk_path = Path(entry["json"])
            if chunk_path.exists():
                with chunk_path.open(encoding="utf-8") as f:
                    total_processed += sum(1 for _ in f)

        # Build manifest with metadata
        manifest = {
            "metadata": {
                "total_dataset_rows": self.total_dataset_rows if self.total_dataset_rows else "unknown",
                "rows_processed": total_processed,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "chunks": entries
        }

        self.chunk_manifest.parent.mkdir(parents=True, exist_ok=True)
        with self.chunk_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        self.manifest_dirty = False

    def close(self) -> None:
        if self.mode == "single":
            if self.csv_handle:
                self.csv_handle.close()
            if self.json_handle:
                self.json_handle.close()
            return
        self._close_chunk()
        self._write_manifest()

def build_config_metadata(args: argparse.Namespace, prompt_source: str) -> Dict[str, Any]:
    """Build metadata dictionary from config for inclusion in requests and outputs."""
    metadata = {
        "endpoint": args.endpoint,
        "model": args.model,
        "justice_files_base_url": args.justice_files_base_url,
        "temperature": args.temperature,
        "max_parallel_requests": args.max_parallel_requests,
        "prompt_source": prompt_source,
    }
    if args.dataset_tag:
        metadata["dataset_tag"] = args.dataset_tag
    if args.dataset_workspace_root:
        metadata["dataset_workspace_root"] = str(args.dataset_workspace_root)
    if args.dataset_source_label:
        metadata["dataset_source_label"] = args.dataset_source_label
    if args.dataset_source_url:
        metadata["dataset_source_url"] = args.dataset_source_url
    if args.reasoning_effort is not None:
        metadata["reasoning_effort"] = args.reasoning_effort
    metadata["skip_low_quality"] = args.skip_low_quality
    metadata["skip_thresholds"] = {
        "min_text_chars": args.min_text_chars,
        "min_text_words": args.min_text_words,
        "min_alpha_ratio": args.min_alpha_ratio,
        "min_unique_word_ratio": args.min_unique_word_ratio,
        "max_repeated_char_run": args.max_repeated_char_run,
    }
    if args.api_key:
        metadata["api_key_used"] = True
    if args.power_watts is not None:
        metadata["power_watts"] = args.power_watts
    if args.electric_rate is not None:
        metadata["electric_rate"] = args.electric_rate
    return metadata


def load_dataset_profile(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Dataset metadata file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset metadata file must contain a JSON object: {path}")
    return data


def write_run_metadata(
    *,
    args: argparse.Namespace,
    prompt_source: str,
    config_metadata: Dict[str, Any],
    workload_stats: Dict[str, int],
    total_dataset_rows: Optional[int],
    dataset_profile: Optional[Dict[str, Any]],
) -> None:
    if not args.run_metadata_file:
        return

    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "tag": args.dataset_tag,
            "source_label": args.dataset_source_label,
            "source_url": args.dataset_source_url,
        },
        "input": {
            "path": str(args.input),
            "is_directory": args.input.is_dir(),
            "input_glob": args.input_glob if args.input.is_dir() else None,
            "total_rows": total_dataset_rows if total_dataset_rows is not None else "unknown",
        },
        "workload": workload_stats,
        "outputs": {
            "output_csv": str(args.output),
            "output_jsonl": str(args.json_output),
            "checkpoint": str(args.checkpoint),
            "chunk_size": args.chunk_size,
            "chunk_dir": str(args.chunk_dir),
            "chunk_manifest": str(args.chunk_manifest),
        },
        "config": config_metadata,
        "prompt_source": prompt_source,
    }
    if dataset_profile is not None:
        payload["dataset_profile"] = dataset_profile

    run_metadata_file = Path(args.run_metadata_file)
    run_metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with run_metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def calculate_energy_cost(
    power_watts: Optional[float],
    electric_rate: Optional[float],
    hours: float,
) -> Optional[Dict[str, float]]:
    """Calculate energy consumption and cost."""
    if power_watts is None or electric_rate is None or hours <= 0:
        return None
    energy_kwh = (power_watts * hours) / 1000.0
    cost = energy_kwh * electric_rate
    return {"energy_kwh": energy_kwh, "cost_usd": cost}


def format_cost_summary(
    power_watts: Optional[float],
    electric_rate: Optional[float],
    elapsed_hours: float,
    override_hours: Optional[float],
) -> Optional[str]:
    if power_watts is None or electric_rate is None:
        return None
    hours = override_hours if override_hours is not None else elapsed_hours
    if hours <= 0:
        return None
    energy_kwh = (power_watts * hours) / 1000.0
    cost = energy_kwh * electric_rate
    return f"Energy ≈ {energy_kwh:.2f} kWh | Est. cost ≈ ${cost:.2f} (rate ${electric_rate}/kWh)"


def list_models(endpoint: str, api_key: Optional[str], timeout: float) -> None:
    """Print available model IDs from the endpoint."""
    url = f"{endpoint.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    if not models:
        print("No models reported by endpoint.")
        return
    print("Available models:")
    for entry in models:
        model_id = entry.get("id")
        created = entry.get("created")
        extra = f" (created {created})" if created else ""
        print(f" - {model_id}{extra}")


def rebuild_manifest(chunk_dir: Path, manifest_path: Path) -> None:
    """Scan chunk directory and rebuild the manifest file."""
    import re

    # Pattern to match chunk files: epstein_ranked_XXXXX_YYYYY.jsonl
    pattern = re.compile(r"epstein_ranked_(\d{5})_(\d{5})\.jsonl")

    chunks = []
    if not chunk_dir.exists():
        print(f"Chunk directory not found: {chunk_dir}")
        return

    for file_path in sorted(chunk_dir.glob("epstein_ranked_*.jsonl")):
        match = pattern.match(file_path.name)
        if not match:
            print(f"Skipping non-matching file: {file_path.name}")
            continue

        start_row = int(match.group(1))
        end_row = int(match.group(2))

        # Use relative path: chunk_dir/filename
        relative_path = chunk_dir / file_path.name

        chunks.append({
            "start_row": start_row,
            "end_row": end_row,
            "json": str(relative_path.as_posix()),
        })

    if not chunks:
        print(f"No chunk files found in {chunk_dir}")
        return

    # Sort by start_row
    chunks.sort(key=lambda c: c["start_row"])

    # Count total rows processed
    total_processed = 0
    for chunk in chunks:
        chunk_path = Path(chunk["json"])
        if chunk_path.exists():
            with chunk_path.open(encoding="utf-8") as f:
                total_processed += sum(1 for _ in f)

    # Try to get total dataset rows from the source CSV
    # Look for common CSV filenames in data/ directory
    csv_candidates = [
        Path("data/EPS_FILES_20K_NOV2026.csv"),
        Path("data") / "epstein_files.csv",
    ]
    total_dataset_rows = None
    for csv_path in csv_candidates:
        if csv_path.exists():
            print(f"Counting total rows in {csv_path}...")
            try:
                total_dataset_rows = count_total_rows(csv_path)
                print(f"Found {total_dataset_rows:,} total rows in dataset")
                break
            except Exception as e:
                print(f"Error counting rows: {e}")

    # Fall back to highest end_row if CSV not found
    if total_dataset_rows is None:
        total_dataset_rows = max((c["end_row"] for c in chunks), default=0) if chunks else "unknown"
        print(f"Source CSV not found, using highest chunk end_row: {total_dataset_rows}")

    # Build manifest with metadata
    manifest = {
        "metadata": {
            "total_dataset_rows": total_dataset_rows,
            "rows_processed": total_processed,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "chunks": chunks
    }

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Rebuilt manifest with {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - Rows {chunk['start_row']:,}–{chunk['end_row']:,}: {chunk['json']}")
    if isinstance(total_dataset_rows, int):
        print(f"Total: {total_processed:,} rows processed out of {total_dataset_rows:,} dataset rows")
    else:
        print(f"Total: {total_processed:,} rows processed")
    print(f"Manifest written to: {manifest_path}")


def build_output_records(
    *,
    idx: int,
    source_row: Dict[str, str],
    result: Dict[str, Any],
    args: argparse.Namespace,
    config_metadata: Dict[str, Any],
    quality: Dict[str, Any],
    processing_status: str,
    skip_reason: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source_pdf_url = derive_justice_pdf_url(
        source_row.get("filename", ""), base_url=args.justice_files_base_url
    )
    key_insights = normalize_text_list(ensure_list(result.get("key_insights")))
    tags = normalize_text_list(ensure_list(result.get("tags")))
    power_mentions = normalize_text_list(
        ensure_list(result.get("power_mentions")), strip_descriptor=True
    )
    agency_involvement = normalize_agencies(
        normalize_text_list(ensure_list(result.get("agency_involvement")), strip_descriptor=True)
    )
    lead_types = normalize_lead_types(ensure_list(result.get("lead_types")))
    action_items = (
        normalize_text_list(ensure_list(result.get("action_items")))
        if args.include_action_items
        else []
    )

    csv_row = {
        "filename": source_row["filename"],
        "source_row_index": idx,
        "source_pdf_url": source_pdf_url or "",
        "processing_status": processing_status,
        "skip_reason": skip_reason,
        "headline": result.get("headline", ""),
        "importance_score": result.get("importance_score", ""),
        "reason": result.get("reason", ""),
        "key_insights": "; ".join(key_insights),
        "tags": "; ".join(tags),
        "power_mentions": "; ".join(power_mentions),
        "agency_involvement": "; ".join(agency_involvement),
        "lead_types": "; ".join(lead_types),
    }
    if args.include_action_items:
        csv_row["action_items"] = "; ".join(action_items)

    json_record: Dict[str, Any] = {
        "filename": source_row["filename"],
        "source_pdf_url": source_pdf_url,
        "headline": result.get("headline", ""),
        "importance_score": result.get("importance_score", ""),
        "reason": result.get("reason", ""),
        "key_insights": key_insights,
        "tags": tags,
        "power_mentions": power_mentions,
        "agency_involvement": agency_involvement,
        "lead_types": lead_types,
        "metadata": {
            "source_row_index": idx,
            "original_row": source_row,
            "config": config_metadata,
            "source_pdf_url": source_pdf_url,
            "processing_status": processing_status,
            "skip_reason": skip_reason,
            "text_quality": quality,
        },
    }
    if args.include_action_items:
        json_record["action_items"] = action_items
    return csv_row, json_record


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_models(args.endpoint, args.api_key, args.timeout)
        return

    if args.rebuild_manifest:
        rebuild_manifest(args.chunk_dir, args.chunk_manifest)
        return

    # Load system prompt from file or use inline/default
    system_prompt, prompt_source = load_system_prompt(args)

    if not args.input.exists():
        sys.exit(f"Input path not found: {args.input}")
    if args.dataset_metadata_file and not args.dataset_metadata_file.exists():
        sys.exit(f"Dataset metadata file not found: {args.dataset_metadata_file}")
    if args.max_parallel_requests < 1:
        sys.exit("--max-parallel-requests must be >= 1")
    if args.min_text_chars < 0:
        sys.exit("--min-text-chars must be >= 0")
    if args.min_text_words < 0:
        sys.exit("--min-text-words must be >= 0")
    if args.min_alpha_ratio < 0:
        sys.exit("--min-alpha-ratio must be >= 0")
    if args.min_unique_word_ratio < 0:
        sys.exit("--min-unique-word-ratio must be >= 0")
    if args.max_repeated_char_run < 1:
        sys.exit("--max-repeated-char-run must be >= 1")
    if args.start_row < 1:
        sys.exit("--start-row must be >= 1")
    if args.end_row is not None and args.end_row < args.start_row:
        sys.exit("--end-row must be greater than or equal to --start-row")
    if args.chunk_size <= 0:
        if (
            args.output.exists()
            and not args.resume
            and not args.overwrite_output
        ):
            sys.exit(
                f"Output file {args.output} already exists. "
                "Use --resume to append/skip or --overwrite-output to replace."
            )
        if (
            args.json_output.exists()
            and not args.resume
            and not args.overwrite_output
        ):
            sys.exit(
                f"JSON output file {args.json_output} already exists. "
                "Use --resume to append/skip or --overwrite-output to replace."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "source_row_index",
        "source_pdf_url",
        "processing_status",
        "skip_reason",
        "headline",
        "importance_score",
        "reason",
        "key_insights",
        "tags",
        "power_mentions",
        "agency_involvement",
        "lead_types",
    ]
    if args.include_action_items:
        fieldnames.append("action_items")

    processed = 0
    completed_filenames: Set[str] = set()
    if args.resume:
        completed_filenames |= load_checkpoint(args.checkpoint)
        completed_filenames |= load_jsonl_filenames(args.json_output)
    for extra_json in args.known_json:
        completed_filenames |= load_jsonl_filenames(Path(extra_json))
    if completed_filenames:
        print(f"Skipping {len(completed_filenames)} pre-processed files.")

    workload_stats = calculate_workload(
        args.input,
        input_glob=args.input_glob,
        max_rows=args.max_rows,
        completed_filenames=completed_filenames if args.resume else set(),
        start_row=args.start_row,
        end_row=args.end_row,
    )
    total_candidates = workload_stats["total"]
    already_done = workload_stats["already_done"] if args.resume else 0
    target_total = workload_stats["workload"]
    if target_total <= 0:
        print("No new rows to process. Exiting.")
        return
    range_desc = f"rows {args.start_row}-{args.end_row if args.end_row else 'end'}"
    print(
        f"Processing {target_total} new rows within {range_desc} "
        f"(skipping {already_done} already completed, total considered {total_candidates})."
    )

    output_router = OutputRouter(args, fieldnames)
    total_dataset_rows: Optional[int] = None

    # Count total rows in dataset for manifest metadata
    if output_router.mode == "chunk":
        print("Counting total rows in dataset...")
        output_router.total_dataset_rows = count_total_rows(args.input, input_glob=args.input_glob)
        total_dataset_rows = output_router.total_dataset_rows
        print(f"Total dataset: {output_router.total_dataset_rows:,} rows")
    elif args.run_metadata_file:
        # Single-file mode still records total row count in run metadata.
        total_dataset_rows = count_total_rows(args.input, input_glob=args.input_glob)

    checkpoint_handle = (
        args.checkpoint.open("a", encoding="utf-8") if args.checkpoint else None
    )

    # Build config metadata to include in requests and outputs
    config_metadata = build_config_metadata(args, prompt_source)
    try:
        dataset_profile = load_dataset_profile(args.dataset_metadata_file)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    write_run_metadata(
        args=args,
        prompt_source=prompt_source,
        config_metadata=config_metadata,
        workload_stats=workload_stats,
        total_dataset_rows=total_dataset_rows,
        dataset_profile=dataset_profile,
    )

    start_time = time.monotonic()
    emit_order: Deque[int] = deque()
    pending_results: Dict[int, Dict[str, Any]] = {}
    in_flight: Dict[concurrent.futures.Future[Dict[str, Any]], Dict[str, Any]] = {}
    scheduled = 0
    skipped = 0
    failed = 0
    model_scored = 0
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_requests)
        if args.max_parallel_requests > 1
        else None
    )

    def flush_ready() -> None:
        nonlocal processed, skipped, failed, model_scored
        while emit_order and emit_order[0] in pending_results:
            row_idx = emit_order.popleft()
            outcome = pending_results.pop(row_idx)
            if outcome["type"] == "error":
                failed += 1
                print(
                    f"  ! Failed to analyze {outcome['row']['filename']}: {outcome['error']}",
                    file=sys.stderr,
                )
                continue

            csv_row, json_record = build_output_records(
                idx=row_idx,
                source_row=outcome["row"],
                result=outcome["result"],
                args=args,
                config_metadata=config_metadata,
                quality=outcome["quality"],
                processing_status=outcome["processing_status"],
                skip_reason=outcome["skip_reason"],
            )
            output_router.write(row_idx, csv_row, json_record)

            filename = outcome["row"]["filename"]
            if checkpoint_handle:
                checkpoint_handle.write(filename + "\n")
                checkpoint_handle.flush()
            completed_filenames.add(filename)
            processed += 1
            if outcome["processing_status"] == "skipped":
                skipped += 1
            else:
                model_scored += 1

    def harvest_completed(*, block: bool) -> None:
        if not in_flight:
            return
        timeout = None if block else 0
        done, _ = concurrent.futures.wait(
            set(in_flight.keys()),
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            return
        for future in done:
            context = in_flight.pop(future)
            row_idx = context["idx"]
            row = context["row"]
            quality = context["quality"]
            try:
                result = future.result()
                pending_results[row_idx] = {
                    "type": "record",
                    "row": row,
                    "result": result,
                    "quality": quality,
                    "processing_status": "processed",
                    "skip_reason": "",
                }
            except Exception as exc:  # noqa: BLE001
                pending_results[row_idx] = {
                    "type": "error",
                    "row": row,
                    "error": str(exc),
                }
        flush_ready()

    try:
        for idx, row in enumerate(
            iter_rows(args.input, input_glob=args.input_glob, include_text=True), start=1
        ):
            if idx < args.start_row:
                continue
            if args.end_row is not None and idx > args.end_row:
                break
            if args.max_rows is not None and scheduled >= args.max_rows:
                break

            filename = row["filename"]
            text = row["text"]

            if filename in completed_filenames:
                print(f"[Row {idx}] [skip] {filename} already processed.", flush=True)
                continue

            scheduled += 1
            emit_order.append(idx)

            quality = assess_text_quality(text)
            skip_reason = build_skip_reason(quality, args) if args.skip_low_quality else None
            if skip_reason:
                print(f"[Row {idx}] [skip] {filename}: {skip_reason}", flush=True)
                pending_results[idx] = {
                    "type": "record",
                    "row": row,
                    "result": build_skipped_model_result(skip_reason),
                    "quality": quality,
                    "processing_status": "skipped",
                    "skip_reason": skip_reason,
                }
                flush_ready()
                continue

            # Show both source row index and processing progress
            if target_total:
                progress_prefix = f"[Row {idx}] [{processed + 1}/{target_total} new]"
            else:
                progress_prefix = f"[Row {idx}] [{processed + 1}]"

            eta_text = format_eta(
                start_time,
                processed,
                target_total,
                args.power_watts,
                args.electric_rate,
            )
            print(f"{progress_prefix} Processing {filename}... {eta_text}", flush=True)

            request_kwargs = {
                "endpoint": args.endpoint,
                "model": args.model,
                "filename": filename,
                "text": text,
                "system_prompt": system_prompt,
                "api_key": args.api_key,
                "timeout": args.timeout,
                "temperature": args.temperature,
                "reasoning_effort": args.reasoning_effort,
                "config_metadata": config_metadata,
            }

            if executor is None:
                try:
                    result = call_model(**request_kwargs)
                    pending_results[idx] = {
                        "type": "record",
                        "row": row,
                        "result": result,
                        "quality": quality,
                        "processing_status": "processed",
                        "skip_reason": "",
                    }
                except Exception as exc:  # noqa: BLE001
                    pending_results[idx] = {
                        "type": "error",
                        "row": row,
                        "error": str(exc),
                    }
                flush_ready()
            else:
                while len(in_flight) >= args.max_parallel_requests:
                    harvest_completed(block=True)
                future = executor.submit(
                    call_model,
                    endpoint=args.endpoint,
                    model=args.model,
                    filename=filename,
                    text=text,
                    system_prompt=system_prompt,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    config_metadata=config_metadata,
                )
                in_flight[future] = {"idx": idx, "row": row, "quality": quality}
                harvest_completed(block=False)

            if args.sleep:
                time.sleep(args.sleep)

        while in_flight:
            harvest_completed(block=True)

        flush_ready()
    finally:
        if executor:
            executor.shutdown(wait=True)
        if checkpoint_handle:
            checkpoint_handle.close()
        output_router.close()

    elapsed = time.monotonic() - start_time
    elapsed_hours = elapsed / 3600
    cost_summary = format_cost_summary(args.power_watts, args.electric_rate, elapsed_hours, args.run_hours)
    if args.chunk_size > 0:
        complete_msg = (
            f"Completed {processed} new rows in {format_duration(elapsed)} "
            f"({model_scored} modeled, {skipped} skipped, {failed} failed). "
            f"Chunks saved under {args.chunk_dir} | manifest: {args.chunk_manifest}"
        )
    else:
        complete_msg = (
            f"Completed {processed} new rows in {format_duration(elapsed)} "
            f"({model_scored} modeled, {skipped} skipped, {failed} failed). "
            f"CSV saved to {args.output} | JSONL appended to {args.json_output}"
        )
    if cost_summary:
        complete_msg = f"{complete_msg}\n{cost_summary}"
    print(complete_msg)


if __name__ == "__main__":
    main()
