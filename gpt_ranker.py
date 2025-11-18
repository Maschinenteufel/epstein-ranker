#!/usr/bin/env python3
"""Rank Epstein file rows by querying a local GPT server."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
    "linking powerful actors to fresh controversy). Reserve 70+ for claims that, if true,\n"
    "would represent major revelations or next-step investigations.\n"
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


def apply_config_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    config_path: Path = args.config  # type: ignore[assignment]
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    for key, value in data.items():
        if not hasattr(args, key):
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
            "Call a local OpenAI-compatible server (e.g. gpt-oss-120b) to extract "
            "useful information and rank each CSV row."
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
        help="Path to the source CSV with 'filename' and 'text' columns.",
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
        default="openai/gpt-oss-120b",
        help="Model identifier exposed by the server (check via --list-models).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override the default system instructions sent to the model.",
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
        default=120.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models exposed by the endpoint and exit.",
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
    if args.config:
        apply_config_defaults(parser, args)
    return args


def call_model(
    *,
    endpoint: str,
    model: str,
    filename: str,
    text: str,
    system_prompt: str,
    api_key: Optional[str],
    timeout: float,
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    """Send the document to the local GPT server and return parsed JSON."""
    payload = {
        "model": model,
        "temperature": 0,
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


def iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "text" not in row or "filename" not in row:
                raise ValueError("Input CSV must contain 'filename' and 'text' columns.")
            yield row


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


def calculate_workload(
    path: Path,
    *,
    max_rows: Optional[int],
    completed_filenames: Set[str],
    start_row: int,
    end_row: Optional[int],
) -> Dict[str, int]:
    total = 0
    already_done = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
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


def format_eta(start_time: float, processed: int, total: int) -> str:
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
    return f"(ETA {format_duration(eta_seconds)})"


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
        self.current_paths: Optional[Tuple[Path, Path]] = None
        if self.mode == "single":
            self._init_single()
        else:
            self._init_chunk()

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

    def _init_chunk(self) -> None:
        self.chunk_dir: Path = self.args.chunk_dir
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_manifest: Path = self.args.chunk_manifest
        self.manifest_entries = self._load_manifest()
        self.manifest_dirty = False

    def _load_manifest(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        if not self.chunk_manifest.exists():
            return {}
        try:
            with self.chunk_manifest.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
        entries: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for entry in data:
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
        self.csv_writer.writerow(csv_row)
        self.csv_handle.flush()
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
        csv_path = self.chunk_dir / f"{base}.csv"
        json_path = self.chunk_dir / f"{base}.jsonl"
        csv_exists = csv_path.exists()
        json_exists = json_path.exists()
        if csv_exists and not self.args.resume and not self.args.overwrite_output:
            raise FileExistsError(
                f"Chunk CSV {csv_path} exists. Use --resume or --overwrite-output."
            )
        if json_exists and not self.args.resume and not self.args.overwrite_output:
            raise FileExistsError(
                f"Chunk JSON {json_path} exists. Use --resume or --overwrite-output."
            )
        csv_mode = "a" if self.args.resume and csv_exists else "w"
        json_mode = "a" if self.args.resume and json_exists else "w"
        self.csv_handle = csv_path.open(csv_mode, newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=self.fieldnames)
        if csv_mode == "w":
            self.csv_writer.writeheader()
        self.json_handle = json_path.open(json_mode, encoding="utf-8")
        self.current_paths = (csv_path, json_path)

    def _close_chunk(self) -> None:
        if self.csv_handle:
            self.csv_handle.close()
            self.csv_handle = None
        if self.json_handle:
            self.json_handle.close()
            self.json_handle = None
        if self.current_chunk and self.current_paths:
            chunk_start, chunk_end = self.current_chunk
            csv_path, json_path = self.current_paths
            entry = {
                "start_row": chunk_start,
                "end_row": chunk_end,
                "csv": str(csv_path.as_posix()),
                "json": str(json_path.as_posix()),
            }
            self.manifest_entries[self.current_chunk] = entry
            self.manifest_dirty = True
        self.current_chunk = None
        self.current_paths = None
        self.csv_writer = None

    def close(self) -> None:
        if self.mode == "single":
            if self.csv_handle:
                self.csv_handle.close()
            if self.json_handle:
                self.json_handle.close()
            return
        self._close_chunk()
        if self.manifest_dirty:
            entries = sorted(self.manifest_entries.values(), key=lambda e: e["start_row"])
            self.chunk_manifest.parent.mkdir(parents=True, exist_ok=True)
            with self.chunk_manifest.open("w", encoding="utf-8") as handle:
                json.dump(entries, handle, indent=2)

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


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_models(args.endpoint, args.api_key, args.timeout)
        return

    if not args.input.exists():
        sys.exit(f"Input CSV not found: {args.input}")
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
    checkpoint_handle = (
        args.checkpoint.open("a", encoding="utf-8") if args.checkpoint else None
    )
    start_time = time.monotonic()
    try:
        for idx, row in enumerate(iter_rows(args.input), start=1):
            if idx < args.start_row:
                continue
            if args.end_row is not None and idx > args.end_row:
                break
            if args.max_rows is not None and processed >= args.max_rows:
                break

            filename = row["filename"]
            text = row["text"]

            if filename in completed_filenames:
                print(f"[skip] {filename} already processed.", flush=True)
                continue

            progress_prefix = (
                f"[{processed + 1}/{target_total}]"
                if target_total
                else f"[{processed + 1}]"
            )
            eta_text = format_eta(start_time, processed, target_total)
            print(f"{progress_prefix} Processing {filename}... {eta_text}", flush=True)

            try:
                result = call_model(
                    endpoint=args.endpoint,
                    model=args.model,
                    filename=filename,
                    text=text,
                    system_prompt=args.system_prompt,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    reasoning_effort=args.reasoning_effort,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ! Failed to analyze {filename}: {exc}", file=sys.stderr)
                continue

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
                "filename": filename,
                "source_row_index": idx,
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
                "filename": filename,
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
                    "original_row": row,
                },
            }
            if args.include_action_items:
                json_record["action_items"] = action_items

            output_router.write(idx, csv_row, json_record)

            if checkpoint_handle:
                checkpoint_handle.write(filename + "\n")
                checkpoint_handle.flush()

            completed_filenames.add(filename)
            processed += 1
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        if checkpoint_handle:
            checkpoint_handle.close()
        output_router.close()

    elapsed = time.monotonic() - start_time
    elapsed_hours = elapsed / 3600
    cost_summary = format_cost_summary(args.power_watts, args.electric_rate, elapsed_hours, args.run_hours)
    if args.chunk_size > 0:
        complete_msg = (
            f"Completed {processed} new rows in {format_duration(elapsed)}. "
            f"Chunks saved under {args.chunk_dir} | manifest: {args.chunk_manifest}"
        )
    else:
        complete_msg = (
            f"Completed {processed} new rows in {format_duration(elapsed)}. "
            f"CSV saved to {args.output} | JSONL appended to {args.json_output}"
        )
    if cost_summary:
        complete_msg = f"{complete_msg}\n{cost_summary}"
    print(complete_msg)


if __name__ == "__main__":
    main()
