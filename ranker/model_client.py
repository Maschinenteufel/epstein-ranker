from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from .constants import RETRIABLE_HTTP_STATUS_CODES


class UnsupportedEndpointError(RuntimeError):
    """Raised when a specific API route is unavailable on the target server."""


class ModelRequestError(RuntimeError):
    """Raised for model request failures with retriable vs permanent classification."""

    def __init__(self, message: str, *, retriable: bool) -> None:
        super().__init__(message)
        self.retriable = retriable


def build_text_analysis_input(filename: str, text: str) -> str:
    return (
        "Analyze the following document and respond with the JSON schema "
        "described in the system prompt.\n"
        f"Filename: {filename}\n"
        "---------\n"
        f"{text.strip()}\n"
        "---------"
    )


def build_image_analysis_instruction(filename: str, page_count: int) -> str:
    pages_label = f"{page_count} page(s)" if page_count > 0 else "the provided page image"
    return (
        "Analyze this source document image and respond with the JSON schema "
        "described in the system prompt.\n"
        f"Filename: {filename}\n"
        f"Input contains {pages_label}."
    )


def infer_api_priority(endpoint: str, *, prefer_openai: bool = False) -> List[str]:
    if prefer_openai:
        return ["openai", "chat"]
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/api/v1"):
        return ["chat", "openai"]
    return ["openai", "chat"]


def derive_localhost_fallback_endpoints(endpoint: str) -> List[str]:
    normalized = endpoint.rstrip("/")
    if not re.match(r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/.*)?$", normalized):
        return []
    scheme = "https" if normalized.startswith("https://") else "http"
    host = "127.0.0.1" if "127.0.0.1" in normalized else "localhost"
    candidates = [
        f"{scheme}://{host}:5555/api/v1",
        f"{scheme}://{host}:5555/v1",
    ]
    return [candidate for candidate in candidates if candidate != normalized]


def build_request_targets(
    endpoint: str,
    api_format: str,
    *,
    prefer_openai: bool = False,
) -> List[Tuple[str, str]]:
    normalized = endpoint.rstrip("/")
    targets: List[Tuple[str, str]] = []

    if api_format == "openai":
        targets.append(("openai", normalized))
        for fallback_endpoint in derive_localhost_fallback_endpoints(normalized):
            targets.append(("openai", fallback_endpoint))
    elif api_format == "chat":
        targets.append(("chat", normalized))
        for fallback_endpoint in derive_localhost_fallback_endpoints(normalized):
            targets.append(("chat", fallback_endpoint))
    else:
        for mode in infer_api_priority(normalized, prefer_openai=prefer_openai):
            targets.append((mode, normalized))
        for fallback_endpoint in derive_localhost_fallback_endpoints(normalized):
            for mode in infer_api_priority(fallback_endpoint, prefer_openai=prefer_openai):
                targets.append((mode, fallback_endpoint))

    deduped: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for mode, base in targets:
        key = (mode, base)
        if key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return deduped


def image_path_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_map.get(suffix)
    if not mime:
        raise RuntimeError(f"Unsupported image suffix for multimodal input: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_pdf_page_to_data_url(pdf_path: Path, *, page_number: int, dpi: int) -> str:
    with tempfile.TemporaryDirectory(prefix="epstein_pdf_render_") as tmpdir:
        out_prefix = Path(tmpdir) / f"page_{page_number}"
        cmd = [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-f",
            str(page_number),
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pdftoppm is required for PDF image mode but was not found on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"Failed to render PDF page for {pdf_path}: {detail}") from exc
        rendered_path = out_prefix.with_suffix(".png")
        if not rendered_path.exists():
            raise RuntimeError(f"PDF rendering produced no image for {pdf_path}")
        encoded = base64.b64encode(rendered_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


def prepare_image_data_urls(
    image_path: Path,
    *,
    max_pages: int,
    render_dpi: int,
) -> List[str]:
    suffix = image_path.suffix.lower()
    if suffix == ".pdf":
        return [
            render_pdf_page_to_data_url(image_path, page_number=page_number, dpi=render_dpi)
            for page_number in range(1, max_pages + 1)
        ]
    return [image_path_to_data_url(image_path)]


def post_request(
    *,
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(url, json=payload, timeout=timeout, headers=headers)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ModelRequestError(f"Request failed for {url}: {exc}", retriable=True) from exc
    except requests.RequestException as exc:
        raise ModelRequestError(f"Request failed for {url}: {exc}", retriable=False) from exc

    if response.status_code in (404, 405):
        raise UnsupportedEndpointError(
            f"HTTP {response.status_code} from {url}: endpoint not supported by this server"
        )
    if response.status_code >= 400:
        snippet = response.text[:500].replace("\n", " ")
        retriable = (
            response.status_code in RETRIABLE_HTTP_STATUS_CODES
            or 500 <= response.status_code < 600
        )
        raise ModelRequestError(
            f"HTTP {response.status_code} from {url}: {snippet}",
            retriable=retriable,
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        snippet = response.text[:500].replace("\n", " ")
        raise ModelRequestError(
            f"Invalid JSON response from {url}: {snippet}",
            retriable=False,
        ) from exc


def extract_openai_content(data: Dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelRequestError(
            f"Unexpected OpenAI response format: {data}",
            retriable=False,
        ) from exc
    if not isinstance(content, str):
        raise ModelRequestError(
            f"Unexpected OpenAI response content type: {type(content)}",
            retriable=False,
        )
    return content


def extract_chat_content(data: Dict[str, Any]) -> str:
    output = data.get("output")
    if isinstance(output, list):
        collected: List[str] = []
        for item in output:
            if isinstance(item, str):
                collected.append(item)
                continue
            if isinstance(item, dict):
                value = item.get("content")
                if isinstance(value, str):
                    collected.append(value)
                    continue
                value = item.get("text")
                if isinstance(value, str):
                    collected.append(value)
        if collected:
            return "\n".join(collected)

    for key in ("content", "response", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ModelRequestError(
        f"Unexpected chat response format: {data}",
        retriable=False,
    )


def ensure_json_dict(content: str) -> Dict[str, Any]:
    candidate = content.strip()
    if not candidate:
        raise ValueError("Model returned an empty message.")

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


def call_model(
    *,
    endpoint: str,
    api_format: str,
    model: str,
    filename: str,
    text: str,
    input_kind: str,
    image_path: Optional[Path],
    image_max_pages: int,
    image_render_dpi: int,
    system_prompt: str,
    api_key: Optional[str],
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    temperature: float,
    reasoning_effort: Optional[str],
    config_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if input_kind not in {"text", "image"}:
        raise RuntimeError(f"Unsupported input kind: {input_kind}")
    if input_kind == "image" and api_format == "chat":
        raise RuntimeError(
            "Image mode requires OpenAI-style chat completions. Use --api-format openai or auto."
        )
    if input_kind == "image" and image_path is None:
        raise RuntimeError("Image mode requires a valid source file path.")

    effective_api_format = "openai" if (input_kind == "image" and api_format == "auto") else api_format
    targets = build_request_targets(
        endpoint,
        effective_api_format,
        prefer_openai=(input_kind == "image"),
    )
    doc_input = build_text_analysis_input(filename, text)
    image_urls: List[str] = []
    if input_kind == "image" and image_path is not None:
        image_urls = prepare_image_data_urls(
            image_path,
            max_pages=max(1, image_max_pages),
            render_dpi=max(72, image_render_dpi),
        )
        doc_input = build_image_analysis_instruction(filename, len(image_urls))

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        saw_retriable_error = False
        for mode, base_url in targets:
            try:
                if mode == "openai":
                    user_content: Any
                    if input_kind == "image":
                        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": doc_input}]
                        for image_url in image_urls:
                            content_blocks.append(
                                {"type": "image_url", "image_url": {"url": image_url}}
                            )
                        user_content = content_blocks
                    else:
                        user_content = doc_input
                    payload: Dict[str, Any] = {
                        "model": model,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                    }
                    if reasoning_effort:
                        payload["reasoning"] = {"effort": reasoning_effort}
                    if config_metadata:
                        payload["metadata"] = config_metadata
                    data = post_request(
                        url=f"{base_url}/chat/completions",
                        payload=payload,
                        api_key=api_key,
                        timeout=timeout,
                    )
                    content = extract_openai_content(data)
                else:
                    payload = {
                        "model": model,
                        "system_prompt": system_prompt,
                        "input": doc_input,
                    }
                    data = post_request(
                        url=f"{base_url}/chat",
                        payload=payload,
                        api_key=api_key,
                        timeout=timeout,
                    )
                    content = extract_chat_content(data)
                return ensure_json_dict(content)
            except UnsupportedEndpointError as exc:
                last_error = exc
                continue
            except ModelRequestError as exc:
                last_error = exc
                saw_retriable_error = saw_retriable_error or exc.retriable
                continue

        if attempt < max_retries and saw_retriable_error:
            sleep_seconds = retry_backoff * (2 ** (attempt - 1))
            time.sleep(max(0.0, sleep_seconds))
            continue
        break

    candidate_urls = ", ".join(
        f"{base}/{'chat/completions' if mode == 'openai' else 'chat'}"
        for mode, base in targets
    )
    detail = str(last_error) if last_error else "unknown error"
    raise RuntimeError(
        f"Failed to analyze after {max_retries} attempt(s). Tried: {candidate_urls}. Last error: {detail}"
    )
