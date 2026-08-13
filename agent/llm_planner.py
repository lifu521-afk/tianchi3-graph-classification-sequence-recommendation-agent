from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .memory import read_events


SECRET_MARKERS = ("api_key", "apikey", "authorization", "secret", "token")


def _configured_value(settings: dict[str, Any], key: str) -> str:
    env_name = str(settings.get(key, "")).strip()
    return os.environ.get(env_name, "").strip() if env_name else ""


def is_configured(config: dict[str, Any]) -> bool:
    settings = config.get("llm_planner", {})
    return bool(
        settings.get("enabled", False)
        and _configured_value(settings, "api_key_env")
        and _configured_value(settings, "base_url_env")
        and _configured_value(settings, "model_env")
    )


def _redact(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in SECRET_MARKERS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value[-20:]]
    if isinstance(value, str):
        return value[-1200:]
    return value


def _recent_memory(path: Path, limit: int) -> list[dict[str, Any]]:
    events = read_events(path)
    useful = [
        event
        for event in events
        if event.get("event") in {"plan", "result", "official_feedback", "stop"}
    ]
    return [_redact(event) for event in useful[-limit:]]


def _endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM planner returned no JSON object")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM planner response must be a JSON object")
    return payload


def _completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM endpoint returned no choices")
    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        if parts:
            return "".join(parts)
    raise ValueError("LLM endpoint returned no text content")


def choose_with_llm(
    config: dict[str, Any],
    available: list[dict[str, Any]],
    official: dict[str, Any],
    targets: dict[str, Any],
    state_summary: dict[str, Any],
    memory_path: Path,
) -> dict[str, Any]:
    settings = config.get("llm_planner", {})
    api_key = _configured_value(settings, "api_key_env")
    base_url = _configured_value(settings, "base_url_env")
    model = _configured_value(settings, "model_env")
    if not api_key or not base_url or not model:
        raise RuntimeError("LLM planner environment is incomplete")

    experiment_catalog = [
        {
            "id": item["id"],
            "task": item.get("task"),
            "kind": item.get("kind"),
            "family": item.get("family", item.get("id")),
            "cost_seconds": item.get("cost_seconds"),
            "hypothesis": item.get("hypothesis", item.get("description", "")),
            "description": item.get("description", ""),
        }
        for item in available
    ]
    context = {
        "official_scores": _redact(official),
        "targets": targets,
        "local_state": _redact(state_summary),
        "registered_available_experiments": experiment_catalog,
        "recent_experiment_memory": _recent_memory(
            memory_path,
            int(settings.get("max_memory_events", 30)),
        ),
    }
    system = (
        "You are the planning component of a serial machine-learning research agent. "
        "Official leaderboard scores and local validation metrics are different. "
        "Choose exactly one registered experiment that best reduces the largest "
        "official target gap. Prefer a new falsifiable hypothesis over repeating a "
        "failed family. Require multi-split or OOF evidence and conservative artifact "
        "promotion. You may not invent commands, scripts, scores, labels, or experiment "
        "IDs. Return JSON only with keys experiment_id, reason, hypothesis, "
        "expected_failure_mode, and stop. stop must be false while targets are unmet "
        "and a registered experiment is available."
    )
    user = json.dumps(context, ensure_ascii=False, sort_keys=True)
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(settings.get("temperature", 0.0)),
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        _endpoint(base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=int(settings.get("timeout_seconds", 45)),
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM planner HTTP error {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("LLM planner connection failed") from exc

    decision = _extract_json(_completion_text(response_payload))
    experiment_id = str(decision.get("experiment_id", "")).strip()
    allowed = {str(item["id"]) for item in available}
    if experiment_id not in allowed:
        raise ValueError("LLM planner selected an unregistered experiment")
    if bool(decision.get("stop", False)):
        raise ValueError("LLM planner attempted to stop with available experiments")
    return {
        "experiment_id": experiment_id,
        "reason": str(decision.get("reason", "")).strip()[:1200],
        "hypothesis": str(decision.get("hypothesis", "")).strip()[:1200],
        "expected_failure_mode": str(
            decision.get("expected_failure_mode", "")
        ).strip()[:1200],
        "source": "llm",
        "model": model,
    }
