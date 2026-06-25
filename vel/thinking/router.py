"""Auto-routing for Vel extended thinking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ThinkingRouteDecision:
    """Decision returned by the thinking router."""

    mode: str
    reason: str
    effort: str
    confidence: float
    category: str
    raw_mode: str


def normalize_effort(value: str | None, default: str = "high") -> str:
    """Normalize requested thinking effort."""

    effort = (value or default).strip().lower()
    return effort if effort in {"low", "medium", "high", "extra", "max"} else default


def effort_overrides(effort: str) -> dict[str, Any]:
    """Map effort labels to conservative reflection loop defaults."""

    normalized = normalize_effort(effort)
    if normalized == "low":
        return {"max_refinements": 1, "confidence_threshold": 0.65}
    if normalized == "medium":
        return {"max_refinements": 2, "confidence_threshold": 0.75}
    if normalized == "extra":
        return {"max_refinements": 4, "confidence_threshold": 0.88}
    if normalized == "max":
        return {"max_refinements": 5, "confidence_threshold": 0.92}
    return {"max_refinements": 3, "confidence_threshold": 0.8}


def _event_type(event: Any) -> str:
    return event.type if hasattr(event, "type") else str(event.get("type", ""))


def _event_delta(event: Any) -> str:
    return event.delta if hasattr(event, "delta") else str(event.get("delta", ""))


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


async def route_thinking(
    *,
    provider: Any,
    model: str,
    message: str,
    context: Optional[List[Dict[str, Any]]] = None,
    effort: str = "high",
    confidence_threshold: float = 0.8,
) -> ThinkingRouteDecision:
    """Classify whether a request should use reflection before answering."""

    prompt = (
        "You are a routing classifier for an AI assistant. Decide whether the "
        "user request should be answered directly or should use an extended "
        "reflection pass first. Return only JSON with keys: mode, confidence, "
        "category, reason. mode must be direct or reflection. confidence must be "
        "a number from 0 to 1. Default to direct unless reflection is clearly "
        "needed. The selected effort is only a depth setting after reflection is "
        "chosen; it must not make a simple request more likely to reflect. Choose "
        "direct for acknowledgements, thanks, greetings, simple follow-ups, "
        "status/listing tasks, short factual questions, or anything that can be "
        "answered without multi-step deliberation. A thank-you after a complex "
        "answer is still direct unless it asks a new analytical question. Choose "
        "reflection only for multi-step design, tradeoff analysis, planning, "
        "modeling, validation repair, or high-impact ambiguous requests."
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "selected_effort": normalize_effort(effort),
                    "message": message,
                    "context_messages": len(context or []),
                }
            ),
        },
    ]

    chunks: list[str] = []
    async for event in provider.stream(
        messages=messages,
        model=model,
        tools={},
        generation_config={"temperature": 0},
    ):
        if _event_type(event) == "text-delta":
            chunks.append(_event_delta(event))

    payload = _parse_json_object("".join(chunks).strip())
    raw_mode = str(payload.get("mode") or "direct").strip().lower()
    confidence = _normalize_confidence(payload.get("confidence"))
    mode = raw_mode if raw_mode in {"direct", "reflection"} else "direct"
    if mode == "reflection" and confidence < confidence_threshold:
        mode = "direct"
    return ThinkingRouteDecision(
        mode=mode,
        reason=str(payload.get("reason") or "No router reason returned.").strip(),
        effort=normalize_effort(effort),
        confidence=confidence,
        category=str(payload.get("category") or "unknown").strip() or "unknown",
        raw_mode=raw_mode,
    )
