"""ErrorEvent wire shape must be AI-SDK-conformant: {type, errorText} only.

Extra top-level keys (statusCode/provider/errorCode/details/…) break the AI SDK
UI Message Stream strict Zod union on the client (unrecognized_keys ->
invalid_union), turning a transient provider error (e.g. an OpenAI 429) into a
cryptic "Type validation failed" wall of text. Regression guard.
"""
from vel.events import ErrorEvent


def test_wire_shape_is_minimal():
    d = ErrorEvent(
        error="boom", status_code=429, provider="openai",
        error_code="rate_limit", error_type="api", details={"raw": "x"},
    ).to_dict()
    assert set(d.keys()) == {"type", "errorText"}
    assert d["type"] == "error"


def test_context_folded_into_text():
    d = ErrorEvent(error="boom", status_code=429, provider="openai").to_dict()
    assert "429" in d["errorText"] and "openai" in d["errorText"] and "boom" in d["errorText"]
    assert set(d.keys()) == {"type", "errorText"}


def test_no_double_tagging_when_message_already_has_context():
    # The real 429 case: the provider already formats status into the message.
    d = ErrorEvent(
        error="OpenAI API error: HTTP 429", status_code=429, provider="openai",
    ).to_dict()
    assert d["errorText"] == "OpenAI API error: HTTP 429"  # unchanged, not doubled
    assert set(d.keys()) == {"type", "errorText"}


def test_plain_error_unchanged():
    assert ErrorEvent(error="plain").to_dict() == {"type": "error", "errorText": "plain"}
