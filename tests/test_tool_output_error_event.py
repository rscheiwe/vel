"""Wire shape of `tool-output-error`, guarded the same way `error` is.

Mirrors tests/test_error_event.py, and for the same reason: the AI SDK UI
Message Stream is parsed as a strict Zod union on the client, so any extra
top-level key fails with `unrecognized_keys -> invalid_union` and the reader
sees "Type validation failed" instead of the error that actually occurred.

The tempting extra key here is `toolName`. It is deliberately absent — the
preceding `tool-input-available` already carries it, and adding it would break
every consumer.

Verified against both installed clients: ai@5.0.28 (sophee-ui) and ai@6.0.149
(ontql-ui) both define this part as {type, toolCallId, errorText} plus optional
providerExecuted / providerMetadata / dynamic.
"""
from __future__ import annotations

from vel.events import ToolOutputErrorEvent


def test_wire_shape_is_minimal():
    d = ToolOutputErrorEvent(tool_call_id='call_1', error_text='boom').to_dict()
    assert set(d.keys()) == {'type', 'toolCallId', 'errorText'}


def test_wire_values():
    d = ToolOutputErrorEvent(tool_call_id='call_1', error_text='boom').to_dict()
    assert d['type'] == 'tool-output-error'
    assert d['toolCallId'] == 'call_1'
    assert d['errorText'] == 'boom'


def test_camel_case_keys_not_snake_case():
    """`tool_call_id` on the dataclass, `toolCallId` on the wire."""
    d = ToolOutputErrorEvent(tool_call_id='c', error_text='e').to_dict()
    assert 'tool_call_id' not in d
    assert 'error_text' not in d


def test_defaults_still_produce_a_valid_shape():
    d = ToolOutputErrorEvent().to_dict()
    assert set(d.keys()) == {'type', 'toolCallId', 'errorText'}
