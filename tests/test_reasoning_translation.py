"""Chat-completions reasoning ingest — field spellings, and the both-in-one-chunk case.

`translators.py` had no test coverage at all, and it is the code that decides
protocol conformance. These pin the three things that were actually broken:

1. Only `reasoning_content` was read, so every OpenAI-*compatible* endpoint that
   spells it `reasoning` (OpenRouter, and vLLM/Groq/Together in some versions)
   dropped its entire reasoning trace silently — a 200 with no trace and no
   diagnostic.
2. `translate_chunk` returns at most one event, and the text branch returned
   first, so reasoning arriving in the same chunk as content was discarded.
3. The reasoning block was only closed on `finish_reason`, so it stayed open for
   the whole answer and clients rendered the last reasoning step as still
   running underneath finished text.

The chunk shapes here are copied from a real OpenRouter response, not invented.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from vel.providers.translators import OpenAIAPITranslator, _reasoning_delta


def _drain(translator: OpenAIAPITranslator, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every event a chunk produces, in order — the returned one plus the queue.

    Mirrors how `OpenAIProvider.stream` consumes the translator: take the return
    value, then drain `get_pending_event()` until empty.
    """
    events = []
    first = translator.translate_chunk(chunk)
    if first is not None:
        events.append(first.to_dict())
    while True:
        pending = translator.get_pending_event()
        if pending is None:
            break
        events.append(pending.to_dict())
    return events


def _chunk(delta: Dict[str, Any]) -> Dict[str, Any]:
    return {'id': 'c1', 'choices': [{'delta': delta, 'finish_reason': None}]}


# --------------------------------------------------------------------------
# 1. Field spellings
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    'delta,expected',
    [
        ({'reasoning_content': 'A'}, 'A'),
        ({'reasoning': 'B'}, 'B'),
        # Flat field wins over the structured duplicate, so the same text is
        # never emitted twice — OpenRouter sends both.
        ({'reasoning': 'B', 'reasoning_details': [{'type': 'reasoning.text', 'text': 'B'}]}, 'B'),
        ({'reasoning_details': [{'type': 'reasoning.text', 'text': 'C'}]}, 'C'),
        ({'reasoning_content': 'A', 'reasoning': 'B'}, 'A'),
        ({'content': 'hi'}, ''),
        ({'reasoning': ''}, ''),
        ({'reasoning_details': 'malformed'}, ''),
    ],
)
def test_reasoning_delta_field_aliases(delta, expected):
    assert _reasoning_delta(delta) == expected


def test_openrouter_shaped_chunk_produces_reasoning():
    """The exact delta OpenRouter sends, verbatim from a live response.

    Note `content: ""` riding along — an empty string is not text, and treating
    it as such would open a spurious text block.
    """
    translator = OpenAIAPITranslator()
    events = _drain(translator, _chunk({
        'content': '',
        'role': 'assistant',
        'reasoning': 'We',
        'reasoning_details': [{'type': 'reasoning.text', 'text': 'We', 'format': 'unknown', 'index': 0}],
    }))

    assert [e['type'] for e in events] == ['reasoning-start', 'reasoning-delta']
    assert events[1]['delta'] == 'We'


def test_deepseek_direct_spelling_still_works():
    """Regression guard: broadening the aliases must not break the original."""
    translator = OpenAIAPITranslator()
    events = _drain(translator, _chunk({'reasoning_content': 'thinking'}))
    assert [e['type'] for e in events] == ['reasoning-start', 'reasoning-delta']
    assert events[1]['delta'] == 'thinking'


# --------------------------------------------------------------------------
# 2. Reasoning and text in one chunk
# --------------------------------------------------------------------------

def test_reasoning_and_content_in_same_chunk_both_survive():
    """Previously the text branch returned first and the reasoning was lost."""
    translator = OpenAIAPITranslator()
    events = _drain(translator, _chunk({'reasoning': 'think', 'content': 'say'}))

    assert [e['type'] for e in events] == [
        'reasoning-start', 'reasoning-delta', 'reasoning-end', 'text-start', 'text-delta',
    ]
    assert events[1]['delta'] == 'think'
    assert events[4]['delta'] == 'say'


def test_text_only_chunk_is_unchanged():
    translator = OpenAIAPITranslator()
    events = _drain(translator, _chunk({'content': 'hello'}))
    assert [e['type'] for e in events] == ['text-start', 'text-delta']
    assert events[1]['delta'] == 'hello'


def test_empty_content_opens_no_text_block():
    translator = OpenAIAPITranslator()
    assert _drain(translator, _chunk({'content': ''})) == []


# --------------------------------------------------------------------------
# 3. The reasoning block closes before the answer starts
# --------------------------------------------------------------------------

def test_reasoning_closes_when_text_begins():
    translator = OpenAIAPITranslator()
    events = _drain(translator, _chunk({'reasoning': 'thinking...'}))
    events += _drain(translator, _chunk({'content': 'answer'}))

    types = [e['type'] for e in events]
    assert types.index('reasoning-end') < types.index('text-start')
    # One block, opened once and closed once.
    assert types.count('reasoning-start') == 1
    assert types.count('reasoning-end') == 1


def test_reasoning_after_text_opens_a_second_block():
    """A model that returns to reasoning gets a new block, not a reopened one."""
    translator = OpenAIAPITranslator()
    events = _drain(translator, _chunk({'reasoning': 'first'}))
    events += _drain(translator, _chunk({'content': 'partial'}))
    events += _drain(translator, _chunk({'reasoning': 'second'}))

    types = [e['type'] for e in events]
    assert types.count('reasoning-start') == 2
    reasoning_ids = {e['id'] for e in events if e['type'] == 'reasoning-start'}
    assert len(reasoning_ids) == 2


def test_finish_still_closes_an_open_reasoning_block():
    """Reasoning with no text at all must still be closed by finish_reason."""
    translator = OpenAIAPITranslator()
    events = _drain(translator, _chunk({'reasoning': 'thinking'}))
    events += _drain(translator, {'id': 'c1', 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})

    types = [e['type'] for e in events]
    assert types.count('reasoning-start') == types.count('reasoning-end') == 1
