"""
Which generation_config keys reach the wire.

The payload builder is a CLOSED whitelist — no `**config` spread — which is the
right default, but it means an unlisted key is dropped in total silence: the
caller gets a normal 200 with none of the behaviour they asked for. There is no
error to notice and nothing in the response that says the request was altered.

So the whitelist is worth pinning explicitly, in both directions: the keys that
must survive, and the fact that arbitrary keys must not.
"""

import json

import pytest

from vel.providers.openai import OpenAIProvider


class _Capture:
    """Stands in for httpx, recording the JSON body the provider builds."""

    def __init__(self):
        self.payload = None

    def __call__(self, *args, **kwargs):
        self.payload = kwargs.get("json")
        raise _Stop()


class _Stop(Exception):
    pass


async def _payload_for(config):
    """Build a request and capture its body without sending it."""
    import httpx

    provider = OpenAIProvider(api_key="test", base_url="https://openrouter.ai/api/v1")
    capture = _Capture()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            return capture(url, **kwargs)

        def stream(self, method, url, **kwargs):
            return capture(url, **kwargs)

    original = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **k: FakeClient()
    try:
        with pytest.raises(_Stop):
            await provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                model="deepseek/deepseek-v4-pro",
                tools=None,
                generation_config=config,
            )
    finally:
        httpx.AsyncClient = original
    return capture.payload


@pytest.mark.asyncio
async def test_provider_routing_preference_reaches_the_wire():
    """OpenRouter's `provider` key is the only lever over routing latency.

    OpenRouter load-balances one model across backing providers whose
    time-to-first-token differs by an order of magnitude — measured on
    deepseek-v4-pro with an identical one-word prompt, default routing gave 1.4s
    and 9.8s on consecutive calls, while `{"sort": "latency"}` gave 1.2s and
    0.66s. Dropping this key silently leaves a caller with no way to avoid the
    slow route and no clue why their agent is slow.
    """
    payload = await _payload_for({"provider": {"sort": "latency"}})

    assert payload["provider"] == {"sort": "latency"}


@pytest.mark.asyncio
async def test_reasoning_controls_still_reach_the_wire():
    payload = await _payload_for(
        {"reasoning": {"effort": "low"}, "include_reasoning": True}
    )

    assert payload["reasoning"] == {"effort": "low"}
    assert payload["include_reasoning"] is True


@pytest.mark.asyncio
async def test_the_whitelist_is_still_closed():
    # The point of the whitelist is that it does NOT forward everything. If this
    # ever passes something through, the guarantee is gone.
    payload = await _payload_for({"totally_made_up_key": "value"})

    assert "totally_made_up_key" not in payload
