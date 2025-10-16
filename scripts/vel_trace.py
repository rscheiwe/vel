# run: PYTHONPATH=. VEL_PROVIDER=openai python scripts/vel_trace.py --prompt "Hello"
# run: PYTHONPATH=. VEL_RESPONSES=1 python scripts/vel_trace.py --prompt "Hello"  # For Responses API
import os, sys, json, asyncio, argparse
import httpx
from vel.providers.translators import OpenAIAPITranslator, OpenAIResponsesAPITranslator
# choose your translator (chat or responses) via env/flag
IS_RESP = os.getenv("VEL_RESPONSES") == "1"

def norm(ev):
    t = ev["type"]
    out = {"t": t}
    if "id" in ev: out["id"] = ev["id"]
    if t in ("text-delta","reasoning-delta"): out["d"] = ev.get("delta") or ev.get("text") or ""
    if t == "tool-input-available":
        out["tool"] = ev.get("toolName"); out["args"] = ev.get("input")
    if t == "tool-output-available":
        out["tool"] = ev.get("toolName"); out["out"] = ev.get("output")
    return out

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="gpt-4o", help="Model to use")
    args = ap.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async def provider_lines():
        """Call OpenAI API and yield parsed events/chunks"""
        if IS_RESP:
            # Responses API (/v1/responses)
            endpoint = "https://api.openai.com/v1/responses"
            payload = {
                "model": args.model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": args.prompt}]}],
                "stream": True
            }
        else:
            # Chat Completions API (/v1/chat/completions)
            endpoint = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": args.prompt}],
                "stream": True
            }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            chunk = json.loads(data_str)
                            yield chunk
                        except json.JSONDecodeError:
                            continue

    translator = OpenAIResponsesAPITranslator() if IS_RESP else OpenAIAPITranslator()
    translator.reset()

    async for e in provider_lines():
        vel_ev = (translator.translate_event(e) if IS_RESP
                  else translator.translate_chunk(e))
        if vel_ev:
            print(json.dumps(norm(vel_ev.to_dict()), ensure_ascii=False))

        # Drain any pending events from translator
        while True:
            pending = translator.get_pending_event()
            if pending is None:
                break
            print(json.dumps(norm(pending.to_dict()), ensure_ascii=False))

    # Finalize (for Chat Completions, emit tool-input-available if needed)
    if not IS_RESP:
        for final_ev in translator.finalize_tool_calls():
            print(json.dumps(norm(final_ev.to_dict()), ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
