# run: python scripts/vel_trace.py --prompt "Hello" > traces/vel.jsonl
import os, sys, json, asyncio, argparse
from dotenv import load_dotenv
from vel import Agent

# Load environment variables from .env file
load_dotenv()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="gpt-4o", help="Model to use")
    args = ap.parse_args()

    # Get API key from environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Create agent (same as AI SDK streamText approach)
    agent = Agent(
        id='trace-agent:v1',
        model={'provider': 'openai', 'model': args.model},
        policies={'max_steps': 1}
    )

    # Stream events and output raw JSON
    async for event in agent.run_stream({'message': args.prompt}):
        print(json.dumps(event, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
