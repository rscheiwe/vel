import asyncio
from dotenv import load_dotenv
from agents import Agent, run_stream

# Load environment variables from .env file
load_dotenv()

async def main():
    agent = Agent(id='chat-general:v1', model={'provider':'openai','model':'gpt-4o'}, tools=['get_weather'], policies={'max_steps':8})
    async for e in run_stream(agent, {'message':'weather please'}):
        print(e)

if __name__=='__main__':
    asyncio.run(main())
