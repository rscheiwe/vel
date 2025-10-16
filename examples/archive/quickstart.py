import asyncio
from dotenv import load_dotenv
from vel import Agent, run_stream, ToolSpec, register_tool

# Load environment variables from .env file
load_dotenv()

# Register a simple weather tool
register_tool(ToolSpec(
    name='get_weather',
    input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
    output_schema={'type': 'object', 'properties': {'temp_f': {'type': 'number'}}, 'required': ['temp_f']},
    handler=lambda inp, ctx: {'temp_f': 72.0}
))

async def main():
    agent = Agent(id='chat-general:v1', model={'provider':'openai','model':'gpt-4o'}, tools=['get_weather'], policies={'max_steps':8})
    async for e in run_stream(agent, {'message':'weather please'}):
        print(e)

if __name__=='__main__':
    asyncio.run(main())
