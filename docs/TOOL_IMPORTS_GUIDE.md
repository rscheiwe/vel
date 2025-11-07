# Tool Imports Quick Reference

**TL;DR:** Import your tool module before creating the agent. The `register_tool()` call happens automatically when the module is imported.

## Basic Pattern

```python
# Step 1: Define tool in separate file
# File: tools/my_tool.py
from vel import ToolSpec, register_tool

my_tool = ToolSpec(
    name='my_tool',
    input_schema={...},
    output_schema={...},
    handler=my_handler
)
register_tool(my_tool)  # Registers globally

# Step 2: Import tool in agent file
# File: my_agent.py
from vel import Agent
from tools.my_tool import my_tool  # This registers the tool

# Step 3: Use tool by name
agent = Agent(
    id='my-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['my_tool']  # Available because we imported it
)
```

## How It Works

1. **Global Registry**: Vel maintains a global tool registry (like a dictionary)
2. **Import = Register**: When you import a tool module, `register_tool()` runs automatically
3. **Reference by Name**: Agent references tools by their string name

```
┌─────────────────────────────────────────┐
│ Global Tool Registry (Singleton)        │
├─────────────────────────────────────────┤
│ 'my_tool'     → ToolSpec(...)          │
│ 'websearch'   → ToolSpec(...)          │
│ 'get_weather' → ToolSpec(...)          │
└─────────────────────────────────────────┘
         ↑                    ↓
    register_tool()     Agent(tools=['my_tool'])
```

## Three Organization Patterns

### Pattern 1: Inline (Simple)

```python
# my_agent.py
from vel import Agent, ToolSpec, register_tool

# Define + register in same file
register_tool(ToolSpec(
    name='my_tool',
    input_schema={...},
    output_schema={...},
    handler=lambda inp, ctx: {...}
))

agent = Agent(tools=['my_tool'])
```

**Use when:** Quick prototype, 1-3 tools

### Pattern 2: Separate Module (Recommended)

```
project/
├── tools/
│   ├── __init__.py        # Import all tools
│   ├── weather.py         # register_tool(weather_tool)
│   └── search.py          # register_tool(search_tool)
└── my_agent.py            # import tools
```

```python
# tools/__init__.py
from .weather import weather_tool
from .search import search_tool

# my_agent.py
from vel import Agent
import tools  # Registers all tools

agent = Agent(tools=['get_weather', 'search'])
```

**Use when:** Production app, 4+ tools, team project

### Pattern 3: Conditional Registration

```python
# tools/optional_tool.py
import os
from vel import ToolSpec, register_tool

if os.getenv('API_KEY'):
    register_tool(ToolSpec(...))
```

**Use when:** Tool requires external dependencies or API keys

## Common Mistakes

### ❌ Mistake 1: Forgot to Import

```python
# my_agent.py
from vel import Agent

agent = Agent(tools=['my_tool'])  # KeyError: 'my_tool' not found!
```

**Fix:** Import the tool module first
```python
from vel import Agent
from tools.my_tool import my_tool  # Import registers it

agent = Agent(tools=['my_tool'])  # ✓ Works!
```

### ❌ Mistake 2: Wrong Tool Name

```python
# tools/my_tool.py
register_tool(ToolSpec(
    name='my_tool',  # Name is 'my_tool'
    ...
))

# my_agent.py
agent = Agent(tools=['mytool'])  # ❌ KeyError: 'mytool' not found!
```

**Fix:** Use exact name from ToolSpec
```python
agent = Agent(tools=['my_tool'])  # ✓ Matches ToolSpec.name
```

### ❌ Mistake 3: Import After Agent Creation

```python
agent = Agent(tools=['my_tool'])  # ❌ Tool not registered yet!
from tools.my_tool import my_tool
```

**Fix:** Import at top of file
```python
from tools.my_tool import my_tool  # ✓ Import first
agent = Agent(tools=['my_tool'])
```

## Real Example: Perplexity Web Search

```python
# examples/multi_step_tools/web_search.py
from vel import ToolSpec, register_tool

web_search_tool = ToolSpec(
    name='websearch',
    input_schema={
        'type': 'object',
        'properties': {
            'query': {'type': 'string'},
            'limit': {'type': 'number', 'default': 5}
        },
        'required': ['query']
    },
    output_schema={
        'type': 'object',
        'properties': {
            'state': {'type': 'string'},
            'results': {'type': 'array'}
        },
        'required': ['state', 'results']
    },
    handler=web_search_handler
)
register_tool(web_search_tool)

# examples/perplexity_web_search_example.py
from vel import Agent
from examples.multi_step_tools.web_search import web_search_tool  # Import

agent = Agent(
    id='research-agent',
    model={'provider': 'openai', 'model': 'gpt-4o'},
    tools=['websearch']  # Use by name
)

result = await agent.run({'message': 'Search for AI trends'})
```

## Key Takeaways

1. ✅ **Import before agent creation**
2. ✅ **Tools registered globally via `register_tool()`**
3. ✅ **Import tool module to register it**
4. ✅ **Reference tools by name string in `tools=[]`**
5. ✅ **All agents share the same tool registry**

## See Also

- [Complete Tools Documentation](tools.md) - Full tool system guide
- [examples/perplexity_web_search_example.py](../examples/perplexity_web_search_example.py) - Real-world example
- [examples/multi_step_tools/](../examples/multi_step_tools/) - Multiple tool examples
