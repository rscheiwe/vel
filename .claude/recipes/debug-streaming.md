# Debug Streaming Recipe

**Goal:** Diagnose and fix streaming issues in Vel
**Prerequisites:** Reproducible issue, basic understanding of event flow
**Estimated Time:** 30 min - 2 hours

---

## Steps

### Step 1: Identify Symptom

| Symptom | Likely Cause | Jump To |
|---------|--------------|---------|
| Stream hangs | Async issue, missing yield | [Hang Diagnosis](#step-2a-hang-diagnosis) |
| Events out of order | Translator state bug | [Event Order](#step-2b-event-order-diagnosis) |
| Missing events | Early termination, filter | [Missing Events](#step-2c-missing-events-diagnosis) |
| Duplicate events | State not reset | [Duplicates](#step-2d-duplicate-diagnosis) |
| Malformed events | Schema/serialization | [Malformed](#step-2e-malformed-diagnosis) |

---

### Step 2a: Hang Diagnosis

**Symptom:** Stream starts but never completes.

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

async for event in agent.run_stream(input):
    print(f"EVENT: {event['type']}")  # See where it stops
```

**Common Causes:**

1. **Missing yield in async generator**
   ```python
   # WRONG - no yield
   async def stream(...):
       response = await api.call()
       return response

   # CORRECT
   async def stream(...):
       async for chunk in api.stream():
           yield translate(chunk)
   ```

2. **Blocking call in async code**
   ```python
   # WRONG - blocks event loop
   result = requests.get(url)

   # CORRECT
   result = await aiohttp.get(url)
   ```

3. **Unhandled exception swallowed**
   ```python
   # WRONG
   try:
       async for chunk in stream:
           yield process(chunk)
   except:
       pass  # Silently fails

   # CORRECT
   try:
       async for chunk in stream:
           yield process(chunk)
   except Exception as e:
       yield {'type': 'error', 'message': str(e)}
       raise
   ```

---

### Step 2b: Event Order Diagnosis

**Symptom:** Events arrive in wrong sequence.

**Expected Order:**
```
text-start → text-delta* → text-end
tool-input-start → tool-input-delta* → tool-input-available → tool-output-available
finish-message
```

**Debug Approach:**

```python
# Capture and analyze event sequence
events = []
async for event in agent.run_stream(input):
    events.append(event)

# Check sequence
types = [e['type'] for e in events]
print(types)

# Verify tool call sequence
tool_events = [e for e in events if 'tool' in e['type']]
for i, e in enumerate(tool_events):
    print(f"{i}: {e['type']} - {e.get('tool_name', 'N/A')}")
```

**Common Causes:**

1. **Translator not buffering tool calls**
   ```python
   # WRONG - emit before complete
   for tc in chunk.tool_calls:
       yield {'type': 'tool-input-available', ...}

   # CORRECT - buffer until complete
   if tc.id not in self._buffer:
       self._buffer[tc.id] = {'name': tc.function.name, 'args': ''}
   self._buffer[tc.id]['args'] += tc.function.arguments
   if is_complete(tc):
       yield {'type': 'tool-input-available', ...}
   ```

2. **State not tracked across chunks**
   ```python
   class Translator:
       def __init__(self):
           self._text_started = False  # Track state!
   ```

---

### Step 2c: Missing Events Diagnosis

**Symptom:** Some events never arrive.

**Debug Approach:**

```python
# Log at provider level
async for chunk in provider.stream(...):
    print(f"RAW CHUNK: {chunk}")
    for event in translator.translate(chunk):
        print(f"TRANSLATED: {event}")
        yield event
```

**Common Causes:**

1. **Early return in loop**
   ```python
   # WRONG
   async for chunk in stream:
       if chunk.finish_reason:
           return  # Misses remaining events
       yield process(chunk)

   # CORRECT
   async for chunk in stream:
       yield process(chunk)
       if chunk.finish_reason:
           break
   ```

2. **Filter too aggressive**
   ```python
   # WRONG
   if event['type'] in ['text-delta']:  # Only text!
       yield event

   # CORRECT
   yield event  # Pass all through
   ```

3. **Guardrail blocking**
   ```python
   # Check guardrails
   agent = Agent(guardrails={'output': []})  # Disable temporarily
   ```

---

### Step 2d: Duplicate Diagnosis

**Symptom:** Same event appears multiple times.

**Debug Approach:**

```python
# Track event IDs
seen = set()
async for event in agent.run_stream(input):
    key = (event['type'], event.get('tool_name'), event.get('delta', '')[:20])
    if key in seen:
        print(f"DUPLICATE: {event}")
    seen.add(key)
```

**Common Causes:**

1. **Translator state not reset**
   ```python
   # WRONG - reusing translator
   translator = Translator()
   for run in runs:
       for event in translator.translate(chunk):  # State leaks!

   # CORRECT - new translator per stream
   translator = Translator()  # Fresh each time
   ```

2. **Event emitted in multiple places**
   ```python
   # Check for duplicate yields
   if text:
       yield text_event
   # ... later ...
   if text:  # Oops, yields again!
       yield text_event
   ```

---

### Step 2e: Malformed Diagnosis

**Symptom:** Events have wrong structure or missing fields.

**Debug Approach:**

```python
# Validate against schema
from vel.events import validate_event

async for event in agent.run_stream(input):
    errors = validate_event(event)
    if errors:
        print(f"INVALID: {event} - {errors}")
```

**Common Causes:**

1. **Missing required fields**
   ```python
   # WRONG
   yield {'type': 'text-delta'}  # Missing 'delta'

   # CORRECT
   yield {'type': 'text-delta', 'delta': text}
   ```

2. **Wrong field types**
   ```python
   # WRONG
   yield {'type': 'finish-message', 'finish_reason': 0}  # Should be string

   # CORRECT
   yield {'type': 'finish-message', 'finish_reason': 'stop'}
   ```

---

## Validation

After fixing, verify with:

```python
# Full stream capture
events = []
async for event in agent.run_stream({'message': 'test'}):
    events.append(event)

# Assertions
assert events[0]['type'] in ['start', 'text-start']
assert events[-1]['type'] in ['finish', 'finish-message']
assert all('type' in e for e in events)

# Run tests
pytest tests/test_events.py -v
```

---

## Quick Reference

### Event Type Checklist

| Event | Required Fields | Emitted When |
|-------|-----------------|--------------|
| `text-start` | - | Text block begins |
| `text-delta` | `delta` | Text content available |
| `text-end` | - | Text block completes |
| `tool-input-start` | `tool_name` | Tool call begins |
| `tool-input-delta` | `delta` | Tool input streaming |
| `tool-input-available` | `tool_name`, `input` | Tool input complete |
| `tool-output-available` | `output` | Tool executed |
| `finish-message` | `finish_reason` | Response complete |
| `error` | `message` | Error occurred |

### Debug Logging Snippet

```python
import logging

# Enable debug for Vel
logging.getLogger('vel').setLevel(logging.DEBUG)
logging.getLogger('vel.providers').setLevel(logging.DEBUG)

# Handler to see output
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logging.getLogger('vel').addHandler(handler)
```
