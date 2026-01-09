---
description: "Run tests for a specific module or all tests"
argument-hint: "[path]"
allowed-tools: Bash(pytest:*)
---

Run pytest for the Vel codebase.

## Instructions

1. If `$1` is provided, run tests for that specific path:
   ```bash
   pytest $1 -v
   ```

2. If no argument provided, run all tests:
   ```bash
   pytest tests/ -v
   ```

3. Show test output with:
   - `-v` for verbose mode
   - Full tracebacks on failures
   - Test duration for slow tests

4. If tests fail, analyze the failures and suggest fixes.

## Common Paths

- `tests/test_memory.py` - Memory system tests
- `tests/test_agent.py` - Core agent tests
- `tests/test_rlm.py` - RLM tests
- `tests/test_thinking.py` - Extended thinking tests
- `tests/test_tools.py` - Tool system tests

## Example Usage

```
/test tests/test_memory.py
/test tests/test_agent.py -k "test_stream"
/test
```
