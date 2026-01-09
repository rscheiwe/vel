---
description: "Scaffold a new LLM provider"
argument-hint: "[provider-name]"
---

Create a new LLM provider implementation for Vel.

## Instructions

1. Create the provider file at `vel/providers/$1.py` following the BaseProvider interface

2. Include these required components:

   **Provider Class:**
   ```python
   from vel.providers.base import BaseProvider

   class ${1^}Provider(BaseProvider):
       def __init__(self, api_key: Optional[str] = None):
           self.api_key = api_key or os.getenv('${1^^}_API_KEY')

       async def stream(self, messages, model, tools=None, **kwargs):
           # Implement streaming
           ...

       async def generate(self, messages, model, **kwargs):
           # Implement non-streaming
           ...
   ```

   **Translator Class:**
   ```python
   class ${1^}StreamTranslator:
       def translate(self, chunk) -> List[StreamEvent]:
           # Convert native events to Vel events
           ...
   ```

3. Add message format handling in `vel/providers/message_translator.py`

4. Register in `vel/providers/__init__.py`:
   ```python
   try:
       from .${1} import ${1^}Provider
       self._providers['${1}'] = ${1^}Provider()
   except (ImportError, ValueError):
       pass
   ```

5. Create tests in `tests/test_providers/test_${1}.py`

## Reference

See existing implementations:
- `vel/providers/openai.py` - OpenAI pattern
- `vel/providers/anthropic.py` - Anthropic pattern
- `vel/providers/google.py` - Gemini pattern
- ADR-003 for architecture decisions
