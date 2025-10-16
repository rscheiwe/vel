"""
RLM Tools

Tool implementations for context probing and recursive calls.
These tools are RLM-internal and NOT registered in Vel's ToolRegistry.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from .context_store import ContextStore
from .utils import truncate_text


def context_probe(
    context_store: ContextStore,
    kind: str,
    query: Optional[str] = None,
    id: Optional[str] = None,
    max_bytes: Optional[int] = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Probe the context store.

    Args:
        context_store: ContextStore instance
        kind: Operation kind ('search', 'read', 'summarize')
        query: Search query (for 'search' kind)
        id: Chunk ID (for 'read' or 'summarize' kind)
        max_bytes: Maximum bytes to return
        max_results: Maximum search results (for 'search' kind)

    Returns:
        Tool result with preview, metadata, and truncation info
    """
    max_bytes = max_bytes or 4096

    try:
        if kind == 'search':
            if not query:
                return {
                    'error': 'Missing required parameter: query',
                    'preview': '',
                    'meta': {'truncated': False}
                }

            # Search context
            results = context_store.search(query, max_results=max_results)

            # Format results
            if not results:
                preview = f"No results found for query: {query}"
                truncated = False
            else:
                lines = [f"Found {len(results)} results:"]
                for i, result in enumerate(results, 1):
                    chunk_id = result['chunk_id']
                    snippet = result['snippet'].replace('\n', ' ')[:200]
                    lines.append(f"{i}. [{chunk_id}] {snippet}...")

                preview = '\n'.join(lines)
                preview, truncated = truncate_text(preview, max_bytes)

            return {
                'preview': preview,
                'meta': {
                    'kind': 'search',
                    'query': query,
                    'num_results': len(results),
                    'truncated': truncated
                }
            }

        elif kind == 'read':
            if not id:
                return {
                    'error': 'Missing required parameter: id',
                    'preview': '',
                    'meta': {'truncated': False}
                }

            # Read chunk
            chunk_data = context_store.read(id, max_bytes=max_bytes)

            if not chunk_data:
                return {
                    'error': f'Chunk not found: {id}',
                    'preview': '',
                    'meta': {'truncated': False}
                }

            return {
                'preview': chunk_data['preview'],
                'meta': {
                    'kind': 'read',
                    'chunk_id': id,
                    'truncated': chunk_data['truncated'],
                    'full_size': chunk_data['full_size'],
                    **chunk_data.get('metadata', {})
                }
            }

        elif kind == 'summarize':
            if not id:
                return {
                    'error': 'Missing required parameter: id',
                    'preview': '',
                    'meta': {'truncated': False}
                }

            # Summarize chunk
            summary = context_store.summarize(id, max_length=max_bytes // 2)
            summary, truncated = truncate_text(summary, max_bytes)

            return {
                'preview': summary,
                'meta': {
                    'kind': 'summarize',
                    'chunk_id': id,
                    'truncated': truncated
                }
            }

        else:
            return {
                'error': f'Unknown kind: {kind}',
                'preview': '',
                'meta': {'truncated': False}
            }

    except Exception as e:
        return {
            'error': str(e),
            'preview': '',
            'meta': {'truncated': False}
        }


async def rlm_call(
    query: str,
    context_slice: Any,
    depth_left: int,
    controller: Any,  # RlmController instance (avoid circular import)
    agent: Any,  # Agent instance
    session_id: Optional[str]
) -> Dict[str, Any]:
    """
    Spawn recursive RLM call.

    Args:
        query: Question to answer recursively
        context_slice: Subset of context (chunk IDs, text, or refs)
        depth_left: Remaining recursion depth
        controller: RlmController instance
        agent: Agent instance
        session_id: Session ID

    Returns:
        Result with answer and notes from child execution
    """
    if depth_left <= 0:
        return {
            'error': 'Maximum recursion depth reached',
            'answer': '',
            'notes': []
        }

    try:
        # Create child controller with reduced depth
        from .controller import RlmController

        child_controller = RlmController(
            config=controller.config,
            agent=agent,
            depth=controller.depth - 1
        )

        # Run child execution
        result = await child_controller.run(
            user_query=query,
            context_refs=context_slice,
            session_id=session_id,
            parent_scratchpad=controller.scratchpad
        )

        return {
            'answer': result.get('answer', ''),
            'notes': [note.text for note in result.get('scratchpad_notes', [])],
            'meta': result.get('meta', {})
        }

    except Exception as e:
        return {
            'error': str(e),
            'answer': '',
            'notes': []
        }


def python_exec(
    code: str,
    context: Any,
    max_bytes: int = 4096,
    timeout: float = 0.5
) -> Dict[str, Any]:
    """
    Execute Python code in sandbox.

    WARNING: This is a security risk and should only be enabled in trusted environments.

    Args:
        code: Python code to execute
        context: CONTEXT variable to bind
        max_bytes: Maximum output bytes
        timeout: Execution timeout in seconds

    Returns:
        Tool result with stdout preview
    """
    # Create sandbox namespace
    namespace = {
        'CONTEXT': context,
        '__builtins__': {
            # Minimal builtins for safety
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'print': print,
        }
    }

    # Capture stdout/stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute code
            # Note: In production, use RestrictedPython or similar for true sandboxing
            exec(code, namespace)

        # Get output
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"

        # Truncate
        output, truncated = truncate_text(output, max_bytes)

        return {
            'preview': output,
            'meta': {
                'kind': 'python_exec',
                'truncated': truncated,
                'n_bytes': len(output)
            }
        }

    except Exception as e:
        error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
        return {
            'error': error_msg,
            'preview': stderr_capture.getvalue() or '',
            'meta': {'truncated': False}
        }


def get_tool_schemas() -> list[Dict[str, Any]]:
    """
    Get JSON schemas for RLM tools.

    Returns:
        List of tool schemas in OpenAI function calling format
    """
    return [
        {
            'name': 'context_probe',
            'description': 'Probe the large context with search, read, or summarize operations',
            'parameters': {
                'type': 'object',
                'properties': {
                    'kind': {
                        'type': 'string',
                        'enum': ['search', 'read', 'summarize'],
                        'description': 'Operation kind'
                    },
                    'query': {
                        'type': 'string',
                        'description': 'Search query (required for kind=search)'
                    },
                    'id': {
                        'type': 'string',
                        'description': 'Chunk ID (required for kind=read or summarize)'
                    },
                    'max_bytes': {
                        'type': 'integer',
                        'description': 'Maximum bytes to return (default 4096)'
                    },
                    'max_results': {
                        'type': 'integer',
                        'description': 'Maximum search results (default 10, for kind=search)'
                    }
                },
                'required': ['kind']
            }
        },
        {
            'name': 'rlm_call',
            'description': 'Spawn a recursive RLM call for a sub-question',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Question to answer recursively'
                    },
                    'context_slice': {
                        'type': 'string',
                        'description': 'Subset of context (chunk IDs or text)'
                    }
                },
                'required': ['query', 'context_slice']
            }
        },
        {
            'name': 'python_exec',
            'description': 'Execute Python code in sandbox (CONTEXT variable is pre-bound)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Python code to execute'
                    }
                },
                'required': ['code']
            }
        }
    ]
