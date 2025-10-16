"""
Utility functions for RLM

Helper functions for parsing tool calls, detecting FINAL(), extracting citations, etc.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple


def detect_final(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect FINAL() or FINAL_VAR() in text.

    Args:
        text: Text to search

    Returns:
        (has_final, final_type, final_value)
        - has_final: Whether FINAL was detected
        - final_type: 'direct' for FINAL(), 'var' for FINAL_VAR(), or None
        - final_value: Extracted value or variable name, or None
    """
    # Pattern for FINAL("...")
    direct_pattern = r'FINAL\(["\'](.+?)["\']\)'
    direct_match = re.search(direct_pattern, text, re.DOTALL)
    if direct_match:
        return True, 'direct', direct_match.group(1)

    # Pattern for FINAL_VAR(var_name)
    var_pattern = r'FINAL_VAR\(([a-zA-Z_][a-zA-Z0-9_]*)\)'
    var_match = re.search(var_pattern, text)
    if var_match:
        return True, 'var', var_match.group(1)

    return False, None, None


def extract_citations(text: str) -> List[str]:
    """
    Extract citation references from text.

    Looks for patterns like [chunk 2.3], [source: filename], etc.

    Args:
        text: Text containing citations

    Returns:
        List of citation strings
    """
    # Pattern for [chunk X], [source: Y], [chunk X.Y], etc.
    citation_pattern = r'\[(chunk\s+[\d.]+|source:\s*[^\]]+|[^\]]+)\]'
    matches = re.findall(citation_pattern, text, re.IGNORECASE)
    return matches


def format_tool_result(result: Dict[str, Any], max_length: int = 4096) -> str:
    """
    Format tool result for LLM consumption.

    Args:
        result: Tool result dictionary
        max_length: Maximum length (truncate if needed)

    Returns:
        Formatted string
    """
    # Handle different result structures
    if 'preview' in result:
        formatted = result['preview']
        if result.get('truncated'):
            formatted += f"\n[truncated at {len(result['preview'])} bytes]"
    elif 'answer' in result:
        formatted = str(result['answer'])
    elif 'notes' in result:
        formatted = "Notes:\n" + "\n".join(f"- {note}" for note in result['notes'])
    else:
        formatted = str(result)

    # Truncate if needed
    if len(formatted) > max_length:
        formatted = formatted[:max_length] + f"\n...[truncated at {max_length} chars]"

    return formatted


def parse_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse tool calls from LLM response.

    Handles OpenAI, Anthropic, and Gemini formats.

    Args:
        response: LLM response dictionary

    Returns:
        List of tool calls with standardized format:
        [{'id': '...', 'name': '...', 'args': {...}}, ...]
    """
    tool_calls = []

    # OpenAI format
    if 'choices' in response:
        message = response['choices'][0].get('message', {})
        if 'tool_calls' in message and message['tool_calls']:
            for tc in message['tool_calls']:
                tool_calls.append({
                    'id': tc.get('id', ''),
                    'name': tc['function']['name'],
                    'args': tc['function'].get('arguments', {})
                })

    # Anthropic format
    elif 'content' in response:
        for block in response.get('content', []):
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                tool_calls.append({
                    'id': block.get('id', ''),
                    'name': block.get('name', ''),
                    'args': block.get('input', {})
                })

    # Gemini format
    elif 'candidates' in response:
        candidate = response['candidates'][0]
        content = candidate.get('content', {})
        for part in content.get('parts', []):
            if 'functionCall' in part:
                fc = part['functionCall']
                tool_calls.append({
                    'id': fc.get('id', ''),
                    'name': fc.get('name', ''),
                    'args': fc.get('args', {})
                })

    return tool_calls


def extract_text_content(response: Dict[str, Any]) -> str:
    """
    Extract text content from LLM response.

    Args:
        response: LLM response dictionary

    Returns:
        Extracted text content
    """
    # OpenAI format
    if 'choices' in response:
        message = response['choices'][0].get('message', {})
        return message.get('content', '')

    # Anthropic format
    elif 'content' in response:
        text_blocks = []
        for block in response.get('content', []):
            if isinstance(block, dict) and block.get('type') == 'text':
                text_blocks.append(block.get('text', ''))
            elif isinstance(block, str):
                text_blocks.append(block)
        return '\n'.join(text_blocks)

    # Gemini format
    elif 'candidates' in response:
        candidate = response['candidates'][0]
        content = candidate.get('content', {})
        text_parts = []
        for part in content.get('parts', []):
            if 'text' in part:
                text_parts.append(part['text'])
        return '\n'.join(text_parts)

    return ''


def truncate_text(text: str, max_bytes: int, suffix: str = "...") -> Tuple[str, bool]:
    """
    Truncate text to max bytes.

    Args:
        text: Text to truncate
        max_bytes: Maximum bytes
        suffix: Suffix to add if truncated

    Returns:
        (truncated_text, was_truncated)
    """
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text, False

    # Truncate and decode
    truncated = encoded[:max_bytes - len(suffix.encode('utf-8'))].decode('utf-8', errors='ignore')
    return truncated + suffix, True


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Basic validation of tool arguments.

    Args:
        tool_name: Tool name
        args: Tool arguments

    Returns:
        (is_valid, error_message)
    """
    # context_probe validation
    if tool_name == 'context_probe':
        if 'kind' not in args:
            return False, "Missing required argument 'kind'"
        if args['kind'] not in ['search', 'read', 'summarize']:
            return False, f"Invalid kind: {args['kind']}"

    # rlm_call validation
    elif tool_name == 'rlm_call':
        if 'query' not in args:
            return False, "Missing required argument 'query'"

    # python_exec validation
    elif tool_name == 'python_exec':
        if 'code' not in args:
            return False, "Missing required argument 'code'"

    return True, None


def format_scratchpad_for_prompt(notes: List[Any], limit: Optional[int] = None) -> str:
    """
    Format scratchpad notes for system prompt.

    Args:
        notes: List of Note objects or strings
        limit: Optional limit on number of recent notes

    Returns:
        Formatted string
    """
    if not notes:
        return "(no notes yet)"

    notes_to_show = notes[-limit:] if limit else notes
    bullets = []

    for note in notes_to_show:
        if hasattr(note, 'text'):
            # Note object
            bullet = f"- {note.text}"
            if hasattr(note, 'source_hint') and note.source_hint:
                bullet += f" [{note.source_hint}]"
        else:
            # String
            bullet = f"- {note}"

        bullets.append(bullet)

    return "\n".join(bullets)
