"""
Context formatters following Anthropic's context engineering best practices.

Provides utilities for:
- XML-structured context formatting
- Context compaction strategies
- Token-aware truncation
- Message formatting for LLMs
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json


class XMLFormatter:
    """
    Formatter for XML-structured context following Anthropic best practices.

    XML provides clear boundaries and structure for LLMs to parse context effectively.
    """

    @staticmethod
    def format_conversation_history(
        messages: List[Dict[str, Any]],
        max_messages: Optional[int] = None,
        compact: bool = False
    ) -> str:
        """
        Format conversation history as XML.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_messages: Maximum number of messages to include (most recent)
            compact: If True, use compact formatting without extra whitespace

        Returns:
            XML-formatted conversation history
        """
        if max_messages:
            messages = messages[-max_messages:]

        if not messages:
            return ""

        lines = ["<conversation_history>"]

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if compact:
                lines.append(f"  <{role}>{content}</{role}>")
            else:
                lines.append(f"  <{role}>")
                lines.append(f"    {content}")
                lines.append(f"  </{role}>")

        lines.append("</conversation_history>")
        return "\n".join(lines)

    @staticmethod
    def format_context_section(
        section_name: str,
        content: str,
        subsections: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format a context section with optional subsections.

        Args:
            section_name: Name of the section (e.g., "background", "task")
            content: Main content for the section
            subsections: Optional dict of subsection_name -> content

        Returns:
            XML-formatted context section
        """
        lines = [f"<{section_name}>"]

        if content:
            lines.append(f"  {content}")

        if subsections:
            for sub_name, sub_content in subsections.items():
                lines.append(f"  <{sub_name}>")
                lines.append(f"    {sub_content}")
                lines.append(f"  </{sub_name}>")

        lines.append(f"</{section_name}>")
        return "\n".join(lines)

    @staticmethod
    def format_list(
        items: List[str],
        tag_name: str = "items",
        item_tag: str = "item"
    ) -> str:
        """
        Format a list of items as XML.

        Args:
            items: List of items to format
            tag_name: Tag name for the container
            item_tag: Tag name for each item

        Returns:
            XML-formatted list
        """
        lines = [f"<{tag_name}>"]
        for item in items:
            lines.append(f"  <{item_tag}>{item}</{item_tag}>")
        lines.append(f"</{tag_name}>")
        return "\n".join(lines)

    @staticmethod
    def format_key_value(
        data: Dict[str, Any],
        container_tag: str = "data"
    ) -> str:
        """
        Format key-value pairs as XML.

        Args:
            data: Dictionary of key-value pairs
            container_tag: Tag name for the container

        Returns:
            XML-formatted key-value pairs
        """
        lines = [f"<{container_tag}>"]
        for key, value in data.items():
            # Handle nested structures
            if isinstance(value, dict):
                lines.append(f"  <{key}>")
                for sub_key, sub_value in value.items():
                    lines.append(f"    <{sub_key}>{sub_value}</{sub_key}>")
                lines.append(f"  </{key}>")
            elif isinstance(value, list):
                lines.append(f"  <{key}>")
                for item in value:
                    lines.append(f"    <item>{item}</item>")
                lines.append(f"  </{key}>")
            else:
                lines.append(f"  <{key}>{value}</{key}>")
        lines.append(f"</{container_tag}>")
        return "\n".join(lines)


class MarkdownFormatter:
    """
    Formatter for Markdown-structured context.

    Alternative to XML for contexts where Markdown is preferred.
    """

    @staticmethod
    def format_conversation_history(
        messages: List[Dict[str, Any]],
        max_messages: Optional[int] = None
    ) -> str:
        """Format conversation history as Markdown"""
        if max_messages:
            messages = messages[-max_messages:]

        if not messages:
            return ""

        lines = ["## Conversation History\n"]

        for msg in messages:
            role = msg.get('role', 'user').capitalize()
            content = msg.get('content', '')
            lines.append(f"**{role}:** {content}\n")

        return "\n".join(lines)

    @staticmethod
    def format_section(
        title: str,
        content: str,
        level: int = 2
    ) -> str:
        """
        Format a section with a title.

        Args:
            title: Section title
            content: Section content
            level: Header level (1-6)

        Returns:
            Markdown-formatted section
        """
        header = "#" * level
        return f"{header} {title}\n\n{content}"

    @staticmethod
    def format_list(items: List[str], ordered: bool = False) -> str:
        """Format a list as Markdown"""
        lines = []
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "-"
            lines.append(f"{prefix} {item}")
        return "\n".join(lines)


class ContextCompactor:
    """
    Strategies for compacting context to fit within token budgets.

    Follows Anthropic's principle: "Context is a finite resource with diminishing marginal returns"
    """

    @staticmethod
    def sliding_window(
        messages: List[Dict[str, Any]],
        max_messages: int,
        preserve_system: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Keep only the most recent N messages.

        Args:
            messages: List of messages
            max_messages: Maximum number of messages to keep
            preserve_system: If True, always keep system messages

        Returns:
            Compacted message list
        """
        if preserve_system:
            system_messages = [msg for msg in messages if msg.get('role') == 'system']
            other_messages = [msg for msg in messages if msg.get('role') != 'system']
            return system_messages + other_messages[-max_messages:]
        return messages[-max_messages:]

    @staticmethod
    def summarize_old_messages(
        messages: List[Dict[str, Any]],
        threshold: int,
        summary_placeholder: str = "[Earlier conversation summarized]"
    ) -> List[Dict[str, Any]]:
        """
        Replace old messages with a summary placeholder.

        Args:
            messages: List of messages
            threshold: Keep this many recent messages, summarize the rest
            summary_placeholder: Text to use as summary

        Returns:
            Compacted message list with summary
        """
        if len(messages) <= threshold:
            return messages

        # Keep recent messages
        recent = messages[-threshold:]

        # Add summary placeholder
        summary_msg = {
            'role': 'system',
            'content': summary_placeholder
        }

        return [summary_msg] + recent

    @staticmethod
    def truncate_long_messages(
        messages: List[Dict[str, Any]],
        max_length: int,
        truncation_indicator: str = "... [truncated]"
    ) -> List[Dict[str, Any]]:
        """
        Truncate individual messages that exceed max_length.

        Args:
            messages: List of messages
            max_length: Maximum character length per message
            truncation_indicator: Text to append to truncated messages

        Returns:
            List of messages with long content truncated
        """
        truncated = []
        for msg in messages:
            content = msg.get('content', '')
            if len(content) > max_length:
                msg = msg.copy()
                msg['content'] = content[:max_length - len(truncation_indicator)] + truncation_indicator
            truncated.append(msg)
        return truncated

    @staticmethod
    def compact_tool_results(
        messages: List[Dict[str, Any]],
        max_result_length: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Compact tool result messages to focus on key information.

        Args:
            messages: List of messages
            max_result_length: Maximum length for tool result content

        Returns:
            Compacted message list
        """
        compacted = []
        for msg in messages:
            content = msg.get('content', '')

            # Check if this is a tool result message
            if 'Tool' in content and 'returned:' in content:
                if len(content) > max_result_length:
                    msg = msg.copy()
                    msg['content'] = content[:max_result_length] + "... [result truncated]"

            compacted.append(msg)
        return compacted


class MessageFormatter:
    """
    Utilities for formatting messages for LLM consumption.
    """

    @staticmethod
    def format_tool_result(
        tool_name: str,
        result: Any,
        format_type: str = 'xml'
    ) -> str:
        """
        Format a tool result for inclusion in context.

        Args:
            tool_name: Name of the tool
            result: Tool result (dict, list, or string)
            format_type: Format type ('xml' or 'markdown')

        Returns:
            Formatted tool result string
        """
        if format_type == 'xml':
            if isinstance(result, dict):
                result_str = XMLFormatter.format_key_value(result, container_tag="result")
            elif isinstance(result, list):
                result_str = XMLFormatter.format_list(result, tag_name="result", item_tag="item")
            else:
                result_str = f"<result>{result}</result>"

            return f"<tool_call name='{tool_name}'>\n{result_str}\n</tool_call>"
        else:
            # Markdown format
            result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
            return f"**Tool: {tool_name}**\n```json\n{result_str}\n```"

    @staticmethod
    def format_system_message(
        content: str,
        structured: bool = True
    ) -> Dict[str, str]:
        """
        Format a system message.

        Args:
            content: System message content
            structured: If True, assumes content is already XML-structured

        Returns:
            Message dict with role='system'
        """
        return {'role': 'system', 'content': content}

    @staticmethod
    def format_user_message(content: str) -> Dict[str, str]:
        """Format a user message"""
        return {'role': 'user', 'content': content}

    @staticmethod
    def format_assistant_message(content: str) -> Dict[str, str]:
        """Format an assistant message"""
        return {'role': 'assistant', 'content': content}
