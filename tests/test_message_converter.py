"""
Tests for convert_to_model_messages utility.

Verifies that UIMessage format (from useChat) is correctly converted to
ModelMessage format (for LLMs).
"""
import pytest
from vel.utils import convert_to_model_messages, convert_from_legacy_format


class TestBasicConversion:
    """Test basic message conversion"""

    def test_simple_user_message(self):
        """Simple user text message"""
        ui_messages = [
            {
                'id': 'msg-1',
                'role': 'user',
                'parts': [
                    {'type': 'text', 'text': 'Hello'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'Hello'  # String format for simple text

    def test_system_message(self):
        """System message"""
        ui_messages = [
            {
                'role': 'system',
                'parts': [
                    {'type': 'text', 'text': 'You are a helpful assistant'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'system'
        assert result[0]['content'] == 'You are a helpful assistant'

    def test_assistant_message(self):
        """Simple assistant text message"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text', 'text': 'Hello! How can I help?'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'Hello! How can I help?'


class TestToolConversion:
    """Test tool call and result conversion"""

    def test_tool_execution_split(self):
        """Tool execution with input and output splits into two messages"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'tool-websearch',
                        'toolCallId': 'call_123',
                        'state': 'output-available',
                        'input': {'query': 'AI trends'},
                        'output': {'results': ['result1', 'result2']}
                    }
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 2

        # First message: tool call
        assert result[0]['role'] == 'assistant'
        assert isinstance(result[0]['content'], list)
        assert result[0]['content'][0]['type'] == 'tool-call'
        assert result[0]['content'][0]['toolCallId'] == 'call_123'
        assert result[0]['content'][0]['toolName'] == 'tool-websearch'
        assert result[0]['content'][0]['input'] == {'query': 'AI trends'}

        # Second message: tool result
        assert result[1]['role'] == 'tool'
        assert result[1]['content'][0]['type'] == 'tool-result'
        assert result[1]['content'][0]['toolCallId'] == 'call_123'
        assert result[1]['content'][0]['output'] == {'results': ['result1', 'result2']}

    def test_multiple_tool_executions(self):
        """Multiple tool executions in one message"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'tool-websearch',
                        'toolCallId': 'call_1',
                        'state': 'output-available',
                        'input': {'query': 'AI'},
                        'output': {'results': ['a']}
                    },
                    {
                        'type': 'tool-news',
                        'toolCallId': 'call_2',
                        'state': 'output-available',
                        'input': {'topic': 'tech'},
                        'output': {'articles': ['b']}
                    }
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 2

        # Assistant message with both tool calls
        assert result[0]['role'] == 'assistant'
        assert len(result[0]['content']) == 2
        assert result[0]['content'][0]['toolName'] == 'tool-websearch'
        assert result[0]['content'][1]['toolName'] == 'tool-news'

        # Tool message with both results
        assert result[1]['role'] == 'tool'
        assert len(result[1]['content']) == 2
        assert result[1]['content'][0]['toolCallId'] == 'call_1'
        assert result[1]['content'][1]['toolCallId'] == 'call_2'

    def test_text_and_tool_call(self):
        """Assistant message with both text and tool call"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text', 'text': 'Let me search for that.'},
                    {
                        'type': 'tool-websearch',
                        'toolCallId': 'call_123',
                        'state': 'output-available',
                        'input': {'query': 'test'},
                        'output': {'results': []}
                    }
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 2

        # Assistant message has both text and tool call
        assert result[0]['role'] == 'assistant'
        assert isinstance(result[0]['content'], list)
        assert result[0]['content'][0]['type'] == 'text'
        assert result[0]['content'][0]['text'] == 'Let me search for that.'
        assert result[0]['content'][1]['type'] == 'tool-call'

        # Tool result
        assert result[1]['role'] == 'tool'

    def test_tool_without_output(self):
        """Tool call without output (state != output-available)"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'tool-websearch',
                        'toolCallId': 'call_123',
                        'state': 'input-available',  # Not output-available
                        'input': {'query': 'test'}
                        # No output field
                    }
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        # Only tool call, no tool result
        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'][0]['type'] == 'tool-call'


class TestMultimodalContent:
    """Test multimodal content (images, files)"""

    def test_user_message_with_image(self):
        """User message with image"""
        ui_messages = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'text': 'What is in this image?'},
                    {'type': 'image', 'image': 'base64data', 'mimeType': 'image/png'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert isinstance(result[0]['content'], list)
        assert result[0]['content'][0]['type'] == 'text'
        assert result[0]['content'][1]['type'] == 'image'
        assert result[0]['content'][1]['image'] == 'base64data'
        assert result[0]['content'][1]['mimeType'] == 'image/png'

    def test_user_message_with_file(self):
        """User message with file (PDF)"""
        ui_messages = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'text': 'Analyze this document'},
                    {'type': 'file', 'data': 'pdfdata', 'mimeType': 'application/pdf'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'][1]['type'] == 'file'
        assert result[0]['content'][1]['data'] == 'pdfdata'
        assert result[0]['content'][1]['mimeType'] == 'application/pdf'

    def test_file_with_uri(self):
        """File with URI instead of data"""
        ui_messages = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'file', 'fileUri': 'https://example.com/doc.pdf', 'mimeType': 'application/pdf'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert result[0]['content'][0]['type'] == 'file'
        assert result[0]['content'][0]['fileUri'] == 'https://example.com/doc.pdf'


class TestReasoningContent:
    """Test reasoning content handling"""

    def test_reasoning_in_assistant_message(self):
        """Reasoning content (OpenAI o1/o3, Anthropic thinking) is preserved"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'reasoning',
                        'text': 'The human is asking me to provide steps to play tennis...',
                        'state': 'done'
                    },
                    {
                        'type': 'text',
                        'text': '# Steps to Play Tennis...',
                        'state': 'done'
                    }
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert isinstance(result[0]['content'], list)
        assert len(result[0]['content']) == 2
        assert result[0]['content'][0]['type'] == 'reasoning'
        assert 'play tennis' in result[0]['content'][0]['text']
        assert result[0]['content'][1]['type'] == 'text'
        assert 'Steps to Play Tennis' in result[0]['content'][1]['text']

    def test_reasoning_with_tool_calls(self):
        """Reasoning combined with tool calls"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'reasoning',
                        'text': 'I need to search for this information',
                        'state': 'done'
                    },
                    {
                        'type': 'tool-websearch',
                        'toolCallId': 'call_123',
                        'state': 'output-available',
                        'input': {'query': 'test'},
                        'output': {'results': []}
                    }
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        # Should produce 2 messages: assistant (reasoning + tool-call) and tool (result)
        assert len(result) == 2
        assert result[0]['role'] == 'assistant'
        assert isinstance(result[0]['content'], list)
        assert result[0]['content'][0]['type'] == 'reasoning'
        assert result[0]['content'][1]['type'] == 'tool-call'
        assert result[1]['role'] == 'tool'

    def test_reasoning_only(self):
        """Message with only reasoning content"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'reasoning',
                        'text': 'Analyzing the request...',
                        'state': 'done'
                    }
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert isinstance(result[0]['content'], list)
        assert len(result[0]['content']) == 1
        assert result[0]['content'][0]['type'] == 'reasoning'
        assert result[0]['content'][0]['text'] == 'Analyzing the request...'


class TestUIElementFiltering:
    """Test filtering of UI-only elements"""

    def test_step_start_filtered(self):
        """step-start parts should be filtered out"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'step-start'},
                    {'type': 'text', 'text': 'Processing...'},
                    {'type': 'step-finish'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['content'] == 'Processing...'  # Only text, steps filtered

    def test_empty_after_filtering(self):
        """Message with only UI elements results in empty message"""
        ui_messages = [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'step-start'},
                    {'type': 'step-finish'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        # Should produce empty message or be filtered entirely
        # Current implementation produces empty content
        assert len(result) <= 1


class TestLegacyFormat:
    """Test conversion from legacy format"""

    def test_string_content(self):
        """Legacy format with string content field"""
        ui_messages = [
            {'role': 'user', 'content': 'Hello'}  # No parts, just content
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'Hello'

    def test_array_content(self):
        """Legacy format with array content field"""
        ui_messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': 'Hello'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'Hello'

    def test_convert_from_legacy_format_helper(self):
        """Test convert_from_legacy_format helper"""
        legacy_messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]

        converted = convert_from_legacy_format(legacy_messages)

        assert len(converted) == 2
        assert converted[0]['parts'][0]['type'] == 'text'
        assert converted[0]['parts'][0]['text'] == 'Hello'


class TestComplexScenarios:
    """Test complex real-world scenarios"""

    def test_full_conversation(self):
        """Complete conversation flow"""
        ui_messages = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'text': 'What are the latest AI trends?'}
                ]
            },
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'step-start'},
                    {'type': 'text', 'text': 'Let me search for that.'},
                    {
                        'type': 'tool-websearch',
                        'toolCallId': 'call_1',
                        'state': 'output-available',
                        'input': {'query': 'latest AI trends 2025'},
                        'output': {'results': ['trend1', 'trend2']}
                    },
                    {'type': 'step-finish'}
                ]
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'text': 'Thanks!'}
                ]
            }
        ]

        result = convert_to_model_messages(ui_messages)

        assert len(result) == 4  # user, assistant, tool, user

        # Check structure
        assert result[0]['role'] == 'user'
        assert result[1]['role'] == 'assistant'
        assert result[2]['role'] == 'tool'
        assert result[3]['role'] == 'user'

        # Verify tool call/result split
        assert result[1]['content'][1]['type'] == 'tool-call'
        assert result[2]['content'][0]['type'] == 'tool-result'

    def test_empty_messages_list(self):
        """Empty messages list"""
        result = convert_to_model_messages([])
        assert result == []

    def test_invalid_input(self):
        """Invalid input raises error"""
        with pytest.raises(ValueError):
            convert_to_model_messages("not a list")

    def test_missing_role(self):
        """Message without role raises error"""
        ui_messages = [
            {'parts': [{'type': 'text', 'text': 'Hello'}]}  # No role
        ]

        with pytest.raises(ValueError) as exc_info:
            convert_to_model_messages(ui_messages)

        assert 'missing \'role\' field' in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
