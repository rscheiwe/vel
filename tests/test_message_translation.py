"""
Tests for message translation layer.

Verifies that ModelMessage format (Vercel AI SDK) is correctly translated
to provider-specific formats (OpenAI, Anthropic, Gemini).
"""
import pytest
from vel.providers.message_translator import (
    translate_to_openai,
    translate_to_anthropic,
    translate_to_gemini,
    MessageTranslationError
)


class TestOpenAITranslation:
    """Test ModelMessage -> OpenAI format translation"""

    def test_simple_text_message(self):
        """Simple user message with text"""
        messages = [
            {'role': 'user', 'content': 'Hello'}
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'Hello'

    def test_system_message(self):
        """System message"""
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'}
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'system'
        assert result[0]['content'] == 'You are a helpful assistant'

    def test_tool_call(self):
        """Assistant message with tool call"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'tool-call', 'toolCallId': 'call_123', 'toolName': 'get_weather', 'input': {'city': 'SF'}}
                ]
            }
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == ''  # OpenAI requires empty string when tool_calls present
        assert 'tool_calls' in result[0]
        assert len(result[0]['tool_calls']) == 1
        assert result[0]['tool_calls'][0]['id'] == 'call_123'
        assert result[0]['tool_calls'][0]['type'] == 'function'
        assert result[0]['tool_calls'][0]['function']['name'] == 'get_weather'
        assert result[0]['tool_calls'][0]['function']['arguments'] == '{"city": "SF"}'

    def test_multiple_tool_calls(self):
        """Assistant message with multiple tool calls"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'tool-call', 'toolCallId': 'call_1', 'toolName': 'tool_a', 'input': {'x': 1}},
                    {'type': 'tool-call', 'toolCallId': 'call_2', 'toolName': 'tool_b', 'input': {'y': 2}}
                ]
            }
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert len(result[0]['tool_calls']) == 2
        assert result[0]['tool_calls'][0]['id'] == 'call_1'
        assert result[0]['tool_calls'][1]['id'] == 'call_2'

    def test_tool_result(self):
        """Tool result message"""
        messages = [
            {
                'role': 'tool',
                'content': [
                    {'type': 'tool-result', 'toolCallId': 'call_123', 'toolName': 'get_weather', 'output': {'temp': 72}}
                ]
            }
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'tool'
        assert result[0]['tool_call_id'] == 'call_123'
        assert result[0]['content'] == '{"temp": 72}'

    def test_mixed_content(self):
        """Assistant message with text and tool call"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': 'Let me check that.'},
                    {'type': 'tool-call', 'toolCallId': 'call_123', 'toolName': 'search', 'input': {'query': 'test'}}
                ]
            }
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert result[0]['content'] == 'Let me check that.'
        assert len(result[0]['tool_calls']) == 1

    def test_string_content(self):
        """Handles both string and array content formats"""
        messages = [
            {'role': 'user', 'content': 'Hello'}  # String content
        ]

        result = translate_to_openai(messages)

        assert result[0]['content'] == 'Hello'

    def test_missing_role(self):
        """Error when role is missing"""
        messages = [
            {'content': 'Hello'}  # No role
        ]

        with pytest.raises(MessageTranslationError) as exc_info:
            translate_to_openai(messages)

        assert 'missing \'role\' field' in str(exc_info.value)
        assert exc_info.value.provider == 'openai'


class TestAnthropicTranslation:
    """Test ModelMessage -> Anthropic format translation"""

    def test_simple_user_message(self):
        """Simple user message"""
        messages = [
            {'role': 'user', 'content': 'Hello'}
        ]

        system, result = translate_to_anthropic(messages)

        assert system is None
        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'][0]['type'] == 'text'
        assert result[0]['content'][0]['text'] == 'Hello'

    def test_system_message_extraction(self):
        """System message extracted separately"""
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'Hello'}
        ]

        system, result = translate_to_anthropic(messages)

        assert system == 'You are a helpful assistant'
        assert len(result) == 1  # System not in messages array
        assert result[0]['role'] == 'user'

    def test_tool_call(self):
        """Assistant message with tool use"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'tool-call', 'toolCallId': 'call_123', 'toolName': 'search', 'input': {'query': 'test'}}
                ]
            }
        ]

        system, result = translate_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'][0]['type'] == 'tool_use'
        assert result[0]['content'][0]['id'] == 'call_123'
        assert result[0]['content'][0]['name'] == 'search'
        assert result[0]['content'][0]['input'] == {'query': 'test'}

    def test_tool_result(self):
        """Tool result converted to user message"""
        messages = [
            {
                'role': 'tool',
                'content': [
                    {'type': 'tool-result', 'toolCallId': 'call_123', 'toolName': 'search', 'output': {'results': []}}
                ]
            }
        ]

        system, result = translate_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'  # Tool results become user messages in Anthropic
        assert result[0]['content'][0]['type'] == 'tool_result'
        assert result[0]['content'][0]['tool_use_id'] == 'call_123'


class TestGeminiTranslation:
    """Test ModelMessage -> Gemini format translation"""

    def test_simple_user_message(self):
        """Simple user message"""
        messages = [
            {'role': 'user', 'content': 'Hello'}
        ]

        system, result = translate_to_gemini(messages)

        assert system is None
        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['parts'][0]['text'] == 'Hello'

    def test_system_instruction(self):
        """System message becomes systemInstruction"""
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'Hello'}
        ]

        system, result = translate_to_gemini(messages)

        assert system is not None
        assert system['parts'][0]['text'] == 'You are a helpful assistant'
        assert len(result) == 1

    def test_assistant_to_model_role(self):
        """Assistant role converted to model role"""
        messages = [
            {'role': 'assistant', 'content': 'Hi there!'}
        ]

        system, result = translate_to_gemini(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'model'  # Not 'assistant'
        assert result[0]['parts'][0]['text'] == 'Hi there!'

    def test_tool_call(self):
        """Tool call uses function_call"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'tool-call', 'toolCallId': 'call_123', 'toolName': 'search', 'input': {'query': 'test'}}
                ]
            }
        ]

        system, result = translate_to_gemini(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'model'
        assert result[0]['parts'][0]['function_call']['name'] == 'search'
        assert result[0]['parts'][0]['function_call']['args'] == {'query': 'test'}

    def test_tool_result(self):
        """Tool result uses function_response"""
        messages = [
            {
                'role': 'tool',
                'content': [
                    {'type': 'tool-result', 'toolCallId': 'call_123', 'toolName': 'search', 'output': {'count': 5}}
                ]
            }
        ]

        system, result = translate_to_gemini(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'  # Tool results become user messages
        assert result[0]['parts'][0]['function_response']['name'] == 'search'
        assert result[0]['parts'][0]['function_response']['response'] == {'count': 5}


class TestReasoningContent:
    """Test reasoning content handling across all providers"""

    def test_reasoning_openai(self):
        """Reasoning content included in OpenAI messages"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'reasoning', 'text': 'Let me think about this...'},
                    {'type': 'text', 'text': 'Here is my answer'}
                ]
            }
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        # Reasoning and text both included in content
        assert 'Let me think about this... Here is my answer' in result[0]['content']

    def test_reasoning_anthropic(self):
        """Reasoning content handled in Anthropic messages"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'reasoning', 'text': 'The user is asking about...'},
                    {'type': 'text', 'text': 'My response'}
                ]
            }
        ]

        system, result = translate_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert len(result[0]['content']) == 2
        assert result[0]['content'][0]['type'] == 'text'
        assert 'user is asking' in result[0]['content'][0]['text']
        assert result[0]['content'][1]['type'] == 'text'
        assert 'My response' in result[0]['content'][1]['text']

    def test_reasoning_gemini(self):
        """Reasoning content handled in Gemini messages"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'reasoning', 'text': 'Processing the query...'},
                    {'type': 'text', 'text': 'Result'}
                ]
            }
        ]

        system, result = translate_to_gemini(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'model'
        assert len(result[0]['parts']) == 2
        assert result[0]['parts'][0]['text'] == 'Processing the query...'
        assert result[0]['parts'][1]['text'] == 'Result'

    def test_reasoning_with_tool_calls_openai(self):
        """Reasoning combined with tool calls in OpenAI"""
        messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'reasoning', 'text': 'I need to search for this'},
                    {'type': 'tool-call', 'toolCallId': 'call_123', 'toolName': 'search', 'input': {'q': 'test'}}
                ]
            }
        ]

        result = translate_to_openai(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert 'I need to search' in result[0]['content']
        assert 'tool_calls' in result[0]
        assert result[0]['tool_calls'][0]['id'] == 'call_123'


class TestMultipleToolCallsAndResults:
    """Test complex scenarios with multiple tool interactions"""

    def test_full_conversation_openai(self):
        """Full conversation with multiple tool calls"""
        messages = [
            {'role': 'user', 'content': 'Compare weather in SF and NYC'},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'tool-call', 'toolCallId': 'call_1', 'toolName': 'weather', 'input': {'city': 'SF'}},
                    {'type': 'tool-call', 'toolCallId': 'call_2', 'toolName': 'weather', 'input': {'city': 'NYC'}}
                ]
            },
            {
                'role': 'tool',
                'content': [
                    {'type': 'tool-result', 'toolCallId': 'call_1', 'output': {'temp': 72}},
                    {'type': 'tool-result', 'toolCallId': 'call_2', 'output': {'temp': 65}}
                ]
            }
        ]

        result = translate_to_openai(messages)

        assert len(result) == 4  # user, assistant, tool, tool
        assert result[1]['role'] == 'assistant'
        assert len(result[1]['tool_calls']) == 2
        assert result[2]['role'] == 'tool'
        assert result[2]['tool_call_id'] == 'call_1'
        assert result[3]['role'] == 'tool'
        assert result[3]['tool_call_id'] == 'call_2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
