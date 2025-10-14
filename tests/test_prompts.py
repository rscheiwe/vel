"""
Unit tests for the prompt module.

Tests:
- PromptTemplate rendering
- PromptRegistry management
- PromptManager integration
- PromptContextManager behavior
- Formatters and utilities
"""
import pytest
from vel.prompts import (
    PromptTemplate,
    SystemPromptBuilder,
    PromptRegistry,
    register_prompt,
    get_prompt,
    has_prompt,
    PromptManager,
    PromptContextManager,
    XMLFormatter,
    MarkdownFormatter,
    ContextCompactor
)


# ================================================================================
# PromptTemplate Tests
# ================================================================================

class TestPromptTemplate:
    """Tests for PromptTemplate class"""

    def test_basic_template_creation(self):
        """Test creating a basic template"""
        template = PromptTemplate(
            id="test:v1",
            system="You are a {{role}} assistant.",
            variables={"role": "helpful"}
        )
        assert template.id == "test:v1"
        assert template.version == "v1"

    def test_template_rendering(self):
        """Test rendering a template with variables"""
        template = PromptTemplate(
            id="test:v1",
            system="Hello {{name}}, you are {{age}} years old.",
            variables={"name": "Alice", "age": 30}
        )
        rendered = template.render()
        assert "Alice" in rendered
        assert "30" in rendered

    def test_template_override_variables(self):
        """Test overriding default variables"""
        template = PromptTemplate(
            id="test:v1",
            system="Hello {{name}}",
            variables={"name": "Alice"}
        )
        rendered = template.render(name="Bob")
        assert "Bob" in rendered
        assert "Alice" not in rendered

    def test_environment_based_templates(self):
        """Test environment-specific templates"""
        template = PromptTemplate(
            id="test:v1",
            environments={
                "dev": "Debug mode: {{message}}",
                "prod": "Production: {{message}}"
            },
            variables={"message": "hello"}
        )

        dev_rendered = template.render(environment="dev")
        prod_rendered = template.render(environment="prod")

        assert "Debug mode" in dev_rendered
        assert "Production" in prod_rendered

    def test_xml_structured_template(self):
        """Test XML-structured templates"""
        template = PromptTemplate(
            id="test:v1",
            system="""
            <system_instructions>
              <role>{{role}}</role>
            </system_instructions>
            """,
            variables={"role": "assistant"}
        )
        rendered = template.render()
        assert "<system_instructions>" in rendered
        assert "<role>assistant</role>" in rendered

    def test_missing_environment_raises_error(self):
        """Test that missing environment raises KeyError"""
        template = PromptTemplate(
            id="test:v1",
            environments={"dev": "Debug"}
        )
        with pytest.raises(KeyError):
            template.render(environment="nonexistent")

    def test_undefined_variable_raises_error(self):
        """Test that undefined variables raise error"""
        template = PromptTemplate(
            id="test:v1",
            system="Hello {{name}}"
        )
        with pytest.raises(Exception):  # Jinja2 UndefinedError
            template.render()

    def test_get_environments(self):
        """Test getting list of environments"""
        template = PromptTemplate(
            id="test:v1",
            environments={"dev": "Dev", "prod": "Prod"}
        )
        envs = template.get_environments()
        assert "dev" in envs
        assert "prod" in envs

    def test_validate_template(self):
        """Test template validation"""
        template = PromptTemplate(
            id="test:v1",
            system="Hello {{name}}",
            variables={"name": "Alice"}
        )

        # Valid rendering
        is_valid, error = template.validate()
        assert is_valid
        assert error is None

        # Invalid rendering (missing variable)
        template2 = PromptTemplate(
            id="test:v2",
            system="Hello {{name}}"
        )
        is_valid, error = template2.validate()
        assert not is_valid
        assert error is not None


# ================================================================================
# SystemPromptBuilder Tests
# ================================================================================

class TestSystemPromptBuilder:
    """Tests for SystemPromptBuilder"""

    def test_basic_builder(self):
        """Test basic prompt building"""
        builder = SystemPromptBuilder()
        builder.add_role("You are a helpful assistant")
        prompt = builder.build()
        assert "<system_instructions>" in prompt
        assert "<role>You are a helpful assistant</role>" in prompt

    def test_capabilities(self):
        """Test adding capabilities"""
        builder = SystemPromptBuilder()
        builder.add_capabilities(["Cap 1", "Cap 2"])
        prompt = builder.build()
        assert "<capabilities>" in prompt
        assert "Cap 1" in prompt
        assert "Cap 2" in prompt

    def test_guidelines(self):
        """Test adding guidelines"""
        builder = SystemPromptBuilder()
        builder.add_guidelines(["Rule 1", "Rule 2"])
        prompt = builder.build()
        assert "<guidelines>" in prompt
        assert "Rule 1" in prompt
        assert "Rule 2" in prompt

    def test_context(self):
        """Test adding context"""
        builder = SystemPromptBuilder()
        builder.add_context("company", "Acme Corp")
        prompt = builder.build()
        assert "<context>" in prompt
        assert "<company>Acme Corp</company>" in prompt

    def test_custom_section(self):
        """Test adding custom sections"""
        builder = SystemPromptBuilder()
        builder.add_section("custom", "Custom content")
        prompt = builder.build()
        assert "<custom>" in prompt
        assert "Custom content" in prompt


# ================================================================================
# PromptRegistry Tests
# ================================================================================

class TestPromptRegistry:
    """Tests for PromptRegistry"""

    def setup_method(self):
        """Reset registry before each test"""
        PromptRegistry.reset()

    def test_register_and_get(self):
        """Test registering and retrieving prompts"""
        registry = PromptRegistry.default()
        template = PromptTemplate(id="test:v1", system="Test")

        registry.register(template)
        retrieved = registry.get("test:v1")

        assert retrieved.id == "test:v1"

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate ID raises error"""
        registry = PromptRegistry.default()
        template = PromptTemplate(id="test:v1", system="Test")

        registry.register(template)
        with pytest.raises(ValueError):
            registry.register(template)

    def test_register_or_update(self):
        """Test register_or_update method"""
        registry = PromptRegistry.default()
        template1 = PromptTemplate(id="test:v1", system="Version 1")
        template2 = PromptTemplate(id="test:v1", system="Version 2")

        registry.register_or_update(template1)
        registry.register_or_update(template2)

        retrieved = registry.get("test:v1")
        assert "Version 2" in retrieved.render()

    def test_get_nonexistent_raises_error(self):
        """Test getting nonexistent prompt raises KeyError"""
        registry = PromptRegistry.default()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_or_none(self):
        """Test get_or_none returns None for missing prompts"""
        registry = PromptRegistry.default()
        result = registry.get_or_none("nonexistent")
        assert result is None

    def test_has_prompt(self):
        """Test checking prompt existence"""
        registry = PromptRegistry.default()
        template = PromptTemplate(id="test:v1", system="Test")
        registry.register(template)

        assert registry.has("test:v1")
        assert not registry.has("nonexistent")

    def test_list_ids(self):
        """Test listing prompt IDs"""
        registry = PromptRegistry.default()
        registry.register(PromptTemplate(id="test1:v1", system="Test 1"))
        registry.register(PromptTemplate(id="test2:v1", system="Test 2"))

        ids = registry.list_ids()
        assert "test1:v1" in ids
        assert "test2:v1" in ids

    def test_list_by_prefix(self):
        """Test listing prompts by prefix"""
        registry = PromptRegistry.default()
        registry.register(PromptTemplate(id="chat:v1", system="Chat v1"))
        registry.register(PromptTemplate(id="chat:v2", system="Chat v2"))
        registry.register(PromptTemplate(id="code:v1", system="Code v1"))

        chat_prompts = registry.list_by_prefix("chat")
        assert len(chat_prompts) == 2

    def test_list_by_version(self):
        """Test listing prompts by version"""
        registry = PromptRegistry.default()
        registry.register(PromptTemplate(id="agent:v1", system="V1"))
        registry.register(PromptTemplate(id="agent:v2", system="V2"))

        versions = registry.list_by_version("agent")
        assert "v1" in versions
        assert "v2" in versions

    def test_remove(self):
        """Test removing a prompt"""
        registry = PromptRegistry.default()
        template = PromptTemplate(id="test:v1", system="Test")
        registry.register(template)

        registry.remove("test:v1")
        assert not registry.has("test:v1")

    def test_clear(self):
        """Test clearing all prompts"""
        registry = PromptRegistry.default()
        registry.register(PromptTemplate(id="test1:v1", system="Test 1"))
        registry.register(PromptTemplate(id="test2:v1", system="Test 2"))

        registry.clear()
        assert registry.count() == 0

    def test_convenience_functions(self):
        """Test global convenience functions"""
        PromptRegistry.reset()
        template = PromptTemplate(id="test:v1", system="Test")

        register_prompt(template)
        assert has_prompt("test:v1")

        retrieved = get_prompt("test:v1")
        assert retrieved.id == "test:v1"


# ================================================================================
# PromptManager Tests
# ================================================================================

class TestPromptManager:
    """Tests for PromptManager"""

    def setup_method(self):
        """Reset registry before each test"""
        PromptRegistry.reset()

    def test_basic_manager(self):
        """Test basic manager creation"""
        template = PromptTemplate(
            id="test:v1",
            system="Hello {{name}}",
            variables={"name": "World"}
        )
        register_prompt(template)

        manager = PromptManager(
            prompt_id="test:v1",
            prompt_vars={"name": "Alice"}
        )

        assert manager.has_template()
        rendered = manager.render_system_prompt()
        assert "Alice" in rendered

    def test_create_system_message(self):
        """Test creating system message dict"""
        template = PromptTemplate(id="test:v1", system="Test prompt")
        register_prompt(template)

        manager = PromptManager(prompt_id="test:v1")
        msg = manager.create_system_message()

        assert msg['role'] == 'system'
        assert 'Test prompt' in msg['content']

    def test_inject_system_prompt(self):
        """Test injecting system prompt into messages"""
        template = PromptTemplate(id="test:v1", system="System prompt")
        register_prompt(template)

        manager = PromptManager(prompt_id="test:v1")
        messages = [
            {'role': 'user', 'content': 'Hello'}
        ]

        injected = manager.inject_system_prompt(messages)
        assert len(injected) == 2
        assert injected[0]['role'] == 'system'
        assert injected[1]['role'] == 'user'

    def test_update_vars(self):
        """Test updating variables"""
        template = PromptTemplate(
            id="test:v1",
            system="Name: {{name}}, Age: {{age}}",
            variables={"name": "Alice", "age": 30}
        )
        register_prompt(template)

        manager = PromptManager(prompt_id="test:v1")
        manager.update_vars(name="Bob", age=25)

        rendered = manager.render_system_prompt()
        assert "Bob" in rendered
        assert "25" in rendered

    def test_environment_switching(self):
        """Test changing environment"""
        template = PromptTemplate(
            id="test:v1",
            environments={
                "dev": "Dev mode",
                "prod": "Prod mode"
            }
        )
        register_prompt(template)

        manager = PromptManager(prompt_id="test:v1", environment="dev")
        assert "Dev mode" in manager.render_system_prompt()

        manager.set_environment("prod")
        assert "Prod mode" in manager.render_system_prompt()


# ================================================================================
# XMLFormatter Tests
# ================================================================================

class TestXMLFormatter:
    """Tests for XMLFormatter"""

    def test_format_conversation_history(self):
        """Test formatting conversation history"""
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ]

        formatted = XMLFormatter.format_conversation_history(messages)
        assert '<conversation_history>' in formatted
        assert '<user>Hello</user>' in formatted
        assert '<assistant>Hi there</assistant>' in formatted

    def test_format_context_section(self):
        """Test formatting context section"""
        formatted = XMLFormatter.format_context_section(
            "background",
            "Main content",
            subsections={"detail": "Detailed info"}
        )

        assert '<background>' in formatted
        assert 'Main content' in formatted
        assert '<detail>Detailed info</detail>' in formatted

    def test_format_list(self):
        """Test formatting lists"""
        items = ["Item 1", "Item 2", "Item 3"]
        formatted = XMLFormatter.format_list(items)

        assert '<items>' in formatted
        assert '<item>Item 1</item>' in formatted

    def test_format_key_value(self):
        """Test formatting key-value pairs"""
        data = {"name": "Alice", "age": 30}
        formatted = XMLFormatter.format_key_value(data)

        assert '<data>' in formatted
        assert '<name>Alice</name>' in formatted
        assert '<age>30</age>' in formatted


# ================================================================================
# ContextCompactor Tests
# ================================================================================

class TestContextCompactor:
    """Tests for ContextCompactor"""

    def test_sliding_window(self):
        """Test sliding window compaction"""
        messages = [
            {'role': 'user', 'content': f'Message {i}'}
            for i in range(10)
        ]

        compacted = ContextCompactor.sliding_window(messages, max_messages=3)
        assert len(compacted) == 3
        assert compacted[0]['content'] == 'Message 7'

    def test_sliding_window_preserve_system(self):
        """Test sliding window with system message preservation"""
        messages = [
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'Msg 1'},
            {'role': 'user', 'content': 'Msg 2'},
            {'role': 'user', 'content': 'Msg 3'}
        ]

        compacted = ContextCompactor.sliding_window(
            messages,
            max_messages=2,
            preserve_system=True
        )
        assert compacted[0]['role'] == 'system'
        assert len(compacted) == 3  # system + 2 recent

    def test_summarize_old_messages(self):
        """Test summarizing old messages"""
        messages = [
            {'role': 'user', 'content': f'Message {i}'}
            for i in range(5)
        ]

        compacted = ContextCompactor.summarize_old_messages(
            messages,
            threshold=2
        )
        assert len(compacted) == 3  # summary + 2 recent
        assert compacted[0]['role'] == 'system'

    def test_truncate_long_messages(self):
        """Test truncating long messages"""
        messages = [
            {'role': 'user', 'content': 'a' * 1000}
        ]

        truncated = ContextCompactor.truncate_long_messages(
            messages,
            max_length=100
        )
        assert len(truncated[0]['content']) <= 100


# ================================================================================
# PromptContextManager Tests
# ================================================================================

class TestPromptContextManager:
    """Tests for PromptContextManager"""

    def setup_method(self):
        """Reset registry before each test"""
        PromptRegistry.reset()

    def test_context_manager_with_prompt(self):
        """Test PromptContextManager with template"""
        template = PromptTemplate(
            id="test:v1",
            system="System: {{var}}",
            variables={"var": "value"}
        )
        register_prompt(template)

        ctx_mgr = PromptContextManager(
            prompt_id="test:v1",
            prompt_vars={"var": "test"}
        )

        assert ctx_mgr.has_prompt_template()

    def test_messages_with_system_prompt(self):
        """Test that messages include system prompt"""
        template = PromptTemplate(
            id="test:v1",
            system="System prompt"
        )
        register_prompt(template)

        ctx_mgr = PromptContextManager(prompt_id="test:v1")
        run_id = "test-run"
        ctx_mgr.set_input(run_id, {"message": "Hello"})

        messages = ctx_mgr.messages_for_llm(run_id)

        # Should have system message + user message
        assert len(messages) >= 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'

    def test_update_prompt_vars(self):
        """Test updating prompt variables"""
        template = PromptTemplate(
            id="test:v1",
            system="Name: {{name}}",
            variables={"name": "Alice"}
        )
        register_prompt(template)

        ctx_mgr = PromptContextManager(
            prompt_id="test:v1",
            prompt_vars={"name": "Alice"}
        )

        ctx_mgr.update_prompt_vars(name="Bob")

        run_id = "test-run"
        ctx_mgr.set_input(run_id, {"message": "Hello"})
        messages = ctx_mgr.messages_for_llm(run_id)

        # Check that system message has updated name
        system_content = messages[0]['content']
        assert "Bob" in system_content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
