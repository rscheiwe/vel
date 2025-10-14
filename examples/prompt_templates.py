"""
Demonstrates the flexible prompt module with various use cases.

Shows:
1. Basic prompt template with XML formatting
2. Environment-based prompts (dev vs prod)
3. Dynamic variable interpolation
4. SystemPromptBuilder for structured prompts
5. Integration with Agent
6. Custom context managers with prompts
"""
import asyncio
from dotenv import load_dotenv
from vel import (
    Agent,
    PromptTemplate,
    register_prompt,
    SystemPromptBuilder,
    PromptContextManager
)

load_dotenv()


# ================================================================================
# Example 1: Basic Prompt Template with XML
# ================================================================================

def example_basic_template():
    """Create a basic prompt template with XML structure"""
    print("=== EXAMPLE 1: Basic Prompt Template ===\n")

    template = PromptTemplate(
        id="chat-assistant:v1",
        system="""
        <system_instructions>
          <role>You are {{role_name}}, a helpful AI assistant.</role>
          <guidelines>
            - Be concise and clear
            - Provide accurate information
            - Admit when you don't know something
          </guidelines>
        </system_instructions>
        """,
        variables={"role_name": "Alex"}
    )

    # Render with default variables
    rendered = template.render()
    print("Rendered with default vars:")
    print(rendered)
    print()

    # Render with custom variables
    rendered = template.render(role_name="Sarah")
    print("Rendered with custom vars (role_name='Sarah'):")
    print(rendered)
    print("\n" + "=" * 80 + "\n")


# ================================================================================
# Example 2: Environment-Based Prompts
# ================================================================================

def example_environment_prompts():
    """Create prompts that vary by environment"""
    print("=== EXAMPLE 2: Environment-Based Prompts ===\n")

    template = PromptTemplate(
        id="customer-support:v1",
        environments={
            "dev": """
            <system_instructions>
              <role>You are a customer support assistant (DEV MODE).</role>
              <debug>Verbose logging enabled. Show reasoning steps.</debug>
            </system_instructions>
            """,
            "prod": """
            <system_instructions>
              <role>You are a professional customer support assistant.</role>
              <guidelines>
                - Be empathetic and helpful
                - Resolve issues efficiently
                - Escalate when necessary
              </guidelines>
            </system_instructions>
            """
        }
    )

    print("Development environment:")
    print(template.render(environment='dev'))
    print()

    print("Production environment:")
    print(template.render(environment='prod'))
    print("\n" + "=" * 80 + "\n")


# ================================================================================
# Example 3: SystemPromptBuilder for Structured Prompts
# ================================================================================

def example_system_prompt_builder():
    """Use SystemPromptBuilder for complex prompts"""
    print("=== EXAMPLE 3: SystemPromptBuilder ===\n")

    builder = SystemPromptBuilder()
    builder.add_role("You are a deployment automation assistant")
    builder.add_capabilities([
        "Deploy applications to cloud environments",
        "Rollback failed deployments",
        "Monitor deployment status"
    ])
    builder.add_guidelines([
        "Always verify environment before deploying",
        "Request approval for production deployments",
        "Log all actions for audit trail"
    ])
    builder.add_context(
        "company_info",
        "You work for Acme Corp. All deployments require security scanning."
    )

    prompt_text = builder.build()
    print("Built prompt:")
    print(prompt_text)
    print("\n" + "=" * 80 + "\n")


# ================================================================================
# Example 4: Integration with Agent
# ================================================================================

async def example_agent_integration():
    """Use prompt templates with Agent"""
    print("=== EXAMPLE 4: Agent Integration ===\n")

    # Register a prompt template
    template = PromptTemplate(
        id="friendly-chat:v1",
        system="""
        <system_instructions>
          <role>You are {{agent_name}}, a friendly conversational AI.</role>
          <personality>{{personality}}</personality>
          <guidelines>
            - Keep responses brief and engaging
            - Use a warm, friendly tone
            - Ask follow-up questions to keep conversation flowing
          </guidelines>
        </system_instructions>
        """,
        variables={
            "agent_name": "Buddy",
            "personality": "Cheerful and enthusiastic"
        }
    )
    register_prompt(template)

    # Create agent with prompt template
    agent = Agent(
        id='friendly-chat:v1',
        model={'provider': 'openai', 'model': 'gpt-4o'},
        prompt_id='friendly-chat:v1',
        prompt_vars={
            'agent_name': 'Buddy',
            'personality': 'Cheerful and enthusiastic, loves helping people'
        },
        prompt_env='prod'
    )

    print("Agent created with prompt template!")
    print(f"Prompt template: {template.id}")
    print(f"Variables: {agent.ctxmgr.prompt_manager.prompt_vars if hasattr(agent.ctxmgr, 'prompt_manager') else 'N/A'}")
    print()

    # Run agent (streaming mode)
    print("Running agent with prompt template...\n")
    full_response = []

    async for event in agent.run_stream({'message': 'Hello! What can you help me with?'}):
        if event['type'] == 'text-delta':
            full_response.append(event['delta'])
            print(event['delta'], end='', flush=True)

    print("\n\n" + "=" * 80 + "\n")


# ================================================================================
# Example 5: Versioned Prompts
# ================================================================================

def example_versioned_prompts():
    """Create multiple versions of a prompt"""
    print("=== EXAMPLE 5: Versioned Prompts ===\n")

    # Version 1 - Basic
    v1 = PromptTemplate(
        id="code-assistant:v1",
        system="""
        <system_instructions>
          <role>You are a code assistant.</role>
        </system_instructions>
        """
    )

    # Version 2 - Enhanced
    v2 = PromptTemplate(
        id="code-assistant:v2",
        system="""
        <system_instructions>
          <role>You are an expert code assistant specialized in {{language}}.</role>
          <capabilities>
            - Write clean, maintainable code
            - Explain complex concepts clearly
            - Suggest best practices and optimizations
          </capabilities>
        </system_instructions>
        """,
        variables={"language": "Python"}
    )

    register_prompt(v1)
    register_prompt(v2)

    print("Version 1:")
    print(v1.render())
    print()

    print("Version 2 (Python):")
    print(v2.render(language="Python"))
    print()

    print("Version 2 (JavaScript):")
    print(v2.render(language="JavaScript"))
    print("\n" + "=" * 80 + "\n")


# ================================================================================
# Example 6: Dynamic Context Injection
# ================================================================================

def example_dynamic_context():
    """Demonstrate dynamic context injection"""
    print("=== EXAMPLE 6: Dynamic Context Injection ===\n")

    template = PromptTemplate(
        id="rag-assistant:v1",
        system="""
        <system_instructions>
          <role>You are a knowledgeable assistant with access to a knowledge base.</role>
          <guidelines>
            - Answer questions based on provided context
            - Cite sources when available
            - Admit when information is not in the context
          </guidelines>
        </system_instructions>

        {% if retrieved_docs %}
        <context>
          <retrieved_knowledge>
            {{retrieved_docs}}
          </retrieved_knowledge>
        </context>
        {% endif %}

        {% if user_preferences %}
        <user_preferences>
          {{user_preferences}}
        </user_preferences>
        {% endif %}
        """,
        variables={
            "retrieved_docs": None,
            "user_preferences": None
        }
    )

    # Render without context
    print("Without context:")
    print(template.render())
    print()

    # Render with RAG context
    print("With retrieved documents:")
    print(template.render(
        retrieved_docs="Document 1: The sky is blue due to Rayleigh scattering.",
        user_preferences="Prefers technical explanations with scientific details"
    ))
    print("\n" + "=" * 80 + "\n")


# ================================================================================
# Example 7: PromptContextManager for Advanced Usage
# ================================================================================

async def example_prompt_context_manager():
    """Use PromptContextManager directly"""
    print("=== EXAMPLE 7: PromptContextManager ===\n")

    # Register a template
    template = PromptTemplate(
        id="learning-assistant:v1",
        system="""
        <system_instructions>
          <role>You are {{teacher_name}}, a patient learning assistant.</role>
          <teaching_style>{{teaching_style}}</teaching_style>
        </system_instructions>
        """,
        variables={
            "teacher_name": "Professor Smith",
            "teaching_style": "Socratic method - guide students with questions"
        }
    )
    register_prompt(template)

    # Create PromptContextManager directly
    ctx_mgr = PromptContextManager(
        prompt_id="learning-assistant:v1",
        prompt_vars={
            "teacher_name": "Dr. Johnson",
            "teaching_style": "Visual and example-based learning"
        },
        prompt_env='prod',
        max_history=10
    )

    # Simulate message flow
    run_id = "test-run-123"
    ctx_mgr.set_input(run_id, {"message": "Can you explain recursion?"})

    # Get messages with injected system prompt
    messages = ctx_mgr.messages_for_llm(run_id)

    print("Messages with system prompt injected:")
    for i, msg in enumerate(messages):
        print(f"\nMessage {i + 1} ({msg['role']}):")
        print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])

    print("\n" + "=" * 80 + "\n")


# ================================================================================
# Main
# ================================================================================

async def main():
    print("PROMPT MODULE EXAMPLES")
    print("=" * 80)
    print()

    # Run examples
    example_basic_template()
    example_environment_prompts()
    example_system_prompt_builder()
    await example_agent_integration()
    example_versioned_prompts()
    example_dynamic_context()
    await example_prompt_context_manager()

    print("All examples completed!")


if __name__ == '__main__':
    asyncio.run(main())
