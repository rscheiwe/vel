"""
Dynamic Prompt Creation Example

Demonstrates how to create and use prompts dynamically without global registration.
Shows:
- Creating prompts from PromptTemplate instances directly
- Passing prompts directly to agents (no registration needed!)
- Runtime prompt creation (simulating prompts from database/API)
- Environment-specific prompts
- Updating prompt variables at runtime
- Mixing with RAG context injection
"""
import asyncio
from vel import Agent, PromptTemplate


# Example 1: Simple dynamic prompt
def create_simple_prompt():
    """Basic prompt template with variables."""
    return PromptTemplate(
        id="assistant:v1",
        system="""
        <system_instructions>
          <role>You are {{role_name}}, a helpful AI assistant.</role>
          <expertise>Your area of expertise is {{expertise}}.</expertise>
          <guidelines>
            - Be concise and clear
            - Provide accurate information
            - Admit when you don't know something
          </guidelines>
        </system_instructions>
        """,
        variables={
            "role_name": "Alex",
            "expertise": "general knowledge"
        }
    )


# Example 2: Content review prompt (use case from user request)
def create_content_reviewer_prompt():
    """Prompt for reviewing content - demonstrates reusable templates."""
    return PromptTemplate(
        id="content-reviewer:v1",
        system="""
        <system_instructions>
          <role>You are an expert content reviewer.</role>
          <task>Review the following URL for content safety: {{url}}</task>
          <guidelines>
            - Analyze the content thoroughly
            - Check for harmful, misleading, or inappropriate content
            - Provide clear reasoning for your decision
          </guidelines>
          <output_format>
            Return your assessment as:
            - VERDICT: SAFE or UNSAFE
            - REASON: Brief explanation
            - CONFIDENCE: High, Medium, or Low
          </output_format>
        </system_instructions>
        """
    )


# Example 3: RAG-enabled prompt
def create_rag_prompt():
    """Prompt with conditional RAG context injection."""
    return PromptTemplate(
        id="rag-assistant:v1",
        system="""
        <system_instructions>
          <role>You are a knowledge base assistant for {{company_name}}.</role>
          <capabilities>
            - Answer questions from the knowledge base
            - Cite sources accurately
            - Admit uncertainty when needed
          </capabilities>
        </system_instructions>

        {% if rag_context %}
        <context>
          <retrieved_documents>
            {{rag_context}}
          </retrieved_documents>
        </context>
        {% endif %}

        <guidelines>
          - Prioritize information from the retrieved context
          - Be precise and factual
          - Provide citations when possible
        </guidelines>
        """,
        variables={
            "company_name": "Acme Corp",
            "rag_context": None
        }
    )


# Example 4: Environment-specific prompt
def create_environment_prompt():
    """Prompt with different versions for dev/staging/prod."""
    return PromptTemplate(
        id="deployment-agent:v1",
        environments={
            "dev": """
            <system>
              DEV MODE - All safety checks disabled
              You are a deployment assistant in development mode.
              Be verbose and include debug information.
            </system>
            """,
            "staging": """
            <system_instructions>
              <environment>Staging</environment>
              <safety_level>Medium</safety_level>
              <role>You are a deployment assistant.</role>
              <guidelines>
                - Validate configurations before deployment
                - Warn about potential issues
              </guidelines>
            </system_instructions>
            """,
            "prod": """
            <system_instructions>
              <environment>Production</environment>
              <safety_level>High</safety_level>
              <approval_required>true</approval_required>
              <role>You are a production deployment assistant.</role>
              <guidelines>
                - Require explicit approval for destructive actions
                - Double-check all configurations
                - Log all actions
                - Be concise and professional
              </guidelines>
            </system_instructions>
            """
        }
    )


# Example 5: Simulate loading prompt from database
def load_prompt_from_database(prompt_id: str) -> PromptTemplate:
    """
    Simulate loading a prompt from a database.
    In a real application, this would query your database.
    """
    # Simulated database of prompts
    db_prompts = {
        "sql-helper:v1": {
            "system": """
            <system_instructions>
              <role>You are a SQL query assistant.</role>
              <database>{{database_type}}</database>
              <schema>{{schema_info}}</schema>
              <guidelines>
                - Write efficient, safe queries
                - Explain query logic
                - Warn about potential performance issues
              </guidelines>
            </system_instructions>
            """,
            "variables": {
                "database_type": "PostgreSQL",
                "schema_info": "Not provided"
            }
        },
        "code-reviewer:v1": {
            "system": """
            <system_instructions>
              <role>You are a code review assistant.</role>
              <language>{{language}}</language>
              <style_guide>{{style_guide}}</style_guide>
              <focus_areas>
                - Security vulnerabilities
                - Performance issues
                - Code clarity
                - Best practices
              </focus_areas>
            </system_instructions>
            """,
            "variables": {
                "language": "Python",
                "style_guide": "PEP 8"
            }
        }
    }

    if prompt_id not in db_prompts:
        raise ValueError(f"Prompt '{prompt_id}' not found in database")

    prompt_data = db_prompts[prompt_id]
    return PromptTemplate(
        id=prompt_id,
        system=prompt_data["system"],
        variables=prompt_data.get("variables", {})
    )


async def main():
    print("=" * 70)
    print("Dynamic Prompt Creation Example")
    print("=" * 70)
    print()

    # ========================================
    # Pattern 1: Simple dynamic prompt
    # ========================================
    print("Pattern 1: Simple dynamic prompt")
    print("-" * 70)

    template = create_simple_prompt()

    agent = Agent(
        id='assistant',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        prompt=template,  # ✅ Pass directly, no registration!
        prompt_vars={
            'role_name': 'Sarah',
            'expertise': 'Python programming'
        }
    )

    # Verify the prompt was rendered correctly
    rendered = agent.ctxmgr.get_rendered_system_prompt()
    print(f"Rendered prompt preview:\n{rendered[:200]}...")
    print()

    # ========================================
    # Pattern 2: Content reviewer (reusable template)
    # ========================================
    print("Pattern 2: Content reviewer (reusable template)")
    print("-" * 70)

    reviewer_template = create_content_reviewer_prompt()

    # Review multiple URLs with the same template
    urls_to_review = [
        "https://example.com/article1",
        "https://example.com/article2",
    ]

    for url in urls_to_review:
        agent = Agent(
            id='content-reviewer',
            model={'provider': 'openai', 'model': 'gpt-4o-mini'},
            prompt=reviewer_template,
            prompt_vars={'url': url}
        )
        print(f"Created reviewer agent for: {url}")
    print()

    # ========================================
    # Pattern 3: RAG-enabled prompt
    # ========================================
    print("Pattern 3: RAG-enabled prompt with conditional context")
    print("-" * 70)

    rag_template = create_rag_prompt()

    # Without RAG context
    agent_no_rag = Agent(
        id='rag-assistant',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        prompt=rag_template,
        prompt_vars={'company_name': 'TechCorp'}
    )
    print("Agent without RAG context:")
    print(f"  Has context section: {'rag_context' in agent_no_rag.ctxmgr.get_rendered_system_prompt()}")

    # With RAG context
    agent_with_rag = Agent(
        id='rag-assistant',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        prompt=rag_template,
        prompt_vars={
            'company_name': 'TechCorp',
            'rag_context': 'Document 1: Product pricing is $99/month...'
        }
    )
    print("Agent with RAG context:")
    print(f"  Has context section: {'retrieved_documents' in agent_with_rag.ctxmgr.get_rendered_system_prompt()}")
    print()

    # ========================================
    # Pattern 4: Environment-specific prompts
    # ========================================
    print("Pattern 4: Environment-specific prompts")
    print("-" * 70)

    env_template = create_environment_prompt()

    for env in ['dev', 'staging', 'prod']:
        agent = Agent(
            id='deployment-agent',
            model={'provider': 'openai', 'model': 'gpt-4o-mini'},
            prompt=env_template,
            prompt_env=env  # Select environment
        )
        rendered = agent.ctxmgr.get_rendered_system_prompt()
        preview = rendered[:60].replace('\n', ' ')
        print(f"  {env}: {preview}...")
    print()

    # ========================================
    # Pattern 5: Load from database (runtime creation)
    # ========================================
    print("Pattern 5: Load prompts from database at runtime")
    print("-" * 70)

    # Simulate API request: "Create an agent with prompt sql-helper:v1"
    prompt_id = "sql-helper:v1"
    template = load_prompt_from_database(prompt_id)

    agent = Agent(
        id='sql-helper',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        prompt=template,
        prompt_vars={
            'database_type': 'MySQL',
            'schema_info': 'users(id, name, email), orders(id, user_id, total)'
        }
    )
    print(f"Created agent from database prompt: {prompt_id}")
    print(f"Rendered prompt preview:\n{agent.ctxmgr.get_rendered_system_prompt()[:200]}...")
    print()

    # ========================================
    # Pattern 6: Update variables at runtime
    # ========================================
    print("Pattern 6: Update prompt variables at runtime")
    print("-" * 70)

    template = create_simple_prompt()
    agent = Agent(
        id='assistant',
        model={'provider': 'openai', 'model': 'gpt-4o-mini'},
        prompt=template,
        prompt_vars={'role_name': 'Alex', 'expertise': 'general'}
    )

    print(f"Initial role: Alex")

    # Update variables at runtime
    agent.ctxmgr.update_prompt_vars(
        role_name='Dr. Smith',
        expertise='medical information'
    )

    updated_prompt = agent.ctxmgr.get_rendered_system_prompt()
    print(f"Updated role: Dr. Smith")
    print(f"Updated expertise in prompt: {'medical information' in updated_prompt}")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ All prompts created dynamically (no global registration)")
    print("✅ Prompts scoped to agent instance (no global state pollution)")
    print("✅ Supports Jinja2 templating with variables")
    print("✅ Environment-specific prompts (dev/staging/prod)")
    print("✅ Can load prompts from database/API at runtime")
    print("✅ Variables can be updated at runtime")
    print("✅ Conditional context injection (RAG)")
    print()
    print("Use cases enabled:")
    print("  - User-created prompts from UI")
    print("  - Prompts stored in database")
    print("  - A/B testing different prompts")
    print("  - Per-tenant prompt customization")
    print("  - Dynamic RAG context injection")


if __name__ == '__main__':
    asyncio.run(main())
