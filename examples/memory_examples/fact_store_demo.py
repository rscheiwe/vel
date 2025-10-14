#!/usr/bin/env python3
"""
Fact Store — Full E2E Demo with Streaming

This example demonstrates:
1. Storing user preferences and facts in the fact store
2. Retrieving context before agent run
3. Running agent with streaming output
4. Saving results back to fact store

The Fact Store is for long-term structured data that persists across conversations:
- User preferences (theme, language, expertise)
- Project metadata (current project, technologies)
- Domain knowledge (company facts, endpoints)
"""

import asyncio
import os
from typing import Any, Dict
from datetime import datetime

from vel import Agent
from vel.core import ContextManager, MemoryConfig


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main():
    print_section("Fact Store Demo")

    # Ensure API key is set
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
        print("   or")
        print("   export OPENAI_API_KEY=sk-...")
        return

    # Choose provider based on available API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        model_config = {"provider": "anthropic", "model": "claude-sonnet-4"}
        print("Using Anthropic Claude Sonnet 4")
    else:
        model_config = {"provider": "openai", "model": "gpt-4o"}
        print("Using OpenAI GPT-4o")

    # ========================================================================
    # Step 1: Configure Fact Store
    # ========================================================================
    print_section("Step 1: Configure Fact Store")

    mem = MemoryConfig(
        mode="facts",              # Enable fact store only
        db_path=".vel/demo.db"     # SQLite database path
    )
    print(f"✓ Memory configured: mode={mem.mode}, db={mem.db_path}")

    # Initialize context manager with memory
    ctx = ContextManager(max_history=10)
    ctx.set_memory_config(mem)
    print("✓ Context manager initialized with fact store")

    # ========================================================================
    # Step 2: Store User Preferences
    # ========================================================================
    print_section("Step 2: Store User Preferences")

    user_namespace = "user:alice"

    # Store various preferences
    preferences = {
        "theme": "dark",
        "language": "Python",
        "expertise_level": "intermediate",
        "preferred_style": "concise with examples"
    }

    for key, value in preferences.items():
        ctx.fact_put(user_namespace, key, value)
        print(f"✓ Stored: {key} = {value}")

    # Store project history
    project_history = [
        {"name": "inventory-api", "status": "completed", "date": "2024-01-15"},
        {"name": "user-auth", "status": "completed", "date": "2024-02-20"}
    ]
    ctx.fact_put(user_namespace, "project_history", project_history)
    print(f"✓ Stored: project_history with {len(project_history)} projects")

    # ========================================================================
    # Step 3: Retrieve Context Before Agent Run
    # ========================================================================
    print_section("Step 3: Retrieve Context from Memory")

    # Retrieve user preferences
    theme = ctx.fact_get(user_namespace, "theme")
    language = ctx.fact_get(user_namespace, "language")
    expertise = ctx.fact_get(user_namespace, "expertise_level")
    style = ctx.fact_get(user_namespace, "preferred_style")
    history = ctx.fact_get(user_namespace, "project_history")

    print(f"Retrieved preferences:")
    print(f"  - Theme: {theme}")
    print(f"  - Language: {language}")
    print(f"  - Expertise: {expertise}")
    print(f"  - Style: {style}")
    print(f"  - Project History: {len(history) if history else 0} projects")

    # Build context-aware prompt
    context_info = f"""
USER CONTEXT (from memory):
- Preferred Language: {language}
- Expertise Level: {expertise}
- Communication Style: {style}
- Previous Projects: {', '.join([p['name'] for p in (history or [])])}
"""

    # ========================================================================
    # Step 4: Run Agent with Context (Streaming)
    # ========================================================================
    print_section("Step 4: Run Agent with Streaming Output")

    # Create agent
    agent = Agent(
        id="episodic-demo:v1",
        model=model_config,
        tools=[],
        policies={"max_steps": 5}
    )

    # User's question
    user_message = "Help me plan a new FastAPI project for a task management system. What should I focus on?"

    print(f"User: {user_message}\n")
    print("Assistant: ", end="", flush=True)

    # Build full message with context
    full_message = f"{context_info}\n\nUSER REQUEST:\n{user_message}"

    # Stream response
    run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_id = "session-alice"

    response_text = ""
    try:
        async for event in agent.run_stream({"message": full_message}, session_id=session_id):
            if event.get("type") == "text-delta":
                delta = event.get("delta", "")
                print(delta, end="", flush=True)
                response_text += delta
            elif event.get("type") == "error":
                print(f"\n❌ Error: {event.get('error')}")
                break

        print("\n")

    except Exception as e:
        print(f"\n❌ Error during agent run: {e}")
        return

    # ========================================================================
    # Step 5: Save Results Back to Memory
    # ========================================================================
    print_section("Step 5: Save Results to Memory")

    # Store the interaction summary
    interaction = {
        "date": datetime.now().isoformat(),
        "question": user_message,
        "response_summary": response_text[:200] + "..." if len(response_text) > 200 else response_text,
        "topics": ["FastAPI", "task management", "project planning"]
    }

    ctx.fact_put(user_namespace, "last_interaction", interaction)
    print("✓ Saved interaction summary")

    # Update project history with new project
    new_project = {
        "name": "task-management-api",
        "status": "planning",
        "date": datetime.now().strftime("%Y-%m-%d")
    }

    current_history = ctx.fact_get(user_namespace, "project_history") or []
    current_history.append(new_project)
    ctx.fact_put(user_namespace, "project_history", current_history)
    print(f"✓ Updated project history (now {len(current_history)} projects)")

    # ========================================================================
    # Step 6: Inspect Stored Memory
    # ========================================================================
    print_section("Step 6: Inspect All Stored Memory")

    all_items = ctx.fact_list(user_namespace, limit=50)

    print(f"Namespace: {user_namespace}")
    print(f"Total items: {len(all_items)}\n")

    for item in all_items:
        key = item["key"]
        value = item["value"]
        updated = datetime.fromtimestamp(item["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")

        # Format value for display
        if isinstance(value, list):
            value_str = f"[{len(value)} items]"
        elif isinstance(value, dict):
            value_str = f"{{...}} ({len(value)} keys)"
        else:
            value_str = str(value)[:50]

        print(f"  {key:20} = {value_str:30} (updated: {updated})")

    # ========================================================================
    # Demonstrate Retrieval Across Sessions
    # ========================================================================
    print_section("Demonstration: Retrieval Across Sessions")

    print("Simulating a new session with the same user...\n")

    # In a new session, we can still retrieve all context
    last_interaction = ctx.fact_get(user_namespace, "last_interaction")

    if last_interaction:
        print("Retrieved last interaction:")
        print(f"  Date: {last_interaction['date']}")
        print(f"  Question: {last_interaction['question']}")
        print(f"  Topics: {', '.join(last_interaction['topics'])}")
        print(f"\nThis context persists across sessions! ✨")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Summary")

    print("✓ Fact Store Demo Complete!")
    print("\nWhat we demonstrated:")
    print("  1. Stored user preferences and facts in namespaced fact store")
    print("  2. Retrieved context before agent run")
    print("  3. Ran agent with streaming output using retrieved context")
    print("  4. Saved interaction results back to fact store")
    print("  5. Showed persistence across sessions")
    print("\nFacts persist in:", mem.db_path)
    print("\nTry running this again to see persistent facts! 🎉")


if __name__ == "__main__":
    asyncio.run(main())
