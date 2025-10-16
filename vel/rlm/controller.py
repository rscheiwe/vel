"""
RLM Controller

Main orchestrator for Recursive Language Model execution.
"""
from __future__ import annotations
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import json

from .config import RlmConfig
from .scratchpad import Scratchpad, Note
from .budget import Budget
from .context_store import ContextStore
from .prompts import (
    get_planner_prompt,
    get_writer_prompt,
    format_scratchpad_for_writer,
    format_scratchpad_update
)
from .tools import context_probe, rlm_call, python_exec, get_tool_schemas
from .utils import (
    detect_final,
    parse_tool_calls,
    extract_text_content,
    format_tool_result,
    validate_tool_args
)


class RlmController:
    """
    Controller for RLM (Recursive Language Model) execution.

    Manages the iterative reasoning loop with context probing, scratchpad notes,
    budget tracking, and FINAL() detection.
    """

    def __init__(
        self,
        config: RlmConfig,
        agent: Any,  # Agent instance
        depth: Optional[int] = None
    ):
        """
        Initialize RLM controller.

        Args:
            config: RLM configuration
            agent: Agent instance (for calling LLM)
            depth: Current recursion depth (overrides config.depth)
        """
        self.config = config
        self.agent = agent
        self.depth = depth if depth is not None else config.depth

        # State
        self.context_store: Optional[ContextStore] = None
        self.scratchpad: Optional[Scratchpad] = None
        self.budget: Optional[Budget] = None
        self.messages: List[Dict[str, Any]] = []

    async def run(
        self,
        user_query: str,
        context_refs: Union[str, List[str], List[Dict[str, Any]]],
        session_id: Optional[str] = None,
        parent_scratchpad: Optional[Scratchpad] = None
    ) -> Dict[str, Any]:
        """
        Run RLM execution (non-streaming).

        Args:
            user_query: User question
            context_refs: Context references to load
            session_id: Optional session ID
            parent_scratchpad: Optional parent scratchpad (for recursive calls)

        Returns:
            Result dict with answer, scratchpad, meta, etc.
        """
        # Initialize
        self._initialize(context_refs, parent_scratchpad)

        # Control loop
        final_answer = await self._control_loop(user_query, session_id)

        # Synthesis if writer model configured
        if self.config.writer_model and final_answer:
            final_answer = await self._synthesize(user_query, final_answer)

        return {
            'answer': final_answer,
            'scratchpad_notes': self.scratchpad.notes if self.scratchpad else [],
            'meta': {
                'rlm_used': True,
                'budget': self.budget.to_dict() if self.budget else {},
                'context_summary': self.context_store.get_chunks_summary() if self.context_store else {},
                'depth': self.depth
            }
        }

    async def run_stream(
        self,
        user_query: str,
        context_refs: Union[str, List[str], List[Dict[str, Any]]],
        session_id: Optional[str] = None,
        parent_scratchpad: Optional[Scratchpad] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run RLM execution (streaming).

        Args:
            user_query: User question
            context_refs: Context references to load
            session_id: Optional session ID
            parent_scratchpad: Optional parent scratchpad (for recursive calls)

        Yields:
            Stream events (RLM-specific events if config.stream_events is True)
        """
        # Initialize
        self._initialize(context_refs, parent_scratchpad)

        # Emit start event
        if self.config.stream_events:
            yield {
                'type': 'data-rlm-start',
                'data': {
                    'config': self.config.to_dict(),
                    'depth': self.depth
                }
            }

        # Control loop with streaming
        final_answer = None
        async for event in self._control_loop_stream(user_query, session_id):
            yield event
            if event.get('type') == 'data-rlm-final':
                final_answer = event.get('data', {}).get('answer')

        # Synthesis if writer model configured
        if self.config.writer_model and final_answer:
            if self.config.stream_events:
                yield {'type': 'data-rlm-synthesis', 'data': {'status': 'starting'}}

            synthesized = await self._synthesize(user_query, final_answer)

            if self.config.stream_events:
                yield {'type': 'data-rlm-synthesis', 'data': {'status': 'complete', 'answer': synthesized}}

            final_answer = synthesized

        # Emit final metadata
        if self.config.stream_events:
            yield {
                'type': 'data-rlm-complete',
                'data': {
                    'answer': final_answer,
                    'meta': {
                        'budget': self.budget.to_dict() if self.budget else {},
                        'context_summary': self.context_store.get_chunks_summary() if self.context_store else {}
                    }
                }
            }

    def _initialize(
        self,
        context_refs: Union[str, List[str], List[Dict[str, Any]]],
        parent_scratchpad: Optional[Scratchpad]
    ):
        """Initialize context store, scratchpad, and budget."""

        # Context store
        self.context_store = ContextStore(
            chunk_size=self.config.tools.get('probe_max_bytes', 4096)
        )
        num_chunks = self.context_store.load(context_refs)

        # Scratchpad
        if parent_scratchpad:
            # Inherit from parent
            self.scratchpad = Scratchpad(max_notes=self.config.notes_cap)
            self.scratchpad.merge(parent_scratchpad.notes, dedup=True)
        else:
            self.scratchpad = Scratchpad(max_notes=self.config.notes_cap)

        # Budget
        max_steps = (
            self.config.budgets['max_steps_child'] if self.depth < self.config.depth
            else self.config.budgets['max_steps_root']
        )
        self.budget = Budget(
            max_steps=max_steps,
            max_tokens=self.config.budgets['max_tokens_total'],
            max_cost=self.config.budgets['max_cost_usd'],
            depth=self.depth
        )

        # Messages
        self.messages = []

    async def _control_loop(
        self,
        user_query: str,
        session_id: Optional[str]
    ) -> str:
        """
        Run control loop (non-streaming).

        Returns:
            Final answer
        """
        # System prompt
        system_prompt = get_planner_prompt(
            depth=self.config.depth,
            depth_left=self.depth,
            tools_enabled=self.config.tools
        )

        self.messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ]

        final_answer = None
        iteration = 0
        max_iterations = self.config.budgets.get('max_steps_root', 12) * 2  # Allow 2x iterations vs tool steps

        while not final_answer:
            iteration += 1

            # Check iteration limit
            if iteration > max_iterations:
                final_answer = self._best_effort_answer(f"max iterations ({max_iterations}) exceeded")
                break

            # Check budget
            is_exhausted, reason = self.budget.exhausted()
            if is_exhausted:
                # Best-effort return
                final_answer = self._best_effort_answer(reason)
                break

            # Call LLM with tools (with error handling)
            try:
                response = await self._call_llm()
            except Exception as e:
                # LLM call failed - return best effort answer
                final_answer = self._best_effort_answer(f"LLM error: {str(e)}")
                break

            # Update budget
            if 'usage' in response:
                self.budget.bump(response)

            # Handle normalized provider response format
            # Format: {'tool': 'name', 'args': {...}} OR {'done': True, 'answer': 'text'}

            text_content = response.get('answer', '')

            # Check for FINAL()
            has_final, final_type, final_value = detect_final(text_content)
            if has_final:
                final_answer = final_value
                break

            # Check if tool call in response
            tool_name = response.get('tool')
            tool_args = response.get('args', {})

            if not tool_name:
                # No tool call and no FINAL -> add assistant message and continue
                if text_content:
                    self.messages.append({'role': 'assistant', 'content': text_content})
                continue

            # Add assistant message (text if any)
            if text_content:
                self.messages.append({'role': 'assistant', 'content': text_content})

            # Validate args
            is_valid, error_msg = validate_tool_args(tool_name, tool_args)
            if not is_valid:
                tool_result = {'error': error_msg}
            else:
                # Dispatch tool
                tool_result = await self._dispatch_tool(tool_name, tool_args, session_id)

            # Increment step counter
            self.budget.bump_step()

            # Update scratchpad from tool result
            self._update_scratchpad_from_tool_result(tool_name, tool_result)

            # Add tool result to messages as user message (like ContextManager does)
            tool_result_text = format_tool_result(tool_result)
            self.messages.append({
                'role': 'user',
                'content': f"Tool {tool_name} returned: {tool_result_text}"
            })

            # Add scratchpad update
            scratchpad_text = self.scratchpad.to_bullets(limit=self.config.notes_window)
            scratchpad_update = format_scratchpad_update(scratchpad_text, self.config.notes_window)
            self.messages.append({'role': 'system', 'content': scratchpad_update})

        return final_answer or ""

    async def _control_loop_stream(
        self,
        user_query: str,
        session_id: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run control loop (streaming).

        Yields:
            Stream events
        """
        # System prompt
        system_prompt = get_planner_prompt(
            depth=self.config.depth,
            depth_left=self.depth,
            tools_enabled=self.config.tools
        )

        self.messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ]

        final_answer = None
        step = 0
        max_iterations = self.config.budgets.get('max_steps_root', 12) * 2

        while not final_answer:
            step += 1

            # Check iteration limit
            if step > max_iterations:
                final_answer = self._best_effort_answer(f"max iterations ({max_iterations}) exceeded")
                if self.config.stream_events:
                    yield {'type': 'data-rlm-final', 'data': {'answer': final_answer, 'reason': 'max_iterations'}}
                break

            # Check budget
            is_exhausted, reason = self.budget.exhausted()
            if is_exhausted:
                if self.config.stream_events:
                    yield {'type': 'data-rlm-budget-exhausted', 'data': {'reason': reason}}

                final_answer = self._best_effort_answer(reason)

                if self.config.stream_events:
                    yield {'type': 'data-rlm-final', 'data': {'answer': final_answer, 'reason': 'budget_exhausted'}}
                break

            # Emit step start
            if self.config.stream_events:
                yield {
                    'type': 'data-rlm-step-start',
                    'data': {
                        'step': step,
                        'budget': self.budget.to_dict()
                    }
                }

            # Call LLM with error handling
            try:
                response = await self._call_llm()
                if 'usage' in response:
                    self.budget.bump(response)
            except Exception as e:
                # LLM call failed - emit event and try to finish gracefully
                if self.config.stream_events:
                    yield {
                        'type': 'data-rlm-error',
                        'data': {
                            'error': str(e),
                            'step': step
                        }
                    }
                # Return best effort answer
                final_answer = self._best_effort_answer(f"LLM error: {str(e)}")
                if self.config.stream_events:
                    yield {'type': 'data-rlm-final', 'data': {'answer': final_answer, 'reason': 'error'}}
                break

            # Handle normalized provider response format
            text_content = response.get('answer', '')

            # Check FINAL()
            has_final, final_type, final_value = detect_final(text_content)
            if has_final:
                final_answer = final_value

                if self.config.stream_events:
                    yield {
                        'type': 'data-rlm-final',
                        'data': {
                            'answer': final_answer,
                            'final_type': final_type
                        }
                    }
                break

            # Check if tool call in response
            tool_name = response.get('tool')
            tool_args = response.get('args', {})

            if not tool_name:
                if text_content:
                    self.messages.append({'role': 'assistant', 'content': text_content})
                continue

            if text_content:
                self.messages.append({'role': 'assistant', 'content': text_content})

            # Emit probe event
            if self.config.stream_events:
                yield {
                    'type': 'data-rlm-probe',
                    'data': {
                        'tool': tool_name,
                        'args': tool_args
                    }
                }

            # Validate and dispatch
            is_valid, error_msg = validate_tool_args(tool_name, tool_args)
            if not is_valid:
                tool_result = {'error': error_msg}
            else:
                tool_result = await self._dispatch_tool(tool_name, tool_args, session_id)

            # Increment step counter
            self.budget.bump_step()

            # Update scratchpad
            self._update_scratchpad_from_tool_result(tool_name, tool_result)

            # Emit note events
            if self.config.stream_events and 'preview' in tool_result:
                preview = tool_result['preview'][:200]
                yield {
                    'type': 'data-rlm-note',
                    'data': {
                        'text': f"{tool_name} result: {preview}...",
                        'source_hint': f"tool:{tool_name}"
                    }
                }

            # Add to messages as user message (like ContextManager does)
            tool_result_text = format_tool_result(tool_result)
            self.messages.append({
                'role': 'user',
                'content': f"Tool {tool_name} returned: {tool_result_text}"
            })

            # Scratchpad update
            scratchpad_text = self.scratchpad.to_bullets(limit=self.config.notes_window)
            scratchpad_update = format_scratchpad_update(scratchpad_text, self.config.notes_window)
            self.messages.append({'role': 'system', 'content': scratchpad_update})

            # Emit step finish
            if self.config.stream_events:
                yield {
                    'type': 'data-rlm-step-finish',
                    'data': {
                        'step': step,
                        'budget': self.budget.to_dict()
                    }
                }

    async def _call_llm(self) -> Dict[str, Any]:
        """
        Call LLM with current messages and tools.

        Returns:
            LLM response
        """
        # Get model config
        model_cfg = self.config.control_model or self.agent.model_cfg

        # Get provider
        provider_name = model_cfg['provider']
        provider = self.agent.providers.get(provider_name)

        # Get tools
        tools = self._get_active_tools()

        # Call provider
        response = await provider.generate(
            messages=self.messages,
            model=model_cfg['model'],
            tools=tools if tools else None
        )
        return response

    def _get_active_tools(self) -> Dict[str, Any]:
        """
        Get active tool schemas based on config and depth.

        Returns:
            Dict of tool schemas in ToolRegistry format: {tool_name: {input: schema, output: schema}}
        """
        all_tools = get_tool_schemas()
        active_tools = {}

        for tool in all_tools:
            tool_name = tool['name']

            # context_probe: always available
            if tool_name == 'context_probe':
                active_tools[tool_name] = {
                    'input': tool['parameters'],
                    'output': {}  # RLM tools don't have strict output schemas
                }

            # rlm_call: only if depth > 0
            elif tool_name == 'rlm_call' and self.depth > 0:
                active_tools[tool_name] = {
                    'input': tool['parameters'],
                    'output': {}
                }

            # python_exec: only if enabled
            elif tool_name == 'python_exec' and self.config.tools.get('allow_exec', False):
                active_tools[tool_name] = {
                    'input': tool['parameters'],
                    'output': {}
                }

        return active_tools

    async def _dispatch_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Dispatch tool call.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            session_id: Session ID

        Returns:
            Tool result
        """
        try:
            if tool_name == 'context_probe':
                return context_probe(
                    context_store=self.context_store,
                    **tool_args
                )

            elif tool_name == 'rlm_call':
                # Parse args - handle both dict and string formats
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                return await rlm_call(
                    query=tool_args['query'],
                    context_slice=tool_args.get('context_slice', ''),
                    depth_left=self.depth - 1,
                    controller=self,
                    agent=self.agent,
                    session_id=session_id
                )

            elif tool_name == 'python_exec':
                return python_exec(
                    code=tool_args['code'],
                    context=self.context_store.get_full_text() if self.context_store else '',
                    max_bytes=self.config.tools.get('probe_max_bytes', 4096)
                )

            else:
                return {'error': f'Unknown tool: {tool_name}'}

        except Exception as e:
            return {'error': f'Tool execution error: {str(e)}'}

    def _update_scratchpad_from_tool_result(self, tool_name: str, tool_result: Dict[str, Any]):
        """
        Extract notes from tool result and add to scratchpad.

        Args:
            tool_name: Tool name
            tool_result: Tool result dict
        """
        if 'error' in tool_result:
            # Add error note
            self.scratchpad.add(
                text=f"Error in {tool_name}: {tool_result['error']}",
                source_hint=f"tool:{tool_name}"
            )
            return

        # Extract relevant info based on tool type
        if tool_name == 'context_probe':
            kind = tool_result.get('meta', {}).get('kind', 'unknown')
            preview = tool_result.get('preview', '')[:200]

            if kind == 'search':
                num_results = tool_result.get('meta', {}).get('num_results', 0)
                self.scratchpad.add(
                    text=f"Search returned {num_results} results",
                    source_hint=f"probe:search"
                )
            else:
                self.scratchpad.add(
                    text=f"{kind}: {preview}...",
                    source_hint=f"probe:{kind}"
                )

        elif tool_name == 'rlm_call':
            answer = tool_result.get('answer', '')
            self.scratchpad.add(
                text=f"Recursive call result: {answer[:200]}",
                source_hint="rlm_call"
            )

            # Merge child notes
            notes = tool_result.get('notes', [])
            for note_text in notes:
                note = Note(text=note_text, source_hint="rlm_call:child")
                self.scratchpad.merge([note], dedup=True)

        elif tool_name == 'python_exec':
            preview = tool_result.get('preview', '')[:200]
            self.scratchpad.add(
                text=f"Exec result: {preview}",
                source_hint="python_exec"
            )

    def _best_effort_answer(self, reason: str) -> str:
        """
        Generate best-effort answer when budget exhausted.

        Args:
            reason: Exhaustion reason

        Returns:
            Best-effort answer from scratchpad
        """
        if not self.scratchpad or len(self.scratchpad) == 0:
            return f"(budget exhausted: {reason}) Unable to provide answer - no notes collected."

        # Return scratchpad summary
        notes_text = self.scratchpad.to_bullets()
        return f"(budget exhausted: {reason}) Partial findings:\n{notes_text}"

    async def _synthesize(self, user_query: str, raw_answer: str) -> str:
        """
        Run writer synthesis pass.

        Args:
            user_query: Original user query
            raw_answer: Raw answer from control loop

        Returns:
            Synthesized answer
        """
        # Get writer model
        model_cfg = self.config.writer_model or self.config.control_model or self.agent.model_cfg

        # Get provider
        provider_name = model_cfg['provider']
        provider = self.agent.providers.get(provider_name)

        # Format scratchpad for writer
        scratchpad_text = self.scratchpad.to_bullets()
        writer_user_prompt = format_scratchpad_for_writer(scratchpad_text, user_query)

        # Build messages
        messages = [
            {'role': 'system', 'content': get_writer_prompt()},
            {'role': 'user', 'content': writer_user_prompt}
        ]

        # Call writer
        response = await provider.generate(
            messages=messages,
            model=model_cfg['model'],
            tools=None  # No tools for writer
        )

        # Extract synthesized answer
        synthesized = extract_text_content(response)

        return synthesized or raw_answer
