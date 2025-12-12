"""
Stream Protocol Events
Based on Vercel AI SDK stream protocol: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""
from __future__ import annotations
from typing import Any, Dict, Literal, Optional
from dataclasses import dataclass

# Event types - V5 UI Stream Protocol
# Note: Custom data-* events are supported via DataEvent class (not in this Literal)
EventType = Literal[
    'start',
    'text-start',
    'text-delta',
    'text-end',
    'reasoning-start',
    'reasoning-delta',
    'reasoning-end',
    'tool-input-start',
    'tool-input-delta',
    'tool-input-available',  # V5 UI Stream Protocol
    'tool-output-available',  # V5 UI Stream Protocol
    'response-metadata',
    'source',
    'file',
    'start-step',  # V5 UI Stream Protocol (multi-step agents)
    'finish-step',  # V5 UI Stream Protocol (multi-step agents)
    'finish-message',
    'finish',  # V5 UI Stream Protocol (end of generation)
    'error'
]

@dataclass
class StreamEvent:
    """Base stream event"""
    type: EventType

    def to_dict(self) -> Dict[str, Any]:
        return {'type': self.type}

@dataclass
class StartEvent(StreamEvent):
    """Message start event"""
    type: Literal['start'] = 'start'
    message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.message_id:
            d['messageId'] = self.message_id
        return d

@dataclass
class TextStartEvent(StreamEvent):
    """Text chunk start"""
    type: Literal['text-start'] = 'text-start'
    block_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'id': self.block_id}

@dataclass
class TextDeltaEvent(StreamEvent):
    """Text chunk delta"""
    type: Literal['text-delta'] = 'text-delta'
    block_id: str = ''
    delta: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'id': self.block_id, 'delta': self.delta}

@dataclass
class TextEndEvent(StreamEvent):
    """Text chunk end"""
    type: Literal['text-end'] = 'text-end'
    block_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'id': self.block_id}

@dataclass
class ReasoningStartEvent(StreamEvent):
    """Reasoning start"""
    type: Literal['reasoning-start'] = 'reasoning-start'
    block_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'id': self.block_id}

@dataclass
class ReasoningDeltaEvent(StreamEvent):
    """Reasoning delta"""
    type: Literal['reasoning-delta'] = 'reasoning-delta'
    block_id: str = ''
    delta: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'id': self.block_id, 'delta': self.delta}

@dataclass
class ReasoningEndEvent(StreamEvent):
    """Reasoning end"""
    type: Literal['reasoning-end'] = 'reasoning-end'
    block_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'id': self.block_id}

@dataclass
class ToolInputStartEvent(StreamEvent):
    """Tool call input start"""
    type: Literal['tool-input-start'] = 'tool-input-start'
    tool_call_id: str = ''
    tool_name: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'toolCallId': self.tool_call_id, 'toolName': self.tool_name}

@dataclass
class ToolInputDeltaEvent(StreamEvent):
    """Tool call input delta (streaming JSON args)"""
    type: Literal['tool-input-delta'] = 'tool-input-delta'
    tool_call_id: str = ''
    input_delta: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'toolCallId': self.tool_call_id, 'inputTextDelta': self.input_delta}

@dataclass
class ToolInputAvailableEvent(StreamEvent):
    """Tool input available event (input fully available for execution)

    Matches Vercel AI SDK V5 UI Stream Protocol.
    Frontend components (useChat, useCompletion) expect this event type.
    """
    type: Literal['tool-input-available'] = 'tool-input-available'
    tool_call_id: str = ''
    tool_name: str = ''
    input: Dict[str, Any] = None
    provider_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}

    def to_dict(self) -> Dict[str, Any]:
        d = {
            **super().to_dict(),
            'toolCallId': self.tool_call_id,
            'toolName': self.tool_name,
            'input': self.input  # V5 UI Protocol uses 'input'
        }
        if self.provider_metadata:
            d['providerMetadata'] = self.provider_metadata
        return d

@dataclass
class ToolOutputAvailableEvent(StreamEvent):
    """Tool output available event (execution result ready)

    Matches Vercel AI SDK V5 UI Stream Protocol.
    Frontend components (useChat, useCompletion) expect this event type.

    NOTE: providerMetadata removed to strictly match AI SDK v5 spec.
    Only type, toolCallId, and output are officially supported.
    """
    type: Literal['tool-output-available'] = 'tool-output-available'
    tool_call_id: str = ''
    output: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'toolCallId': self.tool_call_id, 'output': self.output}

@dataclass
class StepStartEvent(StreamEvent):
    """Start a reasoning/agent step

    Emitted at the beginning of each agent step in multi-step agent patterns.
    Matches Vercel AI SDK V5 UI Stream Protocol for agent steps.
    """
    type: Literal['start-step'] = 'start-step'

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

@dataclass
class StepFinishEvent(StreamEvent):
    """Finish a reasoning/agent step

    Emitted at the end of each agent step in multi-step agent patterns.
    Matches Vercel AI SDK V5 UI Stream Protocol for agent steps.
    """
    type: Literal['finish-step'] = 'finish-step'

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

@dataclass
class FinishEvent(StreamEvent):
    """Finish the entire generation

    Emitted at the very end of a streaming response, after all steps complete.
    Matches Vercel AI SDK V5 UI Stream Protocol.
    """
    type: Literal['finish'] = 'finish'

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

@dataclass
class FinishMessageEvent(StreamEvent):
    """Finish the message"""
    type: Literal['finish-message'] = 'finish-message'
    finish_reason: str = 'stop'

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'finishReason': self.finish_reason}

@dataclass
class ErrorEvent(StreamEvent):
    """Error event with detailed error context

    Provides comprehensive error information for debugging and logging.
    Includes HTTP status, provider details, and error categorization.
    """
    type: Literal['error'] = 'error'
    error: str = ''
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    status_code: Optional[int] = None  # HTTP status code (e.g., 400, 401, 429, 500)
    provider: Optional[str] = None  # Provider name (e.g., 'openai', 'anthropic', 'google')
    details: Optional[Dict[str, Any]] = None  # Additional error details from provider

    def to_dict(self) -> Dict[str, Any]:
        d = {**super().to_dict(), 'errorText': self.error}
        if self.error_code:
            d['errorCode'] = self.error_code
        if self.error_type:
            d['errorType'] = self.error_type
        if self.status_code:
            d['statusCode'] = self.status_code
        if self.provider:
            d['provider'] = self.provider
        if self.details:
            d['details'] = self.details
        return d

@dataclass
class ResponseMetadataEvent(StreamEvent):
    """Response metadata (usage, model info, timing)"""
    type: Literal['response-metadata'] = 'response-metadata'
    id: Optional[str] = None
    model_id: Optional[str] = None
    timestamp: Optional[str] = None  # ISO 8601
    usage: Optional[Dict[str, int]] = None  # {promptTokens, completionTokens, totalTokens}

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.id:
            d['id'] = self.id
        if self.model_id:
            d['modelId'] = self.model_id
        if self.timestamp:
            d['timestamp'] = self.timestamp
        if self.usage:
            d['usage'] = self.usage
        return d

@dataclass
class SourceEvent(StreamEvent):
    """Source/citation event (web search results, document references)"""
    type: Literal['source'] = 'source'
    sources: list[Dict[str, Any]] = None  # [{type, url, title, snippet}, ...]

    def __post_init__(self):
        if self.sources is None:
            self.sources = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'sources': self.sources
        }

@dataclass
class FileEvent(StreamEvent):
    """File attachment event (inline data, images, PDFs)"""
    type: Literal['file'] = 'file'
    content: Any = None  # base64 string or bytes
    name: str = ''
    mime_type: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'content': self.content,
            'name': self.name,
            'mimeType': self.mime_type
        }

@dataclass
class DataEvent(StreamEvent):
    """Custom data event (data-* pattern)

    Supports custom event types following the data-* naming convention.
    Used for application-specific data streaming (progress, status, notifications, etc.).

    Examples:
        - data-notification: UI notifications
        - data-progress: Progress updates
        - data-stage-data: Multi-step stage transitions
        - data-metrics: Real-time metrics

    The `transient` flag controls whether the event is added to message history:
        - transient=True: Event sent to client but NOT saved to message history
        - transient=False (default): Event saved to message history

    Matches Vercel AI SDK V5 custom data pattern.
    """
    type: str = 'data'  # Should follow pattern: data-{customName}
    data: Any = None  # Custom payload (any JSON-serializable value)
    transient: bool = False  # If True, not added to message history

    def to_dict(self) -> Dict[str, Any]:
        d = {**super().to_dict(), 'data': self.data}
        if self.transient:
            d['transient'] = True
        return d


# ============================================================================
# RLM (Recursive Language Model) Events
# ============================================================================
# Custom events for RLM middleware that provides recursive reasoning over long contexts

@dataclass
class RlmStartEvent(DataEvent):
    """RLM execution start event"""
    type: str = 'data-rlm-start'
    config: Optional[Dict[str, Any]] = None
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'config': self.config,
                'depth': self.depth
            },
            'transient': True
        }


@dataclass
class RlmStepStartEvent(DataEvent):
    """RLM reasoning step start"""
    type: str = 'data-rlm-step-start'
    step: int = 0
    budget: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'step': self.step,
                'budget': self.budget
            },
            'transient': True
        }


@dataclass
class RlmStepFinishEvent(DataEvent):
    """RLM reasoning step finish"""
    type: str = 'data-rlm-step-finish'
    step: int = 0
    budget: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'step': self.step,
                'budget': self.budget
            },
            'transient': True
        }


@dataclass
class RlmProbeEvent(DataEvent):
    """RLM context probe execution"""
    type: str = 'data-rlm-probe'
    tool: str = ''
    args: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'tool': self.tool,
                'args': self.args
            },
            'transient': True
        }


@dataclass
class RlmNoteEvent(DataEvent):
    """RLM scratchpad note added"""
    type: str = 'data-rlm-note'
    text: str = ''
    source_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'text': self.text,
                'source_hint': self.source_hint
            },
            'transient': True
        }


@dataclass
class RlmRecursiveCallEvent(DataEvent):
    """RLM recursive call (rlm_call tool)"""
    type: str = 'data-rlm-recursive-call'
    query: str = ''
    depth: int = 0
    status: str = 'starting'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'query': self.query,
                'depth': self.depth,
                'status': self.status
            },
            'transient': True
        }


@dataclass
class RlmSynthesisEvent(DataEvent):
    """RLM writer synthesis phase"""
    type: str = 'data-rlm-synthesis'
    status: str = 'starting'
    answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {'status': self.status}
        if self.answer:
            data['answer'] = self.answer
        return {
            'type': self.type,
            'data': data,
            'transient': True
        }


@dataclass
class RlmFinalEvent(DataEvent):
    """RLM FINAL() detected - execution complete"""
    type: str = 'data-rlm-final'
    answer: str = ''
    final_type: Optional[str] = None  # 'direct' or 'var'
    reason: Optional[str] = None  # e.g., 'budget_exhausted'

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'answer': self.answer,
            'final_type': self.final_type
        }
        if self.reason:
            data['reason'] = self.reason
        return {
            'type': self.type,
            'data': data,
            'transient': False  # Final answer should be saved to history
        }


@dataclass
class RlmBudgetExhaustedEvent(DataEvent):
    """RLM budget exhausted"""
    type: str = 'data-rlm-budget-exhausted'
    reason: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {'reason': self.reason},
            'transient': True
        }


@dataclass
class RlmCompleteEvent(DataEvent):
    """RLM execution complete with metadata"""
    type: str = 'data-rlm-complete'
    answer: str = ''
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'answer': self.answer,
                'meta': self.meta
            },
            'transient': True
        }


# ============================================================================
# Structured Output Streaming Events
# ============================================================================
# Custom events for streaming structured output (arrays and objects)
# These work with useChat's onData handler when output_type is set

@dataclass
class ObjectElementEvent(DataEvent):
    """Emitted when a complete array element is parsed and validated.

    Used when output_type is List[X] - streams validated elements one-by-one.

    Example:
        output_type=List[AIAgent] will emit this for each agent as it's parsed.
    """
    type: str = 'data-object-element'
    index: int = 0
    element: Any = None  # The validated Pydantic instance (serialized to dict)

    def to_dict(self) -> Dict[str, Any]:
        # Serialize Pydantic model to dict if needed
        element_data = self.element
        if hasattr(self.element, 'model_dump'):
            element_data = self.element.model_dump()
        elif hasattr(self.element, 'dict'):
            element_data = self.element.dict()

        return {
            'type': self.type,
            'data': {
                'index': self.index,
                'element': element_data
            },
            'transient': True  # Don't save to message history
        }


@dataclass
class ObjectPartialEvent(DataEvent):
    """Emitted when object fields are updated during streaming.

    Used when output_type is a single Pydantic model - streams partial updates.
    Note: Partial objects are NOT validated (may be incomplete).

    Example:
        output_type=WeatherResponse will emit this as each field is parsed.
    """
    type: str = 'data-object-partial'
    partial: Dict[str, Any] = None  # Partial dict with fields parsed so far

    def __post_init__(self):
        if self.partial is None:
            self.partial = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': {
                'partial': self.partial
            },
            'transient': True
        }


@dataclass
class ObjectCompleteEvent(DataEvent):
    """Emitted when the full structured output is complete and validated.

    Sent at the end of streaming with the final validated object/array.

    Example:
        For output_type=List[AIAgent], contains the full validated list.
        For output_type=WeatherResponse, contains the validated object.
    """
    type: str = 'data-object-complete'
    object: Any = None  # The validated Pydantic instance (serialized to dict)
    mode: str = 'object'  # 'object' or 'array'

    def to_dict(self) -> Dict[str, Any]:
        # Serialize Pydantic model to dict if needed
        object_data = self.object
        if hasattr(self.object, 'model_dump'):
            object_data = self.object.model_dump()
        elif hasattr(self.object, 'dict'):
            object_data = self.object.dict()
        elif isinstance(self.object, list):
            # List of Pydantic models
            object_data = []
            for item in self.object:
                if hasattr(item, 'model_dump'):
                    object_data.append(item.model_dump())
                elif hasattr(item, 'dict'):
                    object_data.append(item.dict())
                else:
                    object_data.append(item)

        return {
            'type': self.type,
            'data': {
                'object': object_data,
                'mode': self.mode
            },
            'transient': False  # Save to message history
        }


# ============================================================================
# Extended Thinking Events
# ============================================================================
# Custom events for Extended Thinking (multi-pass reasoning)
# Pattern: Analyze -> Critique -> Refine -> Conclude

@dataclass
class ThinkingStageEvent(DataEvent):
    """
    Transient event for UI progress during thinking phases.

    Emitted at the start of each thinking phase to let the UI
    show progress (e.g., "Analyzing...", "Refining (60% confident)...").

    Example:
        {
            "type": "data-thinking-stage",
            "data": {"stage": "refining", "step": 3, "iteration": 1, "confidence": 0.6},
            "transient": true
        }
    """
    type: str = 'data-thinking-stage'
    stage: str = ''  # 'analyzing', 'critiquing', 'refining', 'concluding'
    step: int = 0
    iteration: Optional[int] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {'stage': self.stage, 'step': self.step}
        if self.iteration is not None:
            data['iteration'] = self.iteration
        if self.confidence is not None:
            data['confidence'] = self.confidence
        return {
            'type': self.type,
            'data': data,
            'transient': True
        }


@dataclass
class ThinkingCompleteEvent(DataEvent):
    """
    Persistent event with thinking metadata.

    Emitted at the end of thinking with summary information.
    Saved to message history for analytics and debugging.

    Example:
        {
            "type": "data-thinking-complete",
            "data": {
                "steps": 5,
                "iterations": 2,
                "final_confidence": 0.9,
                "thinking_tokens": 2450,
                "thinking_model": "gpt-4o-mini"
            },
            "transient": false
        }
    """
    type: str = 'data-thinking-complete'
    steps: int = 0
    iterations: int = 0
    final_confidence: float = 0.0
    thinking_tokens: Optional[int] = None
    thinking_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'steps': self.steps,
            'iterations': self.iterations,
            'final_confidence': self.final_confidence
        }
        if self.thinking_tokens is not None:
            data['thinking_tokens'] = self.thinking_tokens
        if self.thinking_model is not None:
            data['thinking_model'] = self.thinking_model
        return {
            'type': self.type,
            'data': data,
            'transient': False  # Save to message history
        }
