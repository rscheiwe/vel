"""
Stream Protocol Events
Based on Vercel AI SDK stream protocol: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""
from __future__ import annotations
from typing import Any, Dict, Literal, Optional
from dataclasses import dataclass

# Event types
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
    'tool-input-available',
    'tool-output-available',
    'start-step',
    'finish-step',
    'finish-message',
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
    """Tool call input fully available"""
    type: Literal['tool-input-available'] = 'tool-input-available'
    tool_call_id: str = ''
    tool_name: str = ''
    input: Dict[str, Any] = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'toolCallId': self.tool_call_id,
            'toolName': self.tool_name,
            'input': self.input
        }

@dataclass
class ToolOutputAvailableEvent(StreamEvent):
    """Tool call output available"""
    type: Literal['tool-output-available'] = 'tool-output-available'
    tool_call_id: str = ''
    output: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'toolCallId': self.tool_call_id, 'output': self.output}

@dataclass
class StartStepEvent(StreamEvent):
    """Start a reasoning step"""
    type: Literal['start-step'] = 'start-step'

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

@dataclass
class FinishStepEvent(StreamEvent):
    """Finish a reasoning step"""
    type: Literal['finish-step'] = 'finish-step'

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
    """Error event"""
    type: Literal['error'] = 'error'
    error: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), 'error': self.error}
