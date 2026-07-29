from .agent import Agent, run_stream
from .loader import load_agent
from .tools import ToolSpec, register_tool
from .utils import MessageReducer
from .providers import (
    ProviderRegistry,
    BaseProvider,
    OpenAIProvider,
    GeminiProvider,
    # Translators
    OpenAIAPITranslator,
    OpenAIResponsesAPITranslator,
    OpenAIAgentsSDKTranslator,
    AnthropicAPITranslator,
    GeminiAPITranslator,
    get_openai_api_translator,
    get_openai_responses_translator,
    get_openai_agents_translator,
    get_anthropic_translator,
    get_gemini_translator,
)
from .core import State, Effect, reduce, ContextManager, StatelessContextManager
from .events import (
    DataEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolInputAvailableEvent,
    ToolOutputAvailableEvent,
    ToolOutputErrorEvent,
    StepStartEvent,
    StepFinishEvent,
    ErrorEvent,
    FinishMessageEvent
)
from .prompts import (
    PromptTemplate,
    SystemPromptBuilder,
    PromptRegistry,
    register_prompt,
    get_prompt,
    has_prompt,
    list_prompts,
    PromptManager,
    PromptContextManager,
    XMLFormatter,
    MarkdownFormatter,
    ContextCompactor,
    MessageFormatter
)
from .rlm import (
    RlmConfig,
    RlmController,
    Scratchpad,
    Note,
    Budget,
    ContextStore
)

# Harness Mode (opt-in, default-off). Soft-guarded so a missing optional
# dependency can never break `import vel` (backwards-compat contract §8.5).
_HARNESS_EXPORTS: list = []
try:
    from .harness import (
        HarnessConfig,
        CompactionConfig,
        ApprovalConfig,
        SandboxConfig,
        HarnessBudgetConfig,
        Skill,
        SkillRegistry,
        SkillRef,
        ApprovalRequest,
        ApprovalDecision,
        RunCheckpoint,
        CheckpointStore,
        RunManager,
        SQLiteEventLogStore,
        PostgresEventLogStore,
        PubSub,
        InProcessPubSub,
        RedisPubSub,
    )
    _HARNESS_EXPORTS = [
        'HarnessConfig', 'CompactionConfig', 'ApprovalConfig', 'SandboxConfig',
        'HarnessBudgetConfig', 'Skill', 'SkillRegistry', 'SkillRef',
        'ApprovalRequest', 'ApprovalDecision', 'RunCheckpoint', 'CheckpointStore',
        'RunManager', 'SQLiteEventLogStore', 'PostgresEventLogStore',
        'PubSub', 'InProcessPubSub', 'RedisPubSub',
    ]
except ImportError:  # pragma: no cover - harness optional deps absent
    pass

__all__ = [
    'Agent',
    'run_stream',
    'load_agent',
    'ToolSpec',
    'register_tool',
    'MessageReducer',
    'ProviderRegistry',
    'BaseProvider',
    'OpenAIProvider',
    'GeminiProvider',
    'State',
    'Effect',
    'reduce',
    'ContextManager',
    'StatelessContextManager',
    # Event Translators
    'OpenAIAPITranslator',
    'OpenAIResponsesAPITranslator',
    'OpenAIAgentsSDKTranslator',
    'AnthropicAPITranslator',
    'GeminiAPITranslator',
    'get_openai_api_translator',
    'get_openai_responses_translator',
    'get_openai_agents_translator',
    'get_anthropic_translator',
    'get_gemini_translator',
    # Stream Events
    'DataEvent',
    'StreamEvent',
    'TextDeltaEvent',
    'ToolInputAvailableEvent',
    'ToolOutputAvailableEvent',
    'ToolOutputErrorEvent',
    'StepStartEvent',
    'StepFinishEvent',
    'ErrorEvent',
    'FinishMessageEvent',
    # Prompt module
    'PromptTemplate',
    'SystemPromptBuilder',
    'PromptRegistry',
    'register_prompt',
    'get_prompt',
    'has_prompt',
    'list_prompts',
    'PromptManager',
    'PromptContextManager',
    'XMLFormatter',
    'MarkdownFormatter',
    'ContextCompactor',
    'MessageFormatter',
    # RLM module
    'RlmConfig',
    'RlmController',
    'Scratchpad',
    'Note',
    'Budget',
    'ContextStore',
] + _HARNESS_EXPORTS
