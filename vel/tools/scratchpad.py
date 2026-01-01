"""
Scratchpad: Ephemeral in-memory working memory for Vel agents.

The Scratchpad enables agents to maintain structured context during multi-step
tool execution within a single run. Features:

- Semantic entry types (plan, finding, observation, reasoning, error)
- Thread-safe operations for concurrent tool calls
- Automatic summary generation for context continuity between runs
- Checkpointing for state snapshots
- Zero external dependencies (pure Python stdlib)

Example:
    ```python
    from vel import Agent
    from vel.tools.scratchpad import ScratchpadConfig

    # Simple usage
    agent = Agent(
        model={'provider': 'openai', 'model': 'gpt-4o'},
        scratchpad=True,  # Equivalent to ScratchpadConfig()
    )

    # With configuration
    agent = Agent(
        model={'provider': 'openai', 'model': 'gpt-4o'},
        scratchpad=ScratchpadConfig(
            max_entries=50,
            summary_max_chars=800,
        ),
    )

    # Multi-run with context continuity
    result1 = await agent.run({'message': 'Research quantum computing'})
    result2 = await agent.run({'message': 'Compare the top 3 companies'})
    # ^ Previous summary automatically injected

    # Clear context for new topic
    agent.clear_scratchpad_context()
    ```
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolSpec


# =============================================================================
# Entry Types
# =============================================================================

class EntryType(Enum):
    """Semantic types for scratchpad entries."""
    NOTE = "note"
    PLAN = "plan"
    FINDING = "finding"
    OBSERVATION = "observation"
    CHECKPOINT = "checkpoint"
    REASONING = "reasoning"
    ERROR = "error"
    CONTEXT = "context"


# Icons for formatted output
ICONS = {
    EntryType.PLAN: "📍",
    EntryType.FINDING: "📊",
    EntryType.OBSERVATION: "👁",
    EntryType.REASONING: "💭",
    EntryType.NOTE: "📝",
    EntryType.ERROR: "❌",
    EntryType.CHECKPOINT: "💾",
    EntryType.CONTEXT: "🔍",
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScratchpadConfig:
    """
    Configuration for scratchpad feature.

    Example:
        ```python
        from vel import Agent
        from vel.tools.scratchpad import ScratchpadConfig

        agent = Agent(
            id='researcher',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            scratchpad=ScratchpadConfig(
                max_entries=50,
                summary_max_chars=800,
            )
        )
        ```
    """

    # Entry limits
    max_entries: int = 100
    """Maximum number of entries before eviction (10-500)."""

    max_content_length: int = 50000
    """Maximum content length per entry in characters (1000-100000)."""

    # Summary generation
    summary_max_chars: int = 500
    """Maximum characters for generated summaries (100-2000)."""

    # Optional features
    include_search: bool = True
    """Include search_scratchpad tool."""

    include_checkpoint: bool = True
    """Include checkpoint tools."""

    def __post_init__(self):
        """Validate configuration bounds."""
        if self.max_entries < 10:
            self.max_entries = 10
        elif self.max_entries > 500:
            self.max_entries = 500

        if self.max_content_length < 1000:
            self.max_content_length = 1000
        elif self.max_content_length > 100000:
            self.max_content_length = 100000

        if self.summary_max_chars < 100:
            self.summary_max_chars = 100
        elif self.summary_max_chars > 2000:
            self.summary_max_chars = 2000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ScratchpadEntry:
    """A single entry in the scratchpad."""

    key: str
    """Unique identifier for this entry."""

    content: str
    """Entry content."""

    entry_type: EntryType
    """Semantic type of entry."""

    created_at: datetime
    """UTC timestamp when entry was created."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Flexible metadata (source, confidence, step number, etc.)."""

    content_hash: str = ""
    """SHA-256 hash prefix for deduplication."""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash(self.content)

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA-256 hash prefix of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'key': self.key,
            'content': self.content,
            'entry_type': self.entry_type.value,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'content_hash': self.content_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScratchpadEntry':
        """Deserialize from dictionary."""
        return cls(
            key=data['key'],
            content=data['content'],
            entry_type=EntryType(data['entry_type']),
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {}),
            content_hash=data.get('content_hash', '')
        )


@dataclass
class ScratchpadStats:
    """Statistics about scratchpad contents."""

    total_entries: int
    entries_by_type: Dict[str, int]
    total_content_length: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    checkpoint_count: int


# =============================================================================
# Scratchpad Class
# =============================================================================

class Scratchpad:
    """
    Ephemeral in-memory working memory for agent runs.

    Thread-safe, JSON-serializable, zero external dependencies.

    Example:
        ```python
        scratchpad = Scratchpad()

        # Store execution plan
        scratchpad.set_plan("1. Search docs\\n2. Extract patterns\\n3. Summarize")

        # Record findings
        scratchpad.add_finding("OAuth 2.0 required", source="api_docs")
        scratchpad.add_finding("Rate limit is 5000/hour", source="github")

        # Read all content
        print(scratchpad.read())

        # Generate summary for next run
        summary = scratchpad.get_summary()
        ```
    """

    def __init__(self, config: Optional[ScratchpadConfig] = None):
        self._config = config or ScratchpadConfig()
        self._entries: Dict[str, ScratchpadEntry] = {}
        self._lock = threading.RLock()
        self._finding_counter = 0
        self._reasoning_counter = 0
        self._checkpoints: List[Dict[str, Any]] = []
        self._logger = logging.getLogger('vel.scratchpad')

    # =========================================================================
    # Core Operations
    # =========================================================================

    def write(
        self,
        key: str,
        content: str,
        entry_type: EntryType = EntryType.NOTE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Write entry to scratchpad.

        Args:
            key: Unique identifier for this entry
            content: Entry content
            entry_type: Semantic type (note, plan, finding, etc.)
            metadata: Optional metadata dict

        Returns:
            Confirmation message
        """
        with self._lock:
            # Truncate content if exceeds limit
            if len(content) > self._config.max_content_length:
                content = content[:self._config.max_content_length]
                self._logger.warning(f"Content truncated for key '{key}'")

            entry = ScratchpadEntry(
                key=key,
                content=content,
                entry_type=entry_type,
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            self._entries[key] = entry
            self._evict_if_needed()

            self._logger.debug(f"WRITE: {key} ({entry_type.value})")
            return f"Saved to scratchpad: {key}"

    def read(self, key: Optional[str] = None) -> str:
        """
        Read entry or all entries.

        Args:
            key: Optional specific entry key. If None, returns all entries.

        Returns:
            Entry content or formatted view of all entries
        """
        with self._lock:
            if key:
                entry = self._entries.get(key)
                if not entry:
                    return f"Entry '{key}' not found in scratchpad."
                return f"{ICONS.get(entry.entry_type, '📄')} {key}: {entry.content}"

            return self._format_all_entries()

    def clear(self, preserve_persistent: bool = True) -> str:
        """
        Clear entries from scratchpad.

        Args:
            preserve_persistent: If True, keep plans, errors, and checkpoints

        Returns:
            Count of cleared entries
        """
        with self._lock:
            if preserve_persistent:
                # Keep plans, errors, and checkpoints
                protected = {EntryType.PLAN, EntryType.ERROR, EntryType.CHECKPOINT}
                to_remove = [k for k, v in self._entries.items()
                            if v.entry_type not in protected]
            else:
                to_remove = list(self._entries.keys())

            count = len(to_remove)
            for k in to_remove:
                del self._entries[k]

            self._logger.debug(f"CLEARED: {count} entries")
            return f"Cleared {count} entries from scratchpad."

    # =========================================================================
    # Specialized Write Operations
    # =========================================================================

    def set_plan(self, plan: str) -> str:
        """
        Store execution plan (overwrites existing).

        Args:
            plan: The execution plan text

        Returns:
            Confirmation message
        """
        return self.write('_plan', plan, EntryType.PLAN, {'persistent': True})

    def add_finding(
        self,
        finding: str,
        source: Optional[str] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Add auto-numbered finding.

        Args:
            finding: The finding text
            source: Optional source attribution
            confidence: Confidence score (0-1)

        Returns:
            Confirmation with finding number
        """
        with self._lock:
            self._finding_counter += 1
            key = f"finding_{self._finding_counter}"
            metadata: Dict[str, Any] = {
                'number': self._finding_counter,
                'confidence': confidence
            }
            if source:
                metadata['source'] = source
            result = self.write(key, finding, EntryType.FINDING, metadata)
            return f"Finding #{self._finding_counter} recorded."

    def add_observation(
        self,
        observation: str,
        tool_name: Optional[str] = None
    ) -> str:
        """
        Record tool observation.

        Args:
            observation: The observation text
            tool_name: Optional originating tool name

        Returns:
            Confirmation message
        """
        key = f"obs_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
        metadata: Dict[str, Any] = {}
        if tool_name:
            metadata['tool_name'] = tool_name
        return self.write(key, observation, EntryType.OBSERVATION, metadata)

    def add_reasoning(
        self,
        thought: str,
        step: Optional[int] = None
    ) -> str:
        """
        Add reasoning step.

        Args:
            thought: The reasoning/thought text
            step: Optional step number (auto-incremented if not provided)

        Returns:
            Confirmation message
        """
        with self._lock:
            if step is None:
                self._reasoning_counter += 1
                step = self._reasoning_counter
            key = f"reasoning_{step}"
            return self.write(key, thought, EntryType.REASONING, {'step': step})

    def log_error(
        self,
        error: str,
        context: Optional[str] = None,
        recoverable: bool = True
    ) -> str:
        """
        Log error with context.

        Args:
            error: Error message
            context: Optional context about when/where error occurred
            recoverable: Whether the error is recoverable

        Returns:
            Confirmation message
        """
        key = f"error_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
        metadata: Dict[str, Any] = {'recoverable': recoverable, 'persistent': True}
        if context:
            metadata['context'] = context
        return self.write(key, error, EntryType.ERROR, metadata)

    # =========================================================================
    # Search and Retrieval
    # =========================================================================

    def search(
        self,
        query: str,
        entry_types: Optional[List[EntryType]] = None
    ) -> List[ScratchpadEntry]:
        """
        Case-insensitive search across entries.

        Args:
            query: Search query
            entry_types: Optional filter by entry types

        Returns:
            List of matching entries (sorted by recency)
        """
        with self._lock:
            query_lower = query.lower()
            results = []
            for entry in self._entries.values():
                if entry_types and entry.entry_type not in entry_types:
                    continue
                if query_lower in entry.content.lower() or query_lower in entry.key.lower():
                    results.append(entry)
            return sorted(results, key=lambda e: e.created_at, reverse=True)

    def get_summary(self, max_chars: Optional[int] = None) -> str:
        """
        Generate condensed summary for injection into next run.

        Priority order: errors -> plan -> recent findings -> status

        Args:
            max_chars: Maximum characters (defaults to config value)

        Returns:
            Summary string, or empty string if scratchpad is empty/trivial
        """
        with self._lock:
            if not self._entries:
                return ""

            max_chars = max_chars or self._config.summary_max_chars
            parts = []

            # 1. Errors (highest priority)
            errors = [e for e in self._entries.values() if e.entry_type == EntryType.ERROR]
            if errors:
                parts.append(f"Errors ({len(errors)}):")
                for err in errors[-2:]:
                    err_text = err.content[:80]
                    if len(err.content) > 80:
                        err_text += "..."
                    parts.append(f"  - {err_text}")

            # 2. Plan
            plan_entry = self._entries.get('_plan')
            if plan_entry:
                plan_text = plan_entry.content[:100]
                if len(plan_entry.content) > 100:
                    plan_text += "..."
                parts.append(f"Plan: {plan_text}")

            # 3. Findings
            findings = sorted(
                [e for e in self._entries.values() if e.entry_type == EntryType.FINDING],
                key=lambda e: e.metadata.get('number', 0)
            )
            if findings:
                parts.append(f"Findings ({len(findings)}):")
                for f in findings[-5:]:
                    source = f.metadata.get('source', '')
                    src_text = f" [from: {source}]" if source else ""
                    finding_text = f.content[:60]
                    if len(f.content) > 60:
                        finding_text += "..."
                    parts.append(f"  - {finding_text}{src_text}")

            # 4. Status
            parts.append(f"Status: {len(self._entries)} entries total")

            summary = "\n".join(parts)
            if len(summary) > max_chars:
                summary = summary[:max_chars - 3] + "..."
            return summary

    def get_stats(self) -> ScratchpadStats:
        """Get statistics about scratchpad contents."""
        with self._lock:
            entries_by_type: Dict[str, int] = {}
            total_length = 0
            oldest: Optional[datetime] = None
            newest: Optional[datetime] = None

            for entry in self._entries.values():
                type_name = entry.entry_type.value
                entries_by_type[type_name] = entries_by_type.get(type_name, 0) + 1
                total_length += len(entry.content)

                if oldest is None or entry.created_at < oldest:
                    oldest = entry.created_at
                if newest is None or entry.created_at > newest:
                    newest = entry.created_at

            return ScratchpadStats(
                total_entries=len(self._entries),
                entries_by_type=entries_by_type,
                total_content_length=total_length,
                oldest_entry=oldest,
                newest_entry=newest,
                checkpoint_count=len(self._checkpoints)
            )

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def checkpoint(self, label: str) -> str:
        """
        Create named checkpoint of current state.

        Args:
            label: Checkpoint label

        Returns:
            Confirmation message
        """
        with self._lock:
            checkpoint_data = {
                'label': label,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'state': self.to_dict()
            }
            self._checkpoints.append(checkpoint_data)

            # Also store as an entry
            self.write(
                f"_checkpoint_{label}",
                f"Checkpoint: {label}",
                EntryType.CHECKPOINT,
                {'label': label, 'entry_count': len(self._entries)}
            )

            return f"Checkpoint '{label}' created ({len(self._entries)} entries)"

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints with labels and timestamps.

        Returns:
            List of checkpoint info dicts (sorted by recency)
        """
        with self._lock:
            return [
                {'label': cp['label'], 'created_at': cp['created_at']}
                for cp in sorted(self._checkpoints, key=lambda x: x['created_at'], reverse=True)
            ]

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _evict_if_needed(self) -> int:
        """Evict oldest non-essential entries if over limit."""
        protected = {EntryType.PLAN, EntryType.ERROR, EntryType.CHECKPOINT}
        evicted = 0

        while len(self._entries) > self._config.max_entries:
            # Find oldest evictable entry
            candidates = [
                (k, v) for k, v in self._entries.items()
                if v.entry_type not in protected
            ]
            if not candidates:
                break

            oldest = min(candidates, key=lambda x: x[1].created_at)
            del self._entries[oldest[0]]
            evicted += 1

        if evicted:
            self._logger.debug(f"Evicted {evicted} entries")
        return evicted

    def _format_all_entries(self) -> str:
        """Format all entries for display."""
        if not self._entries:
            return "📋 SCRATCHPAD CONTENTS\n━━━━━━━━━━━━━━━━━━━━━━\n(empty)"

        lines = ["📋 SCRATCHPAD CONTENTS", "━━━━━━━━━━━━━━━━━━━━━━", ""]

        # Group by type
        by_type: Dict[EntryType, List[ScratchpadEntry]] = {}
        for entry in self._entries.values():
            by_type.setdefault(entry.entry_type, []).append(entry)

        # Plan first
        if EntryType.PLAN in by_type:
            lines.append(f"{ICONS[EntryType.PLAN]} CURRENT PLAN:")
            for e in by_type[EntryType.PLAN]:
                lines.append(f"  {e.content}")
            lines.append("")

        # Findings
        if EntryType.FINDING in by_type:
            findings = sorted(by_type[EntryType.FINDING],
                            key=lambda e: e.metadata.get('number', 0))
            lines.append(f"{ICONS[EntryType.FINDING]} FINDINGS ({len(findings)}):")
            for f in findings:
                num = f.metadata.get('number', '?')
                source = f.metadata.get('source', '')
                src_text = f" [from: {source}]" if source else ""
                lines.append(f"  #{num}{src_text} {f.content}")
            lines.append("")

        # Reasoning
        if EntryType.REASONING in by_type:
            reasoning = sorted(by_type[EntryType.REASONING],
                              key=lambda e: e.metadata.get('step', 0))
            lines.append(f"{ICONS[EntryType.REASONING]} REASONING:")
            for r in reasoning:
                step = r.metadata.get('step', '?')
                lines.append(f"  [step {step}] {r.content}")
            lines.append("")

        # Observations
        if EntryType.OBSERVATION in by_type:
            lines.append(f"{ICONS[EntryType.OBSERVATION]} OBSERVATIONS:")
            for o in by_type[EntryType.OBSERVATION]:
                tool = o.metadata.get('tool_name', '')
                tool_text = f" ({tool})" if tool else ""
                lines.append(f"  • {o.content}{tool_text}")
            lines.append("")

        # Errors
        if EntryType.ERROR in by_type:
            lines.append(f"{ICONS[EntryType.ERROR]} ERRORS:")
            for err in by_type[EntryType.ERROR]:
                lines.append(f"  • {err.content}")
            lines.append("")

        # Notes
        if EntryType.NOTE in by_type:
            lines.append(f"{ICONS[EntryType.NOTE]} NOTES:")
            for n in by_type[EntryType.NOTE]:
                lines.append(f"  • {n.key}: {n.content}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (for checkpointing)."""
        with self._lock:
            return {
                'entries': {k: v.to_dict() for k, v in self._entries.items()},
                'finding_counter': self._finding_counter,
                'reasoning_counter': self._reasoning_counter,
                'config': self._config.to_dict()
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scratchpad':
        """Deserialize from dict."""
        config = ScratchpadConfig(**data.get('config', {}))
        scratchpad = cls(config)
        scratchpad._finding_counter = data.get('finding_counter', 0)
        scratchpad._reasoning_counter = data.get('reasoning_counter', 0)
        scratchpad._entries = {
            k: ScratchpadEntry.from_dict(v)
            for k, v in data.get('entries', {}).items()
        }
        return scratchpad


# =============================================================================
# Tool Binding
# =============================================================================

def get_scratchpad_tools(scratchpad: Scratchpad) -> List['ToolSpec']:
    """
    Return tools bound to a specific scratchpad instance.

    Called internally by Agent when scratchpad is enabled.

    Args:
        scratchpad: The Scratchpad instance to bind tools to

    Returns:
        List of ToolSpec instances for the scratchpad tools
    """
    from .registry import ToolSpec

    tools = []

    # Tool 1: write_to_scratchpad
    def write_to_scratchpad(key: str, content: str) -> str:
        """Save information to your working memory for later reference."""
        return scratchpad.write(key, content, EntryType.NOTE)

    tools.append(ToolSpec.from_function(
        write_to_scratchpad,
        name='write_to_scratchpad',
        description='Save information to your working memory for later reference'
    ))

    # Tool 2: read_from_scratchpad
    def read_from_scratchpad(key: Optional[str] = None) -> str:
        """Read from your working memory. Call with no args to see everything."""
        return scratchpad.read(key)

    tools.append(ToolSpec.from_function(
        read_from_scratchpad,
        name='read_from_scratchpad',
        description='Read from your working memory. Call with no args to see everything.'
    ))

    # Tool 3: save_plan
    def save_plan(plan: str) -> str:
        """Save your execution plan. Do this at the start of complex tasks."""
        return scratchpad.set_plan(plan)

    tools.append(ToolSpec.from_function(
        save_plan,
        name='save_plan',
        description='Save your execution plan. Do this at the start of complex tasks.'
    ))

    # Tool 4: record_finding
    def record_finding(finding: str, source: Optional[str] = None) -> str:
        """Record a research finding or discovery."""
        return scratchpad.add_finding(finding, source=source)

    tools.append(ToolSpec.from_function(
        record_finding,
        name='record_finding',
        description='Record a research finding or discovery'
    ))

    # Tool 5: record_observation
    def record_observation(observation: str, tool_name: Optional[str] = None) -> str:
        """Record an observation from a tool call."""
        return scratchpad.add_observation(observation, tool_name=tool_name)

    tools.append(ToolSpec.from_function(
        record_observation,
        name='record_observation',
        description='Record an observation from a tool call'
    ))

    # Optional Tool 6: search_scratchpad
    if scratchpad._config.include_search:
        def search_scratchpad(query: str) -> str:
            """Search your working memory for relevant entries."""
            results = scratchpad.search(query)
            if not results:
                return f"No entries found matching '{query}'"
            lines = [f"Found {len(results)} matching entries:"]
            for r in results[:10]:
                lines.append(f"  - [{r.entry_type.value}] {r.key}: {r.content[:50]}...")
            return "\n".join(lines)

        tools.append(ToolSpec.from_function(
            search_scratchpad,
            name='search_scratchpad',
            description='Search your working memory for relevant entries'
        ))

    # Optional Tool 7-8: checkpoint tools
    if scratchpad._config.include_checkpoint:
        def checkpoint_scratchpad(label: str) -> str:
            """Create a checkpoint of current scratchpad state."""
            return scratchpad.checkpoint(label)

        def list_scratchpad_checkpoints() -> str:
            """List all checkpoints."""
            checkpoints = scratchpad.list_checkpoints()
            if not checkpoints:
                return "No checkpoints saved."
            lines = ["Checkpoints:"]
            for cp in checkpoints:
                lines.append(f"  - {cp['label']} ({cp['created_at']})")
            return "\n".join(lines)

        tools.append(ToolSpec.from_function(
            checkpoint_scratchpad,
            name='checkpoint_scratchpad',
            description='Create a checkpoint of current scratchpad state'
        ))
        tools.append(ToolSpec.from_function(
            list_scratchpad_checkpoints,
            name='list_scratchpad_checkpoints',
            description='List all checkpoints'
        ))

    return tools
