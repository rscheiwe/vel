"""
Scratchpad for RLM

Data structures for maintaining notes during recursive reasoning.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time
import hashlib


@dataclass
class Note:
    """
    A single atomic note in the scratchpad.

    Args:
        id: Unique identifier (auto-generated from content hash if not provided)
        text: Note content
        source_hint: Optional reference to source (e.g., "chunk 2.3", "probe:search")
        parent_id: Optional parent note ID (for hierarchical notes from recursive calls)
        timestamp: Unix timestamp (auto-generated if not provided)
    """
    text: str
    id: Optional[str] = None
    source_hint: Optional[str] = None
    parent_id: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        """Generate ID and timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()

        if self.id is None:
            # Generate ID from content hash + timestamp
            content = f"{self.text}:{self.timestamp}"
            self.id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'text': self.text,
            'source_hint': self.source_hint,
            'parent_id': self.parent_id,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> Note:
        """Create from dictionary."""
        return cls(
            text=data['text'],
            id=data.get('id'),
            source_hint=data.get('source_hint'),
            parent_id=data.get('parent_id'),
            timestamp=data.get('timestamp')
        )


class Scratchpad:
    """
    Scratchpad for accumulating notes during RLM execution.

    Maintains a list of atomic notes with deduplication and capping.
    """

    def __init__(self, max_notes: int = 200):
        """
        Initialize scratchpad.

        Args:
            max_notes: Maximum number of notes before capping/summarization
        """
        self.notes: List[Note] = []
        self.max_notes = max_notes
        self._note_ids = set()  # For fast deduplication

    def add(self, text: str, source_hint: Optional[str] = None, parent_id: Optional[str] = None) -> Note:
        """
        Add a new note to the scratchpad.

        Args:
            text: Note text
            source_hint: Optional source reference
            parent_id: Optional parent note ID

        Returns:
            The created Note
        """
        note = Note(text=text, source_hint=source_hint, parent_id=parent_id)

        # Check for duplicate
        if note.id in self._note_ids:
            return note  # Skip duplicate

        self.notes.append(note)
        self._note_ids.add(note.id)

        # Cap if needed
        if len(self.notes) > self.max_notes:
            self._cap()

        return note

    def merge(self, notes: List[Note], dedup: bool = True):
        """
        Merge multiple notes into scratchpad.

        Args:
            notes: List of notes to merge
            dedup: Whether to deduplicate (default: True)
        """
        for note in notes:
            if dedup and note.id in self._note_ids:
                continue
            self.notes.append(note)
            self._note_ids.add(note.id)

        # Cap if needed
        if len(self.notes) > self.max_notes:
            self._cap()

    def _cap(self):
        """
        Cap notes to max_notes by removing oldest.

        In a production system, this could summarize old notes instead of dropping.
        """
        if len(self.notes) <= self.max_notes:
            return

        # Keep most recent notes
        dropped = self.notes[:len(self.notes) - self.max_notes]
        self.notes = self.notes[len(self.notes) - self.max_notes:]

        # Update ID set
        for note in dropped:
            self._note_ids.discard(note.id)

    def to_bullets(self, limit: Optional[int] = None) -> str:
        """
        Format notes as bulleted list for LLM consumption.

        Args:
            limit: Optional limit on number of recent notes to include

        Returns:
            Formatted bullet list
        """
        if not self.notes:
            return "(no notes yet)"

        notes_to_show = self.notes[-limit:] if limit else self.notes
        bullets = []

        for note in notes_to_show:
            bullet = f"- {note.text}"
            if note.source_hint:
                bullet += f" [{note.source_hint}]"
            bullets.append(bullet)

        return "\n".join(bullets)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'notes': [note.to_dict() for note in self.notes],
            'max_notes': self.max_notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> Scratchpad:
        """Create from dictionary."""
        scratchpad = cls(max_notes=data.get('max_notes', 200))
        notes = [Note.from_dict(n) for n in data.get('notes', [])]
        scratchpad.merge(notes, dedup=False)
        return scratchpad

    def clear(self):
        """Clear all notes."""
        self.notes.clear()
        self._note_ids.clear()

    def __len__(self) -> int:
        """Return number of notes."""
        return len(self.notes)

    def __repr__(self) -> str:
        """String representation."""
        return f"Scratchpad(notes={len(self.notes)}, max={self.max_notes})"
