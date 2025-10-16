"""
Context Store for RLM

Loads and provides probing access to large context (documents, files, URLs).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import re
import os


class ContextChunk:
    """A chunk of context with metadata."""

    def __init__(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize context chunk.

        Args:
            id: Unique chunk identifier
            content: Chunk content
            metadata: Optional metadata (source, line numbers, etc.)
        """
        self.id = id
        self.content = content
        self.metadata = metadata or {}

    def preview(self, max_bytes: int) -> tuple[str, bool]:
        """
        Get preview of chunk with truncation.

        Args:
            max_bytes: Maximum bytes to return

        Returns:
            (preview_text, truncated)
        """
        if len(self.content) <= max_bytes:
            return self.content, False

        return self.content[:max_bytes] + "...", True


class ContextStore:
    """
    Store and probe large context for RLM.

    Supports loading context from:
    - Raw text (passed directly)
    - File paths (local files)
    - URLs (future: fetch content)

    Provides probing methods:
    - search: Keyword/regex search
    - read: Read specific chunk by ID
    - summarize: Summarize chunk or text
    """

    def __init__(self, chunk_size: int = 4096):
        """
        Initialize context store.

        Args:
            chunk_size: Default chunk size for splitting large content
        """
        self.chunk_size = chunk_size
        self.chunks: List[ContextChunk] = []
        self._full_text: str = ""

    def load(self, context_refs: Union[str, List[str], List[Dict[str, Any]]]) -> int:
        """
        Load context from references.

        Args:
            context_refs: Can be:
                - String: Raw text content
                - List[str]: File paths
                - List[Dict]: Structured refs with 'type' and 'source' keys

        Returns:
            Number of chunks created
        """
        # Handle string (raw text)
        if isinstance(context_refs, str):
            return self._load_text(context_refs)

        # Handle list
        if isinstance(context_refs, list):
            total_chunks = 0
            for ref in context_refs:
                if isinstance(ref, str):
                    # Assume file path
                    chunks = self._load_file(ref)
                    total_chunks += chunks
                elif isinstance(ref, dict):
                    # Structured reference
                    ref_type = ref.get('type', 'text')
                    source = ref.get('source', '')

                    if ref_type == 'text':
                        chunks = self._load_text(source)
                    elif ref_type == 'file':
                        chunks = self._load_file(source)
                    elif ref_type == 'url':
                        # Future: implement URL fetching
                        chunks = 0
                    else:
                        chunks = 0

                    total_chunks += chunks

            return total_chunks

        return 0

    def _load_text(self, text: str, source_name: str = "text") -> int:
        """
        Load raw text content.

        Args:
            text: Text content
            source_name: Name for source metadata

        Returns:
            Number of chunks created
        """
        if not text:
            return 0

        # Split into chunks
        chunks = self._chunk_text(text, source_name)
        self.chunks.extend(chunks)
        self._full_text += text + "\n\n"

        return len(chunks)

    def _load_file(self, file_path: str) -> int:
        """
        Load content from file.

        Args:
            file_path: Path to file

        Returns:
            Number of chunks created
        """
        if not os.path.exists(file_path):
            return 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return self._load_text(content, source_name=os.path.basename(file_path))
        except Exception:
            return 0

    def _chunk_text(self, text: str, source_name: str) -> List[ContextChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            source_name: Source name for metadata

        Returns:
            List of ContextChunk objects
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        chunk_idx = 0

        for line in lines:
            line_size = len(line.encode('utf-8'))

            if current_size + line_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                chunk_id = f"{source_name}:chunk_{chunk_idx}"
                chunks.append(ContextChunk(
                    id=chunk_id,
                    content=chunk_content,
                    metadata={'source': source_name, 'chunk_index': chunk_idx}
                ))

                # Reset
                current_chunk = []
                current_size = 0
                chunk_idx += 1

            current_chunk.append(line)
            current_size += line_size

        # Final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk_id = f"{source_name}:chunk_{chunk_idx}"
            chunks.append(ContextChunk(
                id=chunk_id,
                content=chunk_content,
                metadata={'source': source_name, 'chunk_index': chunk_idx}
            ))

        return chunks

    def search(self, query: str, max_results: int = 10, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for query in context.

        Supports keyword search (OR logic for multiple words) and regex patterns.

        Args:
            query: Search query (keywords or regex pattern)
            max_results: Maximum results to return
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of search results with chunk ID, snippet, and metadata
        """
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        # Check if query looks like a regex (has special chars)
        is_regex = any(c in query for c in r'[](){}.*+?^$|\\')

        if is_regex:
            # Try as regex pattern
            try:
                pattern = re.compile(query, flags)
                patterns = [pattern]
            except re.error:
                # Invalid regex, fall back to literal
                pattern = re.compile(re.escape(query), flags)
                patterns = [pattern]
        else:
            # Multi-keyword search: split on whitespace and search for any keyword
            keywords = query.split()
            patterns = [re.compile(re.escape(kw), flags) for kw in keywords if kw]

        # Search chunks
        chunk_scores = []
        for chunk in self.chunks:
            # Count total matches across all keywords
            total_matches = 0
            first_match_pos = None

            for pattern in patterns:
                matches = list(pattern.finditer(chunk.content))
                if matches:
                    total_matches += len(matches)
                    if first_match_pos is None:
                        first_match_pos = matches[0].start()

            if total_matches > 0:
                # Get snippet around first match
                start = max(0, first_match_pos - 100)
                end = min(len(chunk.content), first_match_pos + 200)
                snippet = chunk.content[start:end]

                chunk_scores.append({
                    'chunk_id': chunk.id,
                    'snippet': snippet,
                    'match_count': total_matches,
                    'metadata': chunk.metadata,
                    'score': total_matches  # Sort by number of matches
                })

        # Sort by score (most matches first)
        chunk_scores.sort(key=lambda x: x['score'], reverse=True)

        # Return top results
        results = [
            {
                'chunk_id': r['chunk_id'],
                'snippet': r['snippet'],
                'match_count': r['match_count'],
                'metadata': r['metadata']
            }
            for r in chunk_scores[:max_results]
        ]

        return results

    def read(self, chunk_id: str, max_bytes: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Read specific chunk by ID.

        Args:
            chunk_id: Chunk identifier (accepts both 'text:chunk_0' and 'chunk_0' formats)
            max_bytes: Optional max bytes to return (with truncation)

        Returns:
            Dict with preview, truncated flag, and metadata, or None if not found
        """
        # Normalize chunk_id - handle both full and short forms
        # If user passes 'chunk_0', try to match 'text:chunk_0' or any 'source:chunk_0'
        for chunk in self.chunks:
            # Direct match
            if chunk.id == chunk_id:
                if max_bytes:
                    preview, truncated = chunk.preview(max_bytes)
                else:
                    preview = chunk.content
                    truncated = False

                return {
                    'preview': preview,
                    'truncated': truncated,
                    'metadata': chunk.metadata,
                    'full_size': len(chunk.content)
                }

            # Try matching without source prefix (e.g., 'chunk_0' matches 'text:chunk_0')
            if ':' in chunk.id:
                _, chunk_suffix = chunk.id.split(':', 1)
                if chunk_suffix == chunk_id:
                    if max_bytes:
                        preview, truncated = chunk.preview(max_bytes)
                    else:
                        preview = chunk.content
                        truncated = False

                    return {
                        'preview': preview,
                        'truncated': truncated,
                        'metadata': chunk.metadata,
                        'full_size': len(chunk.content)
                    }

        return None

    def summarize(self, chunk_id_or_text: str, max_length: int = 500) -> str:
        """
        Summarize chunk or text.

        Simple extractive summary (first N chars + last N chars).
        In production, could use LLM for abstractive summarization.

        Args:
            chunk_id_or_text: Chunk ID or raw text
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        # Check if it's a chunk ID
        chunk_data = self.read(chunk_id_or_text)
        if chunk_data:
            text = chunk_data['preview']
        else:
            text = chunk_id_or_text

        if len(text) <= max_length:
            return text

        # Extract first and last parts
        part_size = max_length // 2 - 10
        summary = text[:part_size] + "\n...\n" + text[-part_size:]

        return summary

    def get_full_text(self) -> str:
        """
        Get full text content.

        Warning: May be very large. Use with caution.

        Returns:
            Full text content
        """
        return self._full_text

    def get_chunks_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded chunks.

        Returns:
            Summary with chunk count, total size, etc.
        """
        total_size = sum(len(chunk.content) for chunk in self.chunks)
        sources = set(chunk.metadata.get('source', 'unknown') for chunk in self.chunks)

        return {
            'num_chunks': len(self.chunks),
            'total_size': total_size,
            'sources': list(sources),
            'chunk_ids': [chunk.id for chunk in self.chunks]
        }

    def clear(self):
        """Clear all chunks."""
        self.chunks.clear()
        self._full_text = ""
