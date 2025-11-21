"""
File Output - Support for file/image responses from tools and agents.

Provides FileOutput class for returning binary data (images, PDFs, etc.)
from tools and agents.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import base64
import uuid


@dataclass
class FileOutput:
    """
    Represents a file output from a tool or agent.

    Use this class to return binary data like images, PDFs, CSVs, etc.
    """
    data: bytes
    mime: str
    filename: Optional[str] = None
    file_id: Optional[str] = None

    def __post_init__(self):
        if self.file_id is None:
            self.file_id = str(uuid.uuid4())

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        mime: str,
        filename: Optional[str] = None
    ) -> 'FileOutput':
        """
        Create FileOutput from raw bytes.

        Args:
            data: Raw bytes of the file
            mime: MIME type (e.g., "image/png", "application/pdf")
            filename: Optional filename

        Returns:
            FileOutput instance
        """
        return cls(data=data, mime=mime, filename=filename)

    @classmethod
    def from_base64(
        cls,
        b64_data: str,
        mime: str,
        filename: Optional[str] = None
    ) -> 'FileOutput':
        """
        Create FileOutput from base64-encoded string.

        Args:
            b64_data: Base64-encoded data
            mime: MIME type
            filename: Optional filename

        Returns:
            FileOutput instance
        """
        data = base64.b64decode(b64_data)
        return cls(data=data, mime=mime, filename=filename)

    def to_base64(self) -> str:
        """Convert data to base64 string."""
        return base64.b64encode(self.data).decode('utf-8')

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'file_id': self.file_id,
            'mime': self.mime,
            'filename': self.filename,
            'data': self.to_base64(),
            'size': len(self.data)
        }

    def to_stream_event(self) -> dict:
        """
        Convert to stream protocol event.

        Returns:
            Event dict for file-response
        """
        return {
            'type': 'file-response',
            'file_id': self.file_id,
            'mime': self.mime,
            'filename': self.filename,
            'data': self.to_base64()
        }
