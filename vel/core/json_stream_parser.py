"""
Incremental JSON Stream Parser for Structured Output Streaming.

Parses streaming JSON tokens and emits events when:
- Array elements are complete (for List[X] output types)
- Object fields are updated (for single object output types)

This enables progressive UI updates as structured data streams in.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Generator, List, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass


class OutputMode(Enum):
    """Output mode based on output_type schema."""
    TEXT = "text"      # No structured output
    OBJECT = "object"  # Single object (Pydantic model)
    ARRAY = "array"    # Array of objects (List[Pydantic model])


@dataclass
class StreamedElement:
    """A complete array element that has been parsed and validated."""
    index: int
    element: Any  # Validated Pydantic instance
    raw: str      # Raw JSON string for this element


@dataclass
class PartialObject:
    """A partial object with fields populated so far."""
    partial: Dict[str, Any]  # Partial dict with completed fields
    raw: str                  # Raw JSON accumulated so far


def detect_output_mode(output_type: Optional[Type]) -> OutputMode:
    """
    Detect the output mode from the output_type.

    Args:
        output_type: The Pydantic model or List[Model] type

    Returns:
        OutputMode.ARRAY if List[X], OutputMode.OBJECT if single model, TEXT if None
    """
    if output_type is None:
        return OutputMode.TEXT

    # Check if it's a List type (typing.List or list)
    origin = getattr(output_type, '__origin__', None)
    if origin is list or origin is List:
        return OutputMode.ARRAY

    # Check for typing.List in Python 3.9+ style
    if hasattr(output_type, '__class__') and output_type.__class__.__name__ == '_GenericAlias':
        if str(output_type).startswith('typing.List') or str(output_type).startswith('list['):
            return OutputMode.ARRAY

    # Single object type
    return OutputMode.OBJECT


def get_element_type(output_type: Type) -> Optional[Type]:
    """
    Extract the element type from a List[X] type.

    Args:
        output_type: The List[X] type

    Returns:
        The inner type X, or None if not a List type
    """
    args = getattr(output_type, '__args__', None)
    if args and len(args) > 0:
        return args[0]
    return None


class IncrementalJsonParser:
    """
    Parses streaming JSON and yields complete elements/partial objects.

    For arrays: Yields StreamedElement when each array item is complete
    For objects: Yields PartialObject as fields are completed
    """

    def __init__(self, output_type: Type, element_type: Optional[Type] = None):
        """
        Initialize the parser.

        Args:
            output_type: The full output type (e.g., List[Agent] or Agent)
            element_type: For arrays, the element type (e.g., Agent)
        """
        self.output_type = output_type
        self.element_type = element_type
        self.mode = detect_output_mode(output_type)

        # Buffer for accumulating JSON
        self.buffer = ""

        # State tracking
        self.array_depth = 0
        self.object_depth = 0
        self.in_string = False
        self.escape_next = False

        # For array mode: track elements
        self.current_element_start = -1
        self.elements_emitted = 0

        # For object mode: track last valid partial
        self.last_valid_partial: Optional[Dict[str, Any]] = None

    def feed(self, chunk: str) -> Generator[Union[StreamedElement, PartialObject], None, None]:
        """
        Feed a chunk of JSON text and yield any complete elements/updates.

        Args:
            chunk: New JSON text chunk

        Yields:
            StreamedElement for complete array items, PartialObject for object updates
        """
        for char in chunk:
            self.buffer += char

            # Handle string escaping
            if self.escape_next:
                self.escape_next = False
                continue

            if char == '\\' and self.in_string:
                self.escape_next = True
                continue

            if char == '"' and not self.escape_next:
                self.in_string = not self.in_string
                continue

            # Skip if inside a string
            if self.in_string:
                continue

            # Track depth
            if char == '[':
                if self.array_depth == 0 and self.mode == OutputMode.ARRAY:
                    # Starting the main array
                    self.current_element_start = len(self.buffer)
                self.array_depth += 1

            elif char == ']':
                self.array_depth -= 1
                if self.array_depth == 0 and self.mode == OutputMode.ARRAY:
                    # End of main array - try to parse final element if any
                    element = self._try_extract_array_element()
                    if element:
                        yield element

            elif char == '{':
                if self.mode == OutputMode.ARRAY and self.array_depth == 1 and self.object_depth == 0:
                    # Starting a new array element
                    self.current_element_start = len(self.buffer) - 1
                self.object_depth += 1

            elif char == '}':
                self.object_depth -= 1
                if self.mode == OutputMode.ARRAY and self.array_depth == 1 and self.object_depth == 0:
                    # Completed an array element
                    element = self._try_extract_array_element()
                    if element:
                        yield element
                elif self.mode == OutputMode.OBJECT:
                    # For single objects, try to yield partial updates
                    partial = self._try_parse_partial_object()
                    if partial:
                        yield partial

            elif char == ',' and self.mode == OutputMode.OBJECT and self.object_depth == 1:
                # Field separator in object mode - try to yield partial
                partial = self._try_parse_partial_object()
                if partial:
                    yield partial

    def _try_extract_array_element(self) -> Optional[StreamedElement]:
        """Try to extract and validate the current array element."""
        if self.current_element_start < 0:
            return None

        # Extract the element JSON
        element_json = self.buffer[self.current_element_start:].strip()

        # Remove trailing comma if present
        if element_json.endswith(','):
            element_json = element_json[:-1].strip()

        if not element_json or element_json in [']', '']:
            return None

        try:
            # Parse JSON
            data = json.loads(element_json)

            # Validate with Pydantic if element_type is available
            if self.element_type:
                if hasattr(self.element_type, 'model_validate'):
                    validated = self.element_type.model_validate(data)
                elif hasattr(self.element_type, 'parse_obj'):
                    validated = self.element_type.parse_obj(data)
                else:
                    validated = self.element_type(**data)
            else:
                validated = data

            result = StreamedElement(
                index=self.elements_emitted,
                element=validated,
                raw=element_json
            )
            self.elements_emitted += 1
            self.current_element_start = len(self.buffer)
            return result

        except (json.JSONDecodeError, Exception):
            # Not yet a valid element, continue accumulating
            return None

    def _try_parse_partial_object(self) -> Optional[PartialObject]:
        """Try to parse a partial object from the buffer."""
        # Try to make the buffer valid JSON by closing brackets
        test_json = self.buffer.strip()

        # Count unclosed brackets/braces
        opens = test_json.count('{') - test_json.count('}')

        # Try to close the JSON
        if opens > 0:
            test_json = test_json.rstrip(',') + ('}' * opens)

        try:
            data = json.loads(test_json)

            # Only yield if we have new fields
            if data != self.last_valid_partial:
                self.last_valid_partial = data.copy()
                return PartialObject(
                    partial=data,
                    raw=self.buffer
                )
        except json.JSONDecodeError:
            pass

        return None

    def finalize(self) -> Optional[Any]:
        """
        Finalize parsing and return the complete validated object.

        Returns:
            The fully validated output object, or None if parsing failed
        """
        try:
            # Clean up buffer
            json_str = self.buffer.strip()

            # Remove markdown code blocks if present
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            elif json_str.startswith('```'):
                json_str = json_str[3:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            # Parse
            data = json.loads(json_str)

            # Handle array mode - validate each element with element_type
            if self.mode == OutputMode.ARRAY and isinstance(data, list):
                if self.element_type:
                    validated_list = []
                    for item in data:
                        if hasattr(self.element_type, 'model_validate'):
                            validated_list.append(self.element_type.model_validate(item))
                        elif hasattr(self.element_type, 'parse_obj'):
                            validated_list.append(self.element_type.parse_obj(item))
                        else:
                            validated_list.append(self.element_type(**item) if isinstance(item, dict) else item)
                    return validated_list
                return data

            # Handle object mode - validate with output_type directly
            if hasattr(self.output_type, 'model_validate'):
                return self.output_type.model_validate(data)
            elif hasattr(self.output_type, 'parse_obj'):
                return self.output_type.parse_obj(data)
            else:
                return data

        except Exception:
            return None
