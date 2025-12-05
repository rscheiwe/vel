"""
Structured Output - Pydantic-validated agent responses.

Provides automatic JSON mode forcing, validation, and retry logic
for agents that need to return structured data.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type
import json


@dataclass
class StructuredOutputPolicy:
    """
    Policy for handling structured output validation.

    Attributes:
        max_retries: Number of times to retry on validation failure (default: 1)
        on_failure: What to do when all retries exhausted:
            - "raise": Raise StructuredOutputValidationError
            - "return_raw": Return the raw string output
            - "return_last_valid": Return last valid output if any, else raise
    """
    max_retries: int = 1
    on_failure: Literal["raise", "return_raw", "return_last_valid"] = "raise"


# =============================================================================
# Helper functions (must be defined before classes that use them)
# =============================================================================

def _get_type_name(output_type: Type) -> str:
    """Get a human-readable name for a type."""
    origin = getattr(output_type, '__origin__', None)
    if origin is list:
        element_type = getattr(output_type, '__args__', (None,))[0]
        if element_type and hasattr(element_type, '__name__'):
            return f"List[{element_type.__name__}]"
        return "List"

    if hasattr(output_type, '__name__'):
        return output_type.__name__
    return str(output_type)


def _get_model_schema(model_type: Type) -> dict:
    """Get JSON schema from a Pydantic model."""
    if hasattr(model_type, 'model_json_schema'):
        # Pydantic v2
        return model_type.model_json_schema()
    elif hasattr(model_type, 'schema'):
        # Pydantic v1
        return model_type.schema()
    else:
        return {}


def _get_schema_for_type(output_type: Type) -> dict:
    """Get JSON schema for a type, handling List[X] types."""
    # Check if it's a List type
    origin = getattr(output_type, '__origin__', None)
    if origin is list:
        element_type = getattr(output_type, '__args__', (None,))[0]
        if element_type:
            element_schema = _get_model_schema(element_type)
            return {
                "type": "array",
                "items": element_schema
            }
        return {"type": "array"}

    return _get_model_schema(output_type)


def _validate_single(data: Any, model_type: Type) -> Any:
    """Validate a single item against a Pydantic model."""
    # Support both Pydantic v1 and v2
    if hasattr(model_type, 'model_validate'):
        # Pydantic v2
        return model_type.model_validate(data)
    elif hasattr(model_type, 'parse_obj'):
        # Pydantic v1
        return model_type.parse_obj(data)
    elif isinstance(data, dict):
        # Assume it's a dataclass or similar
        return model_type(**data)
    else:
        return data


# =============================================================================
# Exception class
# =============================================================================

class StructuredOutputValidationError(Exception):
    """
    Raised when structured output validation fails after all retries.
    """
    def __init__(self, validation_error: Exception, raw_output: str, output_type: Type):
        self.validation_error = validation_error
        self.raw_output = raw_output
        self.output_type = output_type
        type_name = _get_type_name(output_type)
        super().__init__(
            f"Structured output validation failed: {validation_error}"
        )


# =============================================================================
# Main functions
# =============================================================================

def parse_structured_output(raw_output: str, output_type: Type) -> Any:
    """
    Parse and validate raw LLM output against a Pydantic model.

    Args:
        raw_output: Raw string output from LLM (should be JSON)
        output_type: Pydantic model class or List[Model] to validate against

    Returns:
        Validated Pydantic model instance or list of instances

    Raises:
        Exception: If parsing or validation fails
    """
    # Try to extract JSON from the output
    # Handle cases where LLM wraps JSON in markdown code blocks
    json_str = raw_output.strip()

    # Remove markdown code blocks if present
    if json_str.startswith('```json'):
        json_str = json_str[7:]
    elif json_str.startswith('```'):
        json_str = json_str[3:]

    if json_str.endswith('```'):
        json_str = json_str[:-3]

    json_str = json_str.strip()

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Check if output_type is a List type (typing.List[X] or list[X])
    origin = getattr(output_type, '__origin__', None)
    if origin is list:
        # It's a List[X] type - validate each element
        element_type = getattr(output_type, '__args__', (None,))[0]
        if element_type and isinstance(data, list):
            validated_list = []
            for item in data:
                validated_list.append(_validate_single(item, element_type))
            return validated_list
        return data

    # Single model validation
    return _validate_single(data, output_type)


def get_retry_prompt(output_type: Type, error: Exception) -> str:
    """
    Generate a system prompt for retrying after validation failure.

    Args:
        output_type: The Pydantic model or List[Model] that failed validation
        error: The validation error

    Returns:
        System prompt instructing the model to fix its output
    """
    schema = _get_schema_for_type(output_type)
    schema_str = json.dumps(schema, indent=2)
    type_name = _get_type_name(output_type)

    return (
        f"Your previous output did not match the required schema {type_name}. "
        f"Error: {error}\n\n"
        f"Please respond again with valid JSON matching this schema:\n{schema_str}"
    )


def get_json_mode_system_prompt(output_type: Type) -> str:
    """
    Generate a system prompt that instructs the model to output JSON.

    Args:
        output_type: The Pydantic model or List[Model] to output

    Returns:
        System prompt with schema
    """
    schema = _get_schema_for_type(output_type)
    schema_str = json.dumps(schema, indent=2)

    return (
        f"You must respond with valid JSON that matches this schema:\n"
        f"{schema_str}\n\n"
        f"Do not include any text before or after the JSON. "
        f"Do not wrap the JSON in markdown code blocks."
    )
