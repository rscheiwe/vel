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


class StructuredOutputValidationError(Exception):
    """
    Raised when structured output validation fails after all retries.
    """
    def __init__(self, validation_error: Exception, raw_output: str, output_type: Type):
        self.validation_error = validation_error
        self.raw_output = raw_output
        self.output_type = output_type
        super().__init__(
            f"Failed to validate output as {output_type.__name__}: {validation_error}"
        )


def parse_structured_output(raw_output: str, output_type: Type) -> Any:
    """
    Parse and validate raw LLM output against a Pydantic model.

    Args:
        raw_output: Raw string output from LLM (should be JSON)
        output_type: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

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

    # Validate with Pydantic
    # Support both Pydantic v1 and v2
    if hasattr(output_type, 'model_validate'):
        # Pydantic v2
        return output_type.model_validate(data)
    elif hasattr(output_type, 'parse_obj'):
        # Pydantic v1
        return output_type.parse_obj(data)
    else:
        # Assume it's a dataclass or similar
        return output_type(**data)


def get_retry_prompt(output_type: Type, error: Exception) -> str:
    """
    Generate a system prompt for retrying after validation failure.

    Args:
        output_type: The Pydantic model that failed validation
        error: The validation error

    Returns:
        System prompt instructing the model to fix its output
    """
    # Get schema from Pydantic model
    if hasattr(output_type, 'model_json_schema'):
        # Pydantic v2
        schema = output_type.model_json_schema()
    elif hasattr(output_type, 'schema'):
        # Pydantic v1
        schema = output_type.schema()
    else:
        schema = {}

    schema_str = json.dumps(schema, indent=2)

    return (
        f"Your previous output did not match the required schema {output_type.__name__}. "
        f"Error: {error}\n\n"
        f"Please respond again with valid JSON matching this schema:\n{schema_str}"
    )


def get_json_mode_system_prompt(output_type: Type) -> str:
    """
    Generate a system prompt that instructs the model to output JSON.

    Args:
        output_type: The Pydantic model to output

    Returns:
        System prompt with schema
    """
    # Get schema from Pydantic model
    if hasattr(output_type, 'model_json_schema'):
        # Pydantic v2
        schema = output_type.model_json_schema()
    elif hasattr(output_type, 'schema'):
        # Pydantic v1
        schema = output_type.schema()
    else:
        schema = {}

    schema_str = json.dumps(schema, indent=2)

    return (
        f"You must respond with valid JSON that matches this schema:\n"
        f"{schema_str}\n\n"
        f"Do not include any text before or after the JSON. "
        f"Do not wrap the JSON in markdown code blocks."
    )
