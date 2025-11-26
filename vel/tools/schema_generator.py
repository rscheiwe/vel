"""
Schema generation utilities for converting Python functions to JSON schemas.

This module provides helpers to automatically generate input/output JSON schemas
from Python function signatures and type hints.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints, get_origin, get_args


def generate_input_schema_from_function(fn: Callable) -> Dict[str, Any]:
    """
    Generate JSON schema from function signature using type hints.

    Args:
        fn: Function to analyze

    Returns:
        JSON schema dict with properties and required fields

    Example:
        def foo(city: str, count: int = 5) -> dict:
            pass

        Returns:
        {
            'type': 'object',
            'properties': {
                'city': {'type': 'string'},
                'count': {'type': 'integer', 'default': 5}
            },
            'required': ['city']
        }
    """
    sig = inspect.signature(fn)

    try:
        hints = get_type_hints(fn)
    except Exception:
        # If type hints fail (e.g., forward references), use empty hints
        hints = {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip context parameters
        if param_name in ['ctx', '_context', 'context']:
            continue

        # Get type hint or default to str
        param_type = hints.get(param_name, str)
        param_schema = python_type_to_json_schema(param_type)

        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            param_schema['default'] = param.default

        properties[param_name] = param_schema

    schema = {
        'type': 'object',
        'properties': properties
    }

    if required:
        schema['required'] = required

    return schema


def generate_output_schema_from_function(fn: Callable, permissive: bool = True) -> Dict[str, Any]:
    """
    Generate JSON schema from return type hint.

    Args:
        fn: Function to analyze
        permissive: If True (default), returns empty schema for Any/dict/None hints.
                   If False, attempts to generate strict schema from hints.

    Returns:
        JSON schema dict (empty dict = no validation)

    Examples:
        def foo() -> dict:  # permissive=True
            pass
        # Returns: {'type': 'object', 'additionalProperties': True}

        def bar() -> Any:  # permissive=True
            pass
        # Returns: {} (no validation)

        def baz() -> int:  # permissive=False
            pass
        # Returns: {'type': 'integer'}
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        # No type hints available
        return {}

    return_type = hints.get('return')

    # No return type hint -> accept anything
    if return_type is None:
        return {}

    # typing.Any -> accept anything (if permissive)
    if return_type is Any:
        return {} if permissive else {'type': 'object', 'additionalProperties': True}

    # dict without type params -> permissive object
    if return_type is dict:
        return {'type': 'object', 'additionalProperties': True}

    # Generate schema from type hint
    return python_type_to_json_schema(return_type)


def python_type_to_json_schema(python_type) -> Dict[str, Any]:
    """
    Convert Python type hint to JSON Schema.

    Supports:
    - Basic types: str, int, float, bool, dict, list
    - Optional[T] (Union[T, None])
    - List[T]
    - Dict[K, V]
    - Literal types (as enums)

    Args:
        python_type: Python type or type hint

    Returns:
        JSON schema dict
    """
    # Handle basic types
    type_map = {
        str: {'type': 'string'},
        int: {'type': 'integer'},
        float: {'type': 'number'},
        bool: {'type': 'boolean'},
        dict: {'type': 'object', 'additionalProperties': True},
        list: {'type': 'array'},
    }

    # Check if it's a basic type
    if python_type in type_map:
        return type_map[python_type]

    # Get generic origin (e.g., list from List[str])
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Union types (including Optional)
    if origin is Union:
        # Remove NoneType from union
        non_none_args = [arg for arg in args if arg is not type(None)]

        # Optional[T] -> just use T's schema
        if len(non_none_args) == 1:
            return python_type_to_json_schema(non_none_args[0])

        # Multiple types -> use anyOf
        return {
            'anyOf': [python_type_to_json_schema(arg) for arg in non_none_args]
        }

    # Handle List[T]
    if origin is list:
        if args:
            return {
                'type': 'array',
                'items': python_type_to_json_schema(args[0])
            }
        return {'type': 'array'}

    # Handle Dict[K, V]
    if origin is dict:
        # JSON Schema doesn't support typed dict keys/values directly
        return {'type': 'object', 'additionalProperties': True}

    # Check if it's a Pydantic model
    if hasattr(python_type, 'model_json_schema'):
        try:
            return python_type.model_json_schema()
        except Exception:
            pass

    # Check if it's a dataclass
    if hasattr(python_type, '__dataclass_fields__'):
        try:
            return _dataclass_to_json_schema(python_type)
        except Exception:
            pass

    # Default fallback - permissive object
    return {'type': 'object', 'additionalProperties': True}


def _dataclass_to_json_schema(dataclass_type) -> Dict[str, Any]:
    """Convert dataclass to JSON schema."""
    from dataclasses import fields

    properties = {}
    required = []

    for field in fields(dataclass_type):
        field_schema = python_type_to_json_schema(field.type)
        properties[field.name] = field_schema

        # Check if field has no default value
        if field.default == field.default_factory == dataclass_type.__dataclass_fields__[field.name].default:
            required.append(field.name)

    schema = {
        'type': 'object',
        'properties': properties
    }

    if required:
        schema['required'] = required

    return schema
