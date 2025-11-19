"""Prompt-based function creation for AIX.

Transform text prompts into callable Python functions with automatic parameter extraction.

This module provides the core functionality for creating reusable AI-powered functions
from natural language prompts. It supports both text generation and structured output.

Examples:
    Create a simple text function:
    >>> from aix.prompts import prompt_func
    >>> translate = prompt_func("Translate to French: {text}")
    >>> translate(text="Hello world")  # doctest: +SKIP
    'Bonjour le monde'

    Create a function with structured output:
    >>> extract_person = prompt_func(
    ...     "Extract person information from: {text}",
    ...     output_schema={"name": str, "age": int}
    ... )
    >>> extract_person(text="Alice is 30 years old")  # doctest: +SKIP
    {'name': 'Alice', 'age': 30}

    Multiple parameters:
    >>> compare = prompt_func(
    ...     "Compare {item1} and {item2} in terms of {aspect}"
    ... )
    >>> compare(item1="Python", item2="JavaScript", aspect="performance")  # doctest: +SKIP
    'Python generally has better...'
"""

import re
import json
from collections.abc import Callable
from typing import Union, Any, get_type_hints
from functools import wraps
from inspect import signature, Parameter

# Import from aix.chat
from aix.chat import chat, DFLT_CHAT_MODEL


def _extract_template_vars(template: str) -> list[str]:
    """Extract variable names from a template string.

    Args:
        template: Template string with {var} placeholders

    Returns:
        List of variable names found in template

    Examples:
        >>> _extract_template_vars("Translate {text} to {language}")
        ['text', 'language']

        >>> _extract_template_vars("No variables here")
        []

        >>> _extract_template_vars("{a} {b} {a}")  # Duplicates removed
        ['a', 'b']
    """
    # Find all {var} patterns
    matches = re.findall(r'\{(\w+)\}', template)
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for var in matches:
        if var not in seen:
            seen.add(var)
            result.append(var)
    return result


def _format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with given parameters.

    Args:
        template: Template string
        **kwargs: Values for template variables

    Returns:
        Formatted prompt string

    Examples:
        >>> _format_prompt("Hello {name}", name="Alice")
        'Hello Alice'
    """
    return template.format(**kwargs)


def _schema_to_json_schema(schema: Union[dict, type]) -> dict:
    """Convert a simple schema to JSON Schema format.

    Args:
        schema: Either a dict mapping field names to types, or a type annotation

    Returns:
        JSON Schema dict

    Examples:
        >>> _schema_to_json_schema({"name": str, "age": int})
        {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}, 'required': ['name', 'age']}

        >>> _schema_to_json_schema(str)
        {'type': 'string'}
    """
    if isinstance(schema, dict):
        # Dictionary mapping field names to types
        properties = {}
        for field_name, field_type in schema.items():
            properties[field_name] = _type_to_json_type(field_type)

        return {
            "type": "object",
            "properties": properties,
            "required": list(schema.keys())
        }
    else:
        # Single type
        return _type_to_json_type(schema)


def _type_to_json_type(python_type: type) -> dict:
    """Convert Python type to JSON Schema type.

    Args:
        python_type: Python type (str, int, float, bool, list, dict)

    Returns:
        JSON Schema type definition
    """
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    return type_map.get(python_type, {"type": "string"})


def _parse_structured_output(response_text: str, schema: dict) -> Any:
    """Parse structured output from LLM response.

    Args:
        response_text: Raw text response from LLM
        schema: Expected schema

    Returns:
        Parsed structured data

    Raises:
        ValueError: If response cannot be parsed
    """
    # Try to extract JSON from response
    # Handle cases where LLM wraps JSON in markdown code blocks
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split('\n')
        # Remove first and last lines (``` markers)
        text = '\n'.join(lines[1:-1])
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse structured output. Expected JSON, got: {text[:100]}..."
        ) from e


def prompt_func(
    template: str,
    *,
    output_schema: Union[dict, type] = None,
    model: str = None,
    temperature: float = None,
    **chat_kwargs
) -> Callable:
    """Create a callable function from a prompt template.

    This is the main function for creating prompt-based functions. It automatically
    detects parameters from the template and creates a function with those parameters.

    Without output_schema: Returns text
    With output_schema: Returns structured data (dict, list, etc.)

    Args:
        template: Prompt template with {var} placeholders for parameters
        output_schema: Optional schema for structured output. Can be:
            - Dict mapping field names to types: {"name": str, "age": int}
            - A single type for simple outputs: str, int, list, etc.
            - None for plain text output (default)
        model: Model to use for this function
        temperature: Temperature for generation
        **chat_kwargs: Additional parameters passed to chat()

    Returns:
        Callable function with parameters derived from template

    Examples:
        >>> # Simple text generation
        >>> summarize = prompt_func("Summarize this text: {text}")
        >>> summarize(text="Long article...")  # doctest: +SKIP
        'Brief summary...'

        >>> # Structured output
        >>> extract = prompt_func(
        ...     "Extract contact info from: {text}",
        ...     output_schema={"name": str, "email": str, "phone": str}
        ... )
        >>> result = extract(text="Contact John at john@example.com, 555-1234")  # doctest: +SKIP
        >>> result['name']  # doctest: +SKIP
        'John'

        >>> # Multiple parameters
        >>> compare = prompt_func(
        ...     "Compare {item1} and {item2} in terms of {aspect}. "
        ...     "Keep it under {word_limit} words."
        ... )
        >>> compare(
        ...     item1="Python",
        ...     item2="Java",
        ...     aspect="learning curve",
        ...     word_limit=50
        ... )  # doctest: +SKIP
        'Python has a gentler learning curve...'

        >>> # With specific model
        >>> creative_writer = prompt_func(
        ...     "Write a creative story about {topic}",
        ...     model="gpt-4o",
        ...     temperature=1.5
        ... )
        >>> creative_writer(topic="a time-traveling cat")  # doctest: +SKIP
        'Once upon a time, there was a cat named Whiskers...'
    """
    # Extract parameter names from template
    param_names = _extract_template_vars(template)

    # Build the actual function
    def generated_function(**kwargs):
        # Validate parameters
        for param in param_names:
            if param not in kwargs:
                raise TypeError(
                    f"Missing required parameter: {param}. "
                    f"Required parameters: {param_names}"
                )

        # Format the prompt
        formatted_prompt = _format_prompt(template, **kwargs)

        # If structured output is requested, modify the prompt
        if output_schema is not None:
            json_schema = _schema_to_json_schema(output_schema)
            # Add instruction for JSON output
            structured_prompt = (
                f"{formatted_prompt}\n\n"
                f"Respond with valid JSON matching this schema: {json.dumps(json_schema)}\n"
                f"Only return the JSON, no additional text."
            )
            response = chat(
                structured_prompt,
                model=model,
                temperature=temperature or 0.0,  # Lower temp for structured output
                **chat_kwargs
            )
            return _parse_structured_output(response, json_schema)
        else:
            # Plain text output
            response = chat(
                formatted_prompt,
                model=model,
                temperature=temperature,
                **chat_kwargs
            )
            return response

    # Set function name and docstring
    generated_function.__name__ = "prompt_based_function"
    generated_function.__doc__ = (
        f"Auto-generated function from prompt template.\n\n"
        f"Template: {template}\n\n"
        f"Parameters:\n"
    )
    for param in param_names:
        generated_function.__doc__ += f"    {param}: Parameter from template\n"

    if output_schema:
        generated_function.__doc__ += f"\nReturns: Structured data matching schema: {output_schema}"
    else:
        generated_function.__doc__ += "\nReturns: Generated text"

    # Add metadata
    generated_function.template = template
    generated_function.output_schema = output_schema
    generated_function.param_names = param_names

    return generated_function


# Convenience aliases for explicit intent
def prompt_to_text(template: str, **kwargs) -> Callable:
    """Create a function that returns text output.

    This is an explicit alias for prompt_func without output_schema.

    Args:
        template: Prompt template
        **kwargs: Additional parameters for prompt_func

    Returns:
        Function that returns text

    Examples:
        >>> summarize = prompt_to_text("Summarize: {text}")
        >>> result = summarize(text="Long text...")  # doctest: +SKIP
        >>> isinstance(result, str)  # doctest: +SKIP
        True
    """
    return prompt_func(template, output_schema=None, **kwargs)


def prompt_to_json(template: str, schema: Union[dict, type], **kwargs) -> Callable:
    """Create a function that returns structured JSON output.

    This is an explicit alias for prompt_func with output_schema.

    Args:
        template: Prompt template
        schema: Output schema
        **kwargs: Additional parameters for prompt_func

    Returns:
        Function that returns structured data

    Examples:
        >>> extract = prompt_to_json(
        ...     "Extract name and age from: {text}",
        ...     schema={"name": str, "age": int}
        ... )
        >>> result = extract(text="Alice is 30")  # doctest: +SKIP
        >>> isinstance(result, dict)  # doctest: +SKIP
        True
    """
    return prompt_func(template, output_schema=schema, **kwargs)


class PromptFuncs:
    """Collection of prompt-based functions.

    Provides a namespace for organizing related prompt functions with
    attribute-based access.

    Examples:
        >>> funcs = PromptFuncs()  # doctest: +SKIP
        >>> funcs.add('summarize', "Summarize: {text}")  # doctest: +SKIP
        >>> funcs.add('translate', "Translate {text} to {language}")  # doctest: +SKIP
        >>> funcs.summarize(text="Long article...")  # doctest: +SKIP
        'Summary...'
        >>> funcs.translate(text="Hello", language="Spanish")  # doctest: +SKIP
        'Hola'
    """

    def __init__(self, model: str = None, **default_kwargs):
        """Initialize collection.

        Args:
            model: Default model for all functions
            **default_kwargs: Default parameters for all functions
        """
        self._functions = {}
        self._model = model
        self._default_kwargs = default_kwargs

    def add(
        self,
        name: str,
        template: str,
        *,
        output_schema: Union[dict, type] = None,
        **kwargs
    ) -> None:
        """Add a function to the collection.

        Args:
            name: Function name (will be accessible as attribute)
            template: Prompt template
            output_schema: Optional schema for structured output
            **kwargs: Additional parameters for prompt_func
        """
        # Merge default kwargs
        func_kwargs = {**self._default_kwargs, **kwargs}
        if self._model:
            func_kwargs.setdefault('model', self._model)

        # Create function
        func = prompt_func(template, output_schema=output_schema, **func_kwargs)
        self._functions[name] = func

        # Make accessible as attribute
        setattr(self, name, func)

    def __getitem__(self, name: str) -> Callable:
        """Get function by name."""
        return self._functions[name]

    def __contains__(self, name: str) -> bool:
        """Check if function exists."""
        return name in self._functions

    def keys(self):
        """Get all function names."""
        return self._functions.keys()


# Pre-built common functions
class CommonFuncs(PromptFuncs):
    """Collection of commonly used prompt functions.

    Examples:
        >>> from aix.prompts import common_funcs
        >>> common_funcs.summarize(text="Long article...")  # doctest: +SKIP
        'Summary...'
        >>> common_funcs.extract_keywords(text="Article about AI")  # doctest: +SKIP
        ['AI', 'artificial', 'intelligence']
    """

    def __init__(self, model: str = None):
        super().__init__(model=model)

        # Text transformation functions
        self.add('summarize', "Summarize this text concisely: {text}")
        self.add('explain', "Explain this concept simply: {concept}")
        self.add('translate', "Translate {text} to {language}")
        self.add('paraphrase', "Paraphrase this text: {text}")

        # Analysis functions
        self.add(
            'extract_keywords',
            "Extract the main keywords from: {text}",
            output_schema=list
        )
        self.add(
            'sentiment',
            "Analyze the sentiment of: {text}",
            output_schema={"sentiment": str, "score": float, "explanation": str}
        )

        # Generation functions
        self.add('continue_text', "Continue this text naturally: {text}")
        self.add('generate_title', "Generate a catchy title for: {text}")


# Create singleton instance
common_funcs = CommonFuncs()
