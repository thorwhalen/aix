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
from aix.config import get_config as _get_config, resolve_model as _resolve_model


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
    matches = re.findall(r"\{(\w+)\}", template)
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
            "required": list(schema.keys()),
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
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        text = "\n".join(lines[1:-1])
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
    egress: Callable[[Any], Any] = None,
    model: str = None,
    temperature: float = None,
    name: str = None,
    **chat_kwargs,
) -> Callable:
    """Create a callable function from a prompt template.

    This is the main function for creating prompt-based functions. It automatically
    detects parameters from the template and creates a function with those parameters.

    Without output_schema: Returns text
    With output_schema: Returns structured data (dict, list, etc.)
    With egress: Returns whatever the egress post-processor returns.

    Args:
        template: Prompt template with {var} placeholders for parameters
        output_schema: Optional schema for structured output. Can be:
            - Dict mapping field names to types: {"name": str, "age": int}
            - A single type for simple outputs: str, int, list, etc.
            - None for plain text output (default)
        egress: Optional post-processor ``(result) -> Any`` applied to the output
            before returning — on both the text and structured paths. Lets a caller
            keep "prompt → typed Python value" inside the facade (e.g. parse the
            LLM text into a list of lines/ids) instead of wrapping the returned
            function. ``None`` (default) returns the raw result unchanged.
        model: Model to use for this function
        temperature: Temperature for generation
        name: Optional ``__name__`` for the generated function (for tracing /
            identity). Defaults to ``"prompt_based_function"``.
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
                **chat_kwargs,
            )
            result = _parse_structured_output(response, json_schema)
        else:
            # Plain text output
            result = chat(
                formatted_prompt, model=model, temperature=temperature, **chat_kwargs
            )
        # Optional post-processing (e.g. parse text into a list of ids/lines).
        return egress(result) if egress is not None else result

    # Set function name and docstring
    generated_function.__name__ = name or "prompt_based_function"
    generated_function.__doc__ = (
        f"Auto-generated function from prompt template.\n\n"
        f"Template: {template}\n\n"
        f"Parameters:\n"
    )
    for param in param_names:
        generated_function.__doc__ += f"    {param}: Parameter from template\n"

    if output_schema:
        generated_function.__doc__ += (
            f"\nReturns: Structured data matching schema: {output_schema}"
        )
    else:
        generated_function.__doc__ += "\nReturns: Generated text"

    # Add metadata
    generated_function.template = template
    generated_function.output_schema = output_schema
    generated_function.egress = egress
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
        **kwargs,
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
            func_kwargs.setdefault("model", self._model)

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
        self.add("summarize", "Summarize this text concisely: {text}")
        self.add("explain", "Explain this concept simply: {concept}")
        self.add("translate", "Translate {text} to {language}")
        self.add("paraphrase", "Paraphrase this text: {text}")

        # Analysis functions
        self.add(
            "extract_keywords",
            "Extract the main keywords from: {text}",
            output_schema=list,
        )
        self.add(
            "sentiment",
            "Analyze the sentiment of: {text}",
            output_schema={"sentiment": str, "score": float, "explanation": str},
        )

        # Generation functions
        self.add("continue_text", "Continue this text naturally: {text}")
        self.add("generate_title", "Generate a catchy title for: {text}")


# Create singleton instance
common_funcs = CommonFuncs()


# -----------------------------------------------------------------------------
# constrained_answer: Force LLM to choose from valid options


def _enhance_prompt_for_json(
    prompt: str,
    valid_answers: Union[list[str], list[int], list[float], type, tuple[float, float]],
    min_val: float = None,
    max_val: float = None,
) -> str:
    """Add JSON formatting instructions to the prompt.

    This is a helper that adds explicit instructions for JSON output.
    Only used when enhance_prompt=True.
    """
    if isinstance(valid_answers, list):
        valid_answers_list = "\n".join(f"- {answer}" for answer in valid_answers)
        return f"""
{prompt}
You must respond with EXACTLY one of these options:
{valid_answers_list}
Choose the most appropriate answer. Return your response as JSON with an "answer" field.
"""
    elif valid_answers is bool:
        return f"""
{prompt}
You must respond with either true or false.
Return your response as JSON with an "answer" field.
"""
    elif valid_answers is int:
        return f"""
{prompt}
You must respond with an integer number.
Return your response as JSON with an "answer" field.
"""
    elif valid_answers is float:
        return f"""
{prompt}
You must respond with a number.
Return your response as JSON with an "answer" field.
"""
    elif isinstance(valid_answers, tuple) and len(valid_answers) == 2:
        return f"""
{prompt}
You must respond with a number between {min_val} and {max_val} (inclusive).
Return your response as JSON with an "answer" field.
"""
    else:
        # Fallback - just add minimal JSON instruction
        return f"{prompt}\nReturn your response as JSON with an 'answer' field."


def constrained_answer(
    prompt: str,
    valid_answers: Union[list[str], list[int], list[float], type, tuple[float, float]],
    *,
    model: str = None,
    temperature: float = None,
    enhance_prompt: bool = False,
    n: int = 1,
):
    """
    Get an answer from the LLM constrained to a set of valid answers or types.

    Uses JSON mode to ensure the LLM returns a valid response based on constraints.
    More flexible than the oa version - works with any model that supports JSON mode
    via LiteLLM.

    This can be seen as a facade for some common structured output use cases, as well
    as a convenient tool to do response statistics and validation (via n>1).

    Args:
        prompt: The question or prompt to ask the LLM
        valid_answers: Can be:
            - list[str]: List of valid string options
            - list[int]: List of valid integer options
            - list[float]: List of valid float options
            - bool: Constrains answer to True or False
            - int: Any integer
            - float: Any number
            - tuple[float, float]: Numerical range (min, max) inclusive
        model: The model to use for the LLM (default: uses DFLT_CHAT_MODEL)
        temperature: Temperature for sampling (default: None, uses model's default).
            Higher values (e.g., 1.0) give more random/varied results.
            Lower values (e.g., 0.0) give more deterministic results.
        enhance_prompt: If True, adds explicit instructions to the prompt about
            JSON formatting and constraints. If False (default), relies on
            response_format alone. Default is False to match oa behavior.
        n: Number of times to call the LLM (default: 1)

    Returns:
        One of the valid answers, respecting the type constraint.
        If n > 1, returns a list of answers.

    Examples:
        >>> # String options
        >>> answer = constrained_answer(
        ...     "Is Python compiled or interpreted?",
        ...     ["compiled", "interpreted", "both"]
        ... )  # doctest: +SKIP
        >>> answer in ["compiled", "interpreted", "both"]  # doctest: +SKIP
        True

        >>> # Boolean
        >>> answer = constrained_answer(
        ...     "Is Python dynamically typed?",
        ...     bool
        ... )  # doctest: +SKIP
        >>> isinstance(answer, bool)  # doctest: +SKIP
        True

        >>> # Integer options
        >>> answer = constrained_answer(
        ...     "How many wheels does a car have?",
        ...     [2, 3, 4, 6, 8]
        ... )  # doctest: +SKIP
        >>> answer in [2, 3, 4, 6, 8]  # doctest: +SKIP
        True

        >>> # Numerical range
        >>> answer = constrained_answer(
        ...     "What is a reasonable hourly rate for a senior Python developer? (USD)",
        ...     (50.0, 300.0)
        ... )  # doctest: +SKIP
        >>> 50.0 <= answer <= 300.0  # doctest: +SKIP
        True

        >>> # Multiple samples for statistics
        >>> answers = constrained_answer(
        ...     "Which is better: cats or dogs?",
        ...     ["cats", "dogs"],
        ...     n=10
        ... )  # doctest: +SKIP
        >>> len(answers)  # doctest: +SKIP
        10
    """
    if n != 1:
        from functools import partial

        f = partial(
            constrained_answer,
            prompt,
            valid_answers,
            model=model,
            temperature=temperature,
            enhance_prompt=enhance_prompt,
            n=1,
        )
        return [f() for _ in range(n)]

    # Determine the expected type for type conversion
    expected_type = None

    if isinstance(valid_answers, list):
        # List of specific options
        if not valid_answers:
            raise ValueError("valid_answers list cannot be empty")

        first_item = valid_answers[0]
        if not isinstance(first_item, (str, int, float)):
            raise ValueError(f"Unsupported list item type: {type(first_item)}")

        expected_type = type(first_item)

    elif valid_answers is bool:
        expected_type = bool

    elif valid_answers is int:
        expected_type = int

    elif valid_answers is float:
        expected_type = float

    elif isinstance(valid_answers, tuple) and len(valid_answers) == 2:
        # Numerical range
        min_val, max_val = valid_answers
        if not isinstance(min_val, (int, float)) or not isinstance(
            max_val, (int, float)
        ):
            raise ValueError("Range tuple must contain two numbers")

        expected_type = float

    else:
        raise ValueError(f"Unsupported valid_answers type: {type(valid_answers)}")

    # Build the prompt template
    if enhance_prompt:
        # Add explicit JSON instructions and constraints
        if isinstance(valid_answers, tuple) and len(valid_answers) == 2:
            template = _enhance_prompt_for_json(
                prompt,
                valid_answers,
                min_val=valid_answers[0],
                max_val=valid_answers[1],
            )
        else:
            template = _enhance_prompt_for_json(prompt, valid_answers)
    else:
        # Use the prompt as-is, but add minimal JSON instruction with type hint
        # (OpenAI requires the word "json" to appear in the prompt when using json_object mode)
        if isinstance(valid_answers, list):
            # For lists, we need to be more specific
            type_hint = f"one of: {', '.join(map(str, valid_answers))}"
        elif valid_answers is bool:
            type_hint = "true or false"
        elif valid_answers is int:
            type_hint = "a number (integer)"
        elif valid_answers is float:
            type_hint = "a number"
        elif isinstance(valid_answers, tuple):
            type_hint = f"a number between {valid_answers[0]} and {valid_answers[1]}"
        else:
            type_hint = "your answer"

        template = f'{prompt}\nRespond in JSON format with an "answer" field containing {type_hint}.'

    # Use LiteLLM's response_format for JSON mode
    # This works across OpenAI, Anthropic (via tools), and other providers
    chat_kwargs = {
        "model": _resolve_model(model or _get_config().chat.model),
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:
        chat_kwargs["temperature"] = temperature

    response = chat(template, **chat_kwargs)

    try:
        result = json.loads(response)
        answer = result["answer"]

        # Convert to expected type if needed
        if expected_type is not None and not isinstance(answer, expected_type):
            # JSON might parse integers as strings or vice versa
            # Try to convert to the expected type
            try:
                answer = expected_type(answer)
            except (ValueError, TypeError):
                # If conversion fails, just return as-is
                pass

        return answer
    except (json.JSONDecodeError, KeyError) as e:
        # Failed to parse - raise informative error
        raise ValueError(
            f"Failed to parse constrained answer. Response: {response[:200]}..."
        ) from e
