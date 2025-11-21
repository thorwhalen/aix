"""Tests for aix.prompts module."""

import pytest
import json
from unittest.mock import Mock, patch
from aix.prompts import (
    prompt_func,
    prompt_to_text,
    prompt_to_json,
    PromptFuncs,
    CommonFuncs,
    _extract_template_vars,
    _format_prompt,
    _schema_to_json_schema,
    _parse_structured_output,
)


class TestExtractTemplateVars:
    """Tests for template variable extraction."""

    def test_single_variable(self):
        """Test extracting single variable."""
        vars = _extract_template_vars("Hello {name}")
        assert vars == ["name"]

    def test_multiple_variables(self):
        """Test extracting multiple variables."""
        vars = _extract_template_vars("{greeting} {name}, how are you?")
        assert vars == ["greeting", "name"]

    def test_no_variables(self):
        """Test template with no variables."""
        vars = _extract_template_vars("Hello world")
        assert vars == []

    def test_duplicate_variables(self):
        """Test that duplicates are removed."""
        vars = _extract_template_vars("{name} {name} {age}")
        assert vars == ["name", "age"]


class TestFormatPrompt:
    """Tests for prompt formatting."""

    def test_simple_format(self):
        """Test simple formatting."""
        result = _format_prompt("Hello {name}", name="Alice")
        assert result == "Hello Alice"

    def test_multiple_params(self):
        """Test formatting with multiple parameters."""
        result = _format_prompt(
            "{greeting} {name}!",
            greeting="Hi",
            name="Bob"
        )
        assert result == "Hi Bob!"


class TestSchemaToJsonSchema:
    """Tests for schema conversion."""

    def test_dict_schema(self):
        """Test converting dict schema."""
        schema = {"name": str, "age": int}
        json_schema = _schema_to_json_schema(schema)

        assert json_schema['type'] == 'object'
        assert 'properties' in json_schema
        assert json_schema['properties']['name']['type'] == 'string'
        assert json_schema['properties']['age']['type'] == 'integer'
        assert json_schema['required'] == ['name', 'age']

    def test_single_type_schema(self):
        """Test converting single type."""
        json_schema = _schema_to_json_schema(str)
        assert json_schema['type'] == 'string'


class TestParseStructuredOutput:
    """Tests for structured output parsing."""

    def test_parse_json(self):
        """Test parsing plain JSON."""
        text = '{"name": "Alice", "age": 30}'
        result = _parse_structured_output(text, {})
        assert result == {"name": "Alice", "age": 30}

    def test_parse_markdown_json(self):
        """Test parsing JSON in markdown code blocks."""
        text = '```json\n{"name": "Bob"}\n```'
        result = _parse_structured_output(text, {})
        assert result == {"name": "Bob"}

    def test_parse_code_block_json(self):
        """Test parsing JSON in plain code blocks."""
        text = '```\n{"name": "Charlie"}\n```'
        result = _parse_structured_output(text, {})
        assert result == {"name": "Charlie"}

    def test_parse_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_structured_output("not json", {})


class TestPromptFunc:
    """Tests for prompt_func."""

    @patch('aix.prompts.chat')
    def test_simple_prompt_func(self, mock_chat):
        """Test creating simple prompt function."""
        mock_chat.return_value = "Response"

        func = prompt_func("Summarize: {text}")
        result = func(text="Long article")

        assert result == "Response"
        # Verify chat was called with formatted prompt
        call_args = mock_chat.call_args[0][0]
        assert "Summarize: Long article" == call_args

    @patch('aix.prompts.chat')
    def test_prompt_func_multiple_params(self, mock_chat):
        """Test prompt function with multiple parameters."""
        mock_chat.return_value = "Comparison"

        func = prompt_func("Compare {item1} and {item2}")
        func(item1="Python", item2="Java")

        call_args = mock_chat.call_args[0][0]
        assert "Compare Python and Java" == call_args

    @patch('aix.prompts.chat')
    def test_prompt_func_missing_param(self, mock_chat):
        """Test that missing parameter raises TypeError."""
        func = prompt_func("Hello {name}")

        with pytest.raises(TypeError, match="Missing required parameter"):
            func()

    @patch('aix.prompts.chat')
    def test_prompt_func_with_schema(self, mock_chat):
        """Test prompt function with output schema."""
        mock_chat.return_value = '{"name": "Alice", "age": 30}'

        func = prompt_func(
            "Extract: {text}",
            output_schema={"name": str, "age": int}
        )
        result = func(text="Alice is 30")

        assert result == {"name": "Alice", "age": 30}
        # Verify lower temperature for structured output
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs['temperature'] == 0.0

    @patch('aix.prompts.chat')
    def test_prompt_func_with_model(self, mock_chat):
        """Test prompt function with specific model."""
        mock_chat.return_value = "Response"

        func = prompt_func("Test: {text}", model="gpt-4o")
        func(text="input")

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs['model'] == "gpt-4o"

    @patch('aix.prompts.chat')
    def test_prompt_func_metadata(self, mock_chat):
        """Test that created function has correct metadata."""
        func = prompt_func("Test {param}")

        assert "prompt_based_function" in func.__name__
        assert "param" in func.__doc__
        assert func.template == "Test {param}"
        assert func.param_names == ["param"]


class TestPromptToText:
    """Tests for prompt_to_text convenience function."""

    @patch('aix.prompts.prompt_func')
    def test_prompt_to_text(self, mock_prompt_func):
        """Test prompt_to_text calls prompt_func correctly."""
        prompt_to_text("Test {text}")

        mock_prompt_func.assert_called_once_with(
            "Test {text}",
            output_schema=None
        )


class TestPromptToJson:
    """Tests for prompt_to_json convenience function."""

    @patch('aix.prompts.prompt_func')
    def test_prompt_to_json(self, mock_prompt_func):
        """Test prompt_to_json calls prompt_func correctly."""
        schema = {"name": str}
        prompt_to_json("Test {text}", schema=schema)

        mock_prompt_func.assert_called_once_with(
            "Test {text}",
            output_schema=schema
        )


class TestPromptFuncs:
    """Tests for PromptFuncs class."""

    @patch('aix.prompts.chat')
    def test_add_function(self, mock_chat):
        """Test adding function to collection."""
        funcs = PromptFuncs()
        funcs.add('summarize', "Summarize: {text}")

        assert 'summarize' in funcs
        assert hasattr(funcs, 'summarize')

    @patch('aix.prompts.chat')
    def test_use_function(self, mock_chat):
        """Test using function from collection."""
        mock_chat.return_value = "Summary"

        funcs = PromptFuncs()
        funcs.add('summarize', "Summarize: {text}")

        result = funcs.summarize(text="Long text")
        assert result == "Summary"

    @patch('aix.prompts.chat')
    def test_collection_with_default_model(self, mock_chat):
        """Test collection with default model."""
        mock_chat.return_value = "Response"

        funcs = PromptFuncs(model="gpt-4o")
        funcs.add('test', "Test: {text}")
        funcs.test(text="input")

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs['model'] == "gpt-4o"

    @patch('aix.prompts.chat')
    def test_getitem(self, mock_chat):
        """Test accessing function via getitem."""
        funcs = PromptFuncs()
        funcs.add('test', "Test: {text}")

        assert funcs['test'] is not None
        assert callable(funcs['test'])

    @patch('aix.prompts.chat')
    def test_keys(self, mock_chat):
        """Test getting function names."""
        funcs = PromptFuncs()
        funcs.add('func1', "Test1: {text}")
        funcs.add('func2', "Test2: {text}")

        keys = list(funcs.keys())
        assert 'func1' in keys
        assert 'func2' in keys


class TestCommonFuncs:
    """Tests for CommonFuncs pre-built collection."""

    def test_has_summarize(self):
        """Test that common_funcs has summarize."""
        from aix.prompts import common_funcs
        assert 'summarize' in common_funcs
        assert callable(common_funcs.summarize)

    def test_has_translate(self):
        """Test that common_funcs has translate."""
        from aix.prompts import common_funcs
        assert 'translate' in common_funcs

    def test_has_sentiment(self):
        """Test that common_funcs has sentiment analysis."""
        from aix.prompts import common_funcs
        assert 'sentiment' in common_funcs
