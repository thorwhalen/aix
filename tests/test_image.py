"""Image generation builds the provider request from what the caller gave, nothing more."""

from unittest.mock import Mock, patch

import aix.image as image_module
from aix.image import generate_image


def _fake_response(b64="aGk="):
    data = Mock()
    data.url = None
    data.b64_json = b64
    data.revised_prompt = "a revised prompt"
    response = Mock()
    response.data = [data]
    return response


def test_response_format_is_sent_only_when_given():
    """OpenAI's images endpoint rejects ``response_format`` for its current models,
    so the parameter must not be sent unless the caller asked for it."""
    fake = Mock(return_value=_fake_response())
    with patch.object(image_module, "_image_generation", return_value=fake):
        result = generate_image("a lighthouse", model="openai/gpt-image-1", size="1024x1024")
        assert "response_format" not in fake.call_args.kwargs
        assert fake.call_args.kwargs["prompt"] == "a lighthouse"
        assert result.b64_json == "aGk=" and result.revised_prompt == "a revised prompt"

        generate_image("a lighthouse", model="openai/gpt-image-1", response_format="b64_json")
        assert fake.call_args.kwargs["response_format"] == "b64_json"


def test_quality_and_style_are_optional_too():
    fake = Mock(return_value=_fake_response())
    with patch.object(image_module, "_image_generation", return_value=fake):
        generate_image("a lighthouse", model="openai/gpt-image-1")
        sent = fake.call_args.kwargs
        assert "quality" not in sent and "style" not in sent
        generate_image("a lighthouse", model="openai/gpt-image-1", quality="high")
        assert fake.call_args.kwargs["quality"] == "high"
