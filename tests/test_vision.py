"""Tests for aix.vision module (offline — LiteLLM is mocked, no API calls)."""

import sys
from unittest.mock import Mock, patch

import pytest

from aix.vision import describe_image, to_image_content

# `aix.vision` may resolve to the exported function name on some builds; patch
# the submodule object directly (same robustness trick as test_chat.py).
import aix.vision  # noqa: F401

_aix_vision = sys.modules["aix.vision"]


def _mock_response(text="A description."):
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = text
    return response


class TestToImageContent:
    """The multimodal content-block builder."""

    def test_http_url_passthrough(self):
        block = to_image_content("https://example.com/a.jpg")
        assert block == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.jpg"},
        }

    def test_data_uri_passthrough(self):
        uri = "data:image/png;base64,AAAA"
        assert to_image_content(uri)["image_url"]["url"] == uri

    def test_bytes_to_data_uri_sniffs_png(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        url = to_image_content(png)["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_bytes_default_mime_when_unknown(self):
        url = to_image_content(b"not-a-known-magic")["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_path_to_data_uri(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        url = to_image_content(str(p))["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_detail_forwarded(self):
        block = to_image_content("https://x/y.jpg", detail="low")
        assert block["image_url"]["detail"] == "low"

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            to_image_content(12345)


class TestDescribeImage:
    """The image->text primitive."""

    @patch.object(_aix_vision, "_litellm_completion")
    def test_url_builds_multimodal_message(self, mock_completion):
        mock_completion.return_value = _mock_response("A grey cat.")

        result = describe_image("https://example.com/cat.jpg", model="gpt-4o")

        assert result == "A grey cat."
        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args[1]
        content = kwargs["messages"][0]["content"]
        # text block first, then the image_url block
        assert content[0] == {"type": "text", "text": "Describe this image in detail."}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/cat.jpg"

    @patch.object(_aix_vision, "_litellm_completion")
    def test_custom_prompt(self, mock_completion):
        mock_completion.return_value = _mock_response("Red.")

        describe_image("https://x/car.jpg", prompt="What colour?", model="gpt-4o")

        content = mock_completion.call_args[1]["messages"][0]["content"]
        assert content[0]["text"] == "What colour?"

    @patch.object(_aix_vision, "_litellm_completion")
    def test_max_tokens_and_temperature_threaded(self, mock_completion):
        mock_completion.return_value = _mock_response()

        describe_image(
            "https://x/a.jpg", model="gpt-4o", max_tokens=64, temperature=0.0
        )

        kwargs = mock_completion.call_args[1]
        assert kwargs["max_tokens"] == 64
        assert kwargs["temperature"] == 0.0

    @patch.object(_aix_vision, "_litellm_completion")
    def test_detail_threaded_into_image_block(self, mock_completion):
        mock_completion.return_value = _mock_response()

        describe_image("https://x/a.jpg", model="gpt-4o", detail="high")

        block = mock_completion.call_args[1]["messages"][0]["content"][1]
        assert block["image_url"]["detail"] == "high"

    @patch.object(_aix_vision, "_litellm_completion")
    def test_extra_kwargs_forwarded(self, mock_completion):
        mock_completion.return_value = _mock_response()

        describe_image("https://x/a.jpg", model="gpt-4o", top_p=0.5)

        assert mock_completion.call_args[1]["top_p"] == 0.5

    def test_missing_litellm_raises_importerror(self):
        with patch.object(_aix_vision, "_litellm_completion", None):
            with pytest.raises(ImportError, match="LiteLLM is required"):
                describe_image("https://x/a.jpg", model="gpt-4o")
