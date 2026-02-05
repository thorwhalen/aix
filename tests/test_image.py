"""Tests for aix.image module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from aix.image import (
    generate_image,
    generate_images,
    GeneratedImage,
)


class TestGeneratedImage:
    """Tests for GeneratedImage class."""

    def test_initialization_with_url(self):
        """Test initialization with URL."""
        img = GeneratedImage(
            url="https://example.com/image.png", model="dall-e-3", prompt="A cat"
        )
        assert img.url == "https://example.com/image.png"
        assert img.model == "dall-e-3"
        assert img.prompt == "A cat"

    def test_initialization_with_b64(self):
        """Test initialization with base64 data."""
        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        img = GeneratedImage(b64_json=b64_data, model="dall-e-2")
        assert img.b64_json == b64_data

    @patch("aix.image.base64.b64decode")
    def test_as_bytes_from_b64(self, mock_b64decode):
        """Test getting bytes from base64."""
        mock_b64decode.return_value = b"fake_image_data"
        img = GeneratedImage(b64_json="fake_b64_string")

        data = img.as_bytes()

        assert data == b"fake_image_data"
        mock_b64decode.assert_called_once_with("fake_b64_string")

    @patch("requests.get")
    def test_as_bytes_from_url(self, mock_get):
        """Test getting bytes from URL."""
        mock_response = Mock()
        mock_response.content = b"image_from_url"
        mock_get.return_value = mock_response

        img = GeneratedImage(url="https://example.com/image.png")
        data = img.as_bytes()

        assert data == b"image_from_url"
        mock_get.assert_called_once_with("https://example.com/image.png")

    def test_repr(self):
        """Test string representation."""
        img = GeneratedImage(url="https://example.com/img.png", model="dall-e-3")
        repr_str = repr(img)
        assert "GeneratedImage" in repr_str
        assert "dall-e-3" in repr_str


class TestGenerateImage:
    """Tests for generate_image function."""

    @patch("aix.image._litellm_image_generation")
    def test_simple_generation(self, mock_image_gen):
        """Test simple image generation."""
        # Mock response
        mock_data = Mock()
        mock_data.url = "https://example.com/generated.png"
        mock_data.b64_json = None
        mock_data.revised_prompt = None

        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_image_gen.return_value = mock_response

        result = generate_image("A beautiful sunset")

        assert isinstance(result, GeneratedImage)
        assert result.url == "https://example.com/generated.png"
        assert result.prompt == "A beautiful sunset"

        mock_image_gen.assert_called_once()
        call_kwargs = mock_image_gen.call_args[1]
        assert call_kwargs["prompt"] == "A beautiful sunset"
        assert call_kwargs["n"] == 1

    @patch("aix.image._litellm_image_generation")
    def test_generation_with_model(self, mock_image_gen):
        """Test generation with specific model."""
        mock_data = Mock()
        mock_data.url = "https://example.com/img.png"
        mock_data.b64_json = None
        mock_data.revised_prompt = None

        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_image_gen.return_value = mock_response

        generate_image("Test prompt", model="dall-e-3")

        call_kwargs = mock_image_gen.call_args[1]
        assert call_kwargs["model"] == "dall-e-3"

    @patch("aix.image._litellm_image_generation")
    def test_generation_with_size(self, mock_image_gen):
        """Test generation with specific size."""
        mock_data = Mock()
        mock_data.url = "https://example.com/img.png"
        mock_data.b64_json = None
        mock_data.revised_prompt = None

        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_image_gen.return_value = mock_response

        generate_image("Test", size="512x512")

        call_kwargs = mock_image_gen.call_args[1]
        assert call_kwargs["size"] == "512x512"

    @patch("aix.image._litellm_image_generation")
    def test_generation_with_quality(self, mock_image_gen):
        """Test generation with quality setting."""
        mock_data = Mock()
        mock_data.url = "https://example.com/img.png"
        mock_data.b64_json = None
        mock_data.revised_prompt = None

        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_image_gen.return_value = mock_response

        generate_image("Test", quality="hd")

        call_kwargs = mock_image_gen.call_args[1]
        assert call_kwargs["quality"] == "hd"


class TestGenerateImages:
    """Tests for generate_images function."""

    @patch("aix.image._litellm_image_generation")
    def test_multiple_images(self, mock_image_gen):
        """Test generating multiple images."""
        # Mock multiple image responses
        mock_data1 = Mock()
        mock_data1.url = "https://example.com/img1.png"
        mock_data1.b64_json = None
        mock_data1.revised_prompt = None

        mock_data2 = Mock()
        mock_data2.url = "https://example.com/img2.png"
        mock_data2.b64_json = None
        mock_data2.revised_prompt = None

        mock_response = Mock()
        mock_response.data = [mock_data1, mock_data2]
        mock_image_gen.return_value = mock_response

        results = generate_images("A robot", n=2)

        assert len(results) == 2
        assert isinstance(results[0], GeneratedImage)
        assert isinstance(results[1], GeneratedImage)
        assert results[0].url == "https://example.com/img1.png"
        assert results[1].url == "https://example.com/img2.png"

        call_kwargs = mock_image_gen.call_args[1]
        assert call_kwargs["n"] == 2

    @patch("aix.image._litellm_image_generation")
    def test_default_number(self, mock_image_gen):
        """Test default number of images."""
        mock_data = Mock()
        mock_data.url = "https://example.com/img.png"
        mock_data.b64_json = None
        mock_data.revised_prompt = None

        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_image_gen.return_value = mock_response

        results = generate_images("Test")

        call_kwargs = mock_image_gen.call_args[1]
        # Should default to DFLT_NUM_IMAGES (1)
        assert call_kwargs["n"] >= 1
