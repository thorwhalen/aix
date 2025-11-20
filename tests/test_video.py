"""Tests for aix.video module."""

import pytest
from unittest.mock import Mock, patch
from aix.video import (
    GeneratedVideo,
    generate_video,
    animate_image,
    get_available_providers,
    estimate_cost,
)


class TestGeneratedVideo:
    """Tests for GeneratedVideo class."""

    def test_initialization(self):
        """Test initialization."""
        video = GeneratedVideo(
            url="https://example.com/video.mp4",
            model="test-model",
            prompt="A cat playing",
            duration=5.0,
            resolution="1920x1080",
            status="completed"
        )
        assert video.url == "https://example.com/video.mp4"
        assert video.model == "test-model"
        assert video.prompt == "A cat playing"
        assert video.duration == 5.0
        assert video.resolution == "1920x1080"
        assert video.status == "completed"

    @patch('requests.get')
    def test_as_bytes_from_url(self, mock_get):
        """Test getting bytes from URL."""
        mock_response = Mock()
        mock_response.content = b'video_data'
        mock_get.return_value = mock_response

        video = GeneratedVideo(url="https://example.com/video.mp4")
        data = video.as_bytes()

        assert data == b'video_data'

    def test_as_bytes_from_data(self):
        """Test getting bytes from stored data."""
        video = GeneratedVideo(data=b'video_bytes')
        assert video.as_bytes() == b'video_bytes'

    def test_repr(self):
        """Test string representation."""
        video = GeneratedVideo(
            model="test", duration=5.0, status="completed"
        )
        repr_str = repr(video)
        assert "GeneratedVideo" in repr_str
        assert "test" in repr_str
        assert "5.0s" in repr_str


class TestGenerateVideo:
    """Tests for generate_video function."""

    def test_not_implemented(self):
        """Test that generate_video raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Video generation requires"):
            generate_video("A beautiful sunset")

    def test_error_message_helpful(self):
        """Test that error message is helpful."""
        try:
            generate_video("Test prompt")
        except NotImplementedError as e:
            error_msg = str(e)
            assert "provider configuration" in error_msg.lower()
            assert "RUNWAY_API_KEY" in error_msg or "api" in error_msg.lower()


class TestAnimateImage:
    """Tests for animate_image function."""

    def test_not_implemented(self):
        """Test that animate_image raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            animate_image("image.jpg", prompt="Move clouds")


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""

    @patch.dict('os.environ', {}, clear=True)
    def test_no_providers(self):
        """Test with no providers configured."""
        providers = get_available_providers()
        # May have some providers based on installed packages
        assert isinstance(providers, list)

    @patch.dict('os.environ', {'RUNWAY_API_KEY': 'test-key'})
    def test_runway_available(self):
        """Test with Runway configured."""
        providers = get_available_providers()
        assert 'runway' in providers

    @patch.dict('os.environ', {'PIKA_API_KEY': 'test-key'})
    def test_pika_available(self):
        """Test with Pika configured."""
        providers = get_available_providers()
        assert 'pika' in providers


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_estimate_single_provider(self):
        """Test cost estimation for single provider."""
        cost = estimate_cost(duration=5, provider='runway')
        assert 'runway' in cost
        assert isinstance(cost['runway'], (int, float))
        assert cost['runway'] > 0

    def test_estimate_all_providers(self):
        """Test cost estimation for all providers."""
        cost = estimate_cost(duration=10)
        assert isinstance(cost, dict)
        assert len(cost) > 0
        # Should have estimates for known providers
        assert 'runway' in cost or 'pika' in cost

    def test_cost_scales_with_duration(self):
        """Test that cost increases with duration."""
        cost_short = estimate_cost(duration=2, provider='runway')
        cost_long = estimate_cost(duration=10, provider='runway')

        assert cost_long['runway'] > cost_short['runway']
