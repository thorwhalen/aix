"""Tests for aix.vision.compare_images (offline — LiteLLM is mocked, $0 spend).

These mirror test_vision.py's approach: patch the ``_litellm_completion`` object
on the ``aix.vision`` submodule and feed it a canned response, so no real API
call (and no token spend) ever happens.
"""

import json
import sys
from unittest.mock import Mock, patch

import pytest

from aix.vision import (
    compare_images,
    ImageComparison,
    RubricVerdict,
    DFLT_COMPARE_RUBRIC,
    DFLT_FILM_RUBRIC,
)

# `aix.vision` may resolve to the exported function name on some builds; patch
# the submodule object directly (same robustness trick as test_vision.py).
import aix.vision  # noqa: F401

_aix_vision = sys.modules["aix.vision"]


def _verdict_json(rubric=DFLT_COMPARE_RUBRIC, *, match=True, confidence=0.9):
    """A well-formed JSON verdict covering every aspect in ``rubric``."""
    return json.dumps(
        {
            "match": match,
            "confidence": confidence,
            "explanation": "Candidate matches the reference well.",
            "aspects": [
                {
                    "aspect": aspect,
                    "match": match,
                    "confidence": confidence,
                    "note": f"{aspect} looks consistent.",
                }
                for aspect in rubric
            ],
        }
    )


def _mock_response(text):
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = text
    return response


class TestStructuredResult:
    """The verdict parses into the documented typed shape."""

    @patch.object(_aix_vision, "_litellm_completion")
    def test_default_rubric_shape(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json())

        result = compare_images(
            "https://x/candidate.jpg", "https://x/ref.jpg", model="gpt-4o", api_key="sk-test"
        )

        assert isinstance(result, ImageComparison)
        assert result.match is True
        assert 0.0 <= result.confidence <= 1.0
        assert result.explanation
        assert result.model == "gpt-4o"
        # One verdict per default rubric aspect, in order, mapping-like access.
        assert len(result) == len(DFLT_COMPARE_RUBRIC)
        assert tuple(result) == DFLT_COMPARE_RUBRIC
        for aspect in DFLT_COMPARE_RUBRIC:
            assert aspect in result
            verdict = result[aspect]
            assert isinstance(verdict, RubricVerdict)
            assert verdict.aspect == aspect
            assert isinstance(verdict.match, bool)
            assert 0.0 <= verdict.confidence <= 1.0

    @patch.object(_aix_vision, "_litellm_completion")
    def test_mapping_helpers(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json())
        result = compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")
        # .get with default, __contains__ negative
        assert result.get("identity") is result["identity"]
        assert result.get("nonexistent") is None
        assert "nonexistent" not in result


class TestRubricHonored:
    """Caller-supplied rubric drives the aspects."""

    @patch.object(_aix_vision, "_litellm_completion")
    def test_custom_rubric_aspects_appear(self, mock_completion):
        custom = ("face", "architecture", "props", "lighting")
        mock_completion.return_value = _mock_response(_verdict_json(custom))

        result = compare_images(
            "https://x/c.jpg", "https://x/r.jpg", rubric=custom, model="gpt-4o", api_key="sk-test"
        )

        assert tuple(result) == custom
        # The rubric aspects must be threaded into the prompt text.
        prompt = mock_completion.call_args[1]["messages"][0]["content"][0]["text"]
        for aspect in custom:
            assert aspect in prompt

    @patch.object(_aix_vision, "_litellm_completion")
    def test_film_rubric(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json(DFLT_FILM_RUBRIC))
        result = compare_images(
            "https://x/c.jpg", "https://x/r.jpg", rubric=DFLT_FILM_RUBRIC, model="gpt-4o", api_key="sk-test"
        )
        assert "face_identity" in result
        assert result["skin_realism"].aspect == "skin_realism"

    @patch.object(_aix_vision, "_litellm_completion")
    def test_missing_aspect_filled_conservatively(self, mock_completion):
        # Model omits one aspect; result must still cover the full rubric.
        partial = json.dumps(
            {
                "match": False,
                "confidence": 0.4,
                "explanation": "Mixed.",
                "aspects": [
                    {"aspect": "identity", "match": True, "confidence": 0.9, "note": "ok"}
                ],
            }
        )
        mock_completion.return_value = _mock_response(partial)

        result = compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")

        assert len(result) == len(DFLT_COMPARE_RUBRIC)
        assert result["identity"].match is True
        # An omitted aspect (e.g. "props") is present as a conservative non-match.
        assert result["props"].match is False
        assert result["props"].confidence == 0.0


class TestReferenceInputs:
    """Reference may be a single image or a locked set (sequence)."""

    @patch.object(_aix_vision, "_litellm_completion")
    def test_single_reference_one_image_block(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json())

        compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")

        content = mock_completion.call_args[1]["messages"][0]["content"]
        # text + candidate + 1 reference = 3 blocks
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 2
        # candidate is first image block
        assert image_blocks[0]["image_url"]["url"] == "https://x/c.jpg"
        assert image_blocks[1]["image_url"]["url"] == "https://x/r.jpg"

    @patch.object(_aix_vision, "_litellm_completion")
    def test_reference_sequence_builds_multiple_blocks(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json())

        refs = ["https://x/r1.jpg", "https://x/r2.jpg", "https://x/r3.jpg"]
        compare_images("https://x/c.jpg", refs, model="gpt-4o", api_key="sk-test")

        content = mock_completion.call_args[1]["messages"][0]["content"]
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        # candidate + 3 references = 4 image blocks
        assert len(image_blocks) == 4
        urls = [b["image_url"]["url"] for b in image_blocks]
        assert urls == ["https://x/c.jpg", *refs]

    @patch.object(_aix_vision, "_litellm_completion")
    def test_data_uri_reference_treated_as_atomic(self, mock_completion):
        # A data: URI string must NOT be iterated character-wise.
        mock_completion.return_value = _mock_response(_verdict_json())
        uri = "data:image/png;base64,AAAA"
        compare_images("https://x/c.jpg", uri, model="gpt-4o", api_key="sk-test")
        content = mock_completion.call_args[1]["messages"][0]["content"]
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 2
        assert image_blocks[1]["image_url"]["url"] == uri


class TestThreading:
    """Model params are threaded into the LiteLLM call."""

    @patch.object(_aix_vision, "_litellm_completion")
    def test_temperature_defaults_to_zero(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json())
        compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")
        assert mock_completion.call_args[1]["temperature"] == 0.0

    @patch.object(_aix_vision, "_litellm_completion")
    def test_max_tokens_and_detail_threaded(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json())
        compare_images(
            "https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test", max_tokens=256, detail="high"
        )
        kwargs = mock_completion.call_args[1]
        assert kwargs["max_tokens"] == 256
        blocks = [b for b in kwargs["messages"][0]["content"] if b.get("type") == "image_url"]
        assert all(b["image_url"]["detail"] == "high" for b in blocks)

    @patch.object(_aix_vision, "_litellm_completion")
    def test_extra_kwargs_forwarded(self, mock_completion):
        mock_completion.return_value = _mock_response(_verdict_json())
        compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test", top_p=0.3)
        assert mock_completion.call_args[1]["top_p"] == 0.3


class TestLenientParsing:
    """The JSON contract tolerates code fences and surrounding prose."""

    @patch.object(_aix_vision, "_litellm_completion")
    def test_code_fence_stripped(self, mock_completion):
        fenced = "```json\n" + _verdict_json() + "\n```"
        mock_completion.return_value = _mock_response(fenced)
        result = compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")
        assert result.match is True

    @patch.object(_aix_vision, "_litellm_completion")
    def test_prose_wrapped_json_recovered(self, mock_completion):
        wrapped = "Here is my verdict:\n" + _verdict_json() + "\nHope that helps!"
        mock_completion.return_value = _mock_response(wrapped)
        result = compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")
        assert len(result) == len(DFLT_COMPARE_RUBRIC)

    @patch.object(_aix_vision, "_litellm_completion")
    def test_unparseable_reply_raises(self, mock_completion):
        mock_completion.return_value = _mock_response("not json at all, sorry")
        with pytest.raises(ValueError, match="parse a JSON verdict"):
            compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")


class TestErrors:
    """Sensible errors on empty / invalid input — raised before any API call."""

    def test_empty_rubric_raises(self):
        with pytest.raises(ValueError, match="at least one aspect"):
            compare_images("https://x/c.jpg", "https://x/r.jpg", rubric=(), model="gpt-4o", api_key="sk-test")

    def test_empty_reference_sequence_raises(self):
        with pytest.raises(ValueError, match="at least one image"):
            compare_images("https://x/c.jpg", [], model="gpt-4o", api_key="sk-test")

    def test_missing_litellm_raises_importerror(self):
        with patch.object(_aix_vision, "_litellm_completion", None):
            with pytest.raises(ImportError, match="LiteLLM is required"):
                compare_images("https://x/c.jpg", "https://x/r.jpg", model="gpt-4o", api_key="sk-test")

    @patch.object(_aix_vision, "_litellm_completion")
    def test_errors_raised_before_api_call(self, mock_completion):
        # Empty rubric must short-circuit BEFORE the (paid) completion call.
        with pytest.raises(ValueError):
            compare_images("https://x/c.jpg", "https://x/r.jpg", rubric=[], model="gpt-4o", api_key="sk-test")
        mock_completion.assert_not_called()
