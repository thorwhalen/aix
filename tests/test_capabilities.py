"""Tests for seed / determinism discoverability (issue #44)."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import aix
import aix.capabilities
import aix.structured
from aix.capabilities import CAPABILITIES, supports

_aix_chat = sys.modules["aix.chat"]


def _response(text="x", fingerprint="fp_1"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason="stop"
    )
    return SimpleNamespace(
        choices=[choice], usage=None, model="m", system_fingerprint=fingerprint
    )


def test_top_level_exports():
    assert aix.supports is supports
    assert aix.supported_params is aix.capabilities.supported_params


def test_seed_support_from_param_list():
    with patch.object(
        aix.capabilities, "_litellm_get_supported_openai_params", lambda model: ["seed"]
    ):
        assert supports("m", "seed") and not supports("m", "n")


def test_response_schema_capability_routes_to_schema_check():
    assert "response_schema" in CAPABILITIES
    with (
        patch.object(
            aix.structured, "_litellm_supports_response_schema", lambda model: True
        ),
        patch.object(
            aix.capabilities, "_litellm_get_supported_openai_params", lambda model: []
        ),
    ):
        assert supports("m", "response_schema")
    with patch.object(
        aix.structured, "_litellm_supports_response_schema", lambda model: False
    ):
        assert not supports("m", "response_schema")


def test_capability_resolves_alias():
    seen = {}

    def check(model):
        seen["model"] = model
        return True

    with patch.dict(CAPABILITIES, {"response_schema": check}):
        with aix.using(aliases={"judge": "openai/gpt-x"}):
            supports("judge", "response_schema")
    assert seen["model"] == "openai/gpt-x"


def test_real_litellm_seed_support():
    pytest.importorskip("litellm")
    assert supports("gpt-4o-mini", "seed") is True
    assert supports("gpt-4o-mini", "response_schema") is True
    assert supports("anthropic/claude-sonnet-4-20250514", "seed") is False


def test_completion_and_samples_carry_fingerprint():
    with patch.object(_aix_chat, "_litellm_completion") as mock:
        mock.return_value = _response(fingerprint="fp_abc")
        assert aix.complete("q", seed=1).system_fingerprint == "fp_abc"
        out = aix.sample("q", n=2, seed=1, strategy="independent")
    assert {s.system_fingerprint for s in out} == {"fp_abc"}


def test_fingerprint_absent_is_none():
    with patch.object(_aix_chat, "_litellm_completion") as mock:
        mock.return_value = SimpleNamespace(
            choices=_response().choices, usage=None, model="m"
        )
        assert aix.complete("q").system_fingerprint is None


def test_non_string_param_is_false():
    assert supports("gpt-4o-mini", ["seed"]) is False
    assert supports("gpt-4o-mini", None) is False


def test_broken_capability_check_is_false():
    with patch.dict(CAPABILITIES, {"x": lambda m: 1 / 0, "y": lambda m: "yes"}):
        assert supports("m", "x") is False
        assert supports("m", "y") is True


def test_suite_uses_bundled_cost_map():
    """The root conftest must use a value current litellm honours ("true")."""
    import os

    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"].lower() == "true"
