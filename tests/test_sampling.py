"""Tests for aix.sampling.sample / concurrent_map and aix.capabilities (issue #42)."""

import json
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import aix
import aix.capabilities
from aix.chat import Usage
from aix.sampling import Sample, Samples, concurrent_map, sample
from aix.capabilities import supports, supported_params

_aix_chat = sys.modules["aix.chat"]


def _response(texts, *, reason="stop", usage=None):
    choices = [
        SimpleNamespace(message=SimpleNamespace(content=t), finish_reason=reason)
        for t in texts
    ]
    return SimpleNamespace(choices=choices, usage=usage, model="m")


def _usage(p=2, c=3):
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


@pytest.fixture
def params_support():
    """Set what the (mocked) litellm param lookup reports: a set, or None."""
    state = {"params": None}

    def lookup(model):
        return state["params"]

    with patch.object(aix.capabilities, "_litellm_get_supported_openai_params", lookup):
        yield state


@pytest.fixture
def fake_completion():
    with patch.object(_aix_chat, "_litellm_completion") as mock:
        yield mock


def _provider_side_effect(**kwargs):
    n = kwargs.get("n", 1)
    return _response([f"t{i}" for i in range(n)], usage=_usage())


class TestCapabilities:
    def test_supports(self, params_support):
        params_support["params"] = ["n", "seed"]
        assert supports("gpt-x", "n") and not supports("gpt-x", "tools")
        assert supported_params("gpt-x") == frozenset({"n", "seed"})

    def test_unknown_model_is_not_supported(self, params_support):
        params_support["params"] = None
        assert supported_params("???") is None
        assert supports("???", "n") is False

    def test_lookup_error_is_not_supported(self):
        def boom(model):
            raise RuntimeError("cannot route")

        with patch.object(
            aix.capabilities, "_litellm_get_supported_openai_params", boom
        ):
            assert supports("x", "n") is False


class TestConcurrentMap:
    def test_order_preserved(self):
        def slow_identity(x):
            time.sleep(0.01 * (5 - x))
            return x

        assert concurrent_map(slow_identity, range(5), concurrency=5) == [0, 1, 2, 3, 4]

    def test_actually_concurrent(self):
        barrier = threading.Barrier(3, timeout=5)
        # Deadlocks (BrokenBarrierError) unless 3 calls run at the same time.
        assert concurrent_map(
            lambda x: barrier.wait() is not None, range(3), concurrency=3
        )

    def test_first_error_propagates(self):
        def f(x):
            if x == 2:
                raise KeyError(x)
            return x

        with pytest.raises(KeyError):
            concurrent_map(f, range(4), concurrency=2)


class TestSample:
    def test_provider_n_when_supported(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        fake_completion.side_effect = _provider_side_effect
        out = sample("q", n=3, seed=7)
        assert isinstance(out, Samples) and len(out) == 3
        assert out.strategy == "provider_n"
        assert fake_completion.call_count == 1
        sent = fake_completion.call_args.kwargs
        assert sent["n"] == 3 and sent["seed"] == 7
        assert out.texts == ("t0", "t1", "t2")
        assert all(s.usage is None and s.strategy == "provider_n" for s in out)
        assert out.usage == Usage(2, 3, 5)

    def test_independent_when_unsupported(self, fake_completion, params_support):
        params_support["params"] = ["temperature"]
        fake_completion.side_effect = _provider_side_effect
        out = sample("q", n=4, seed=10)
        assert out.strategy == "independent"
        assert fake_completion.call_count == 4
        assert all("n" not in call.kwargs for call in fake_completion.call_args_list)
        assert sorted(c.kwargs["seed"] for c in fake_completion.call_args_list) == [
            10,
            11,
            12,
            13,
        ]
        assert [s.seed for s in out] == [10, 11, 12, 13]
        assert all(s.usage == Usage(2, 3, 5) for s in out)
        assert out.usage == Usage(8, 12, 20)

    def test_unknown_support_falls_back_to_independent(
        self, fake_completion, params_support
    ):
        params_support["params"] = None
        fake_completion.side_effect = _provider_side_effect
        assert sample("q", n=2).strategy == "independent"

    def test_short_provider_batch_is_topped_up(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        fake_completion.side_effect = lambda **kw: _response(
            ["only-one"], usage=_usage()
        )
        out = sample("q", n=3, seed=0)
        assert out.strategy == "mixed" and len(out) == 3
        assert [s.strategy for s in out] == ["provider_n", "independent", "independent"]
        assert [s.seed for s in out] == [0, 1, 2]
        assert out.usage == Usage(6, 9, 15)

    def test_forced_strategies(self, fake_completion, params_support):
        params_support["params"] = None
        fake_completion.side_effect = _provider_side_effect
        assert sample("q", n=2, strategy="provider").strategy == "provider_n"
        params_support["params"] = ["n"]
        assert sample("q", n=2, strategy="independent").strategy == "independent"

    def test_n1_never_uses_provider_n(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        fake_completion.side_effect = _provider_side_effect
        out = sample("q", n=1)
        assert (
            out.strategy == "independent"
            and "n" not in fake_completion.call_args.kwargs
        )

    def test_parse_and_parse_errors(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        fake_completion.return_value = _response(['{"a": 1}', "not json"])
        out = sample("q", n=2, parse=json.loads)
        assert out[0].payload == {"a": 1} and out[0].ok
        assert out[1].payload is None and not out[1].ok
        assert isinstance(out[1].error, json.JSONDecodeError)

    def test_finish_reason_kept(self, fake_completion, params_support):
        params_support["params"] = None
        fake_completion.return_value = _response(["..."], reason="content_filter")
        assert sample("q", n=2)[0].finish_reason == "content_filter"

    def test_no_usage_reported(self, fake_completion, params_support):
        params_support["params"] = None
        fake_completion.return_value = _response(["x"])
        assert sample("q", n=2).usage is None

    def test_kwargs_forwarded(self, fake_completion, params_support):
        params_support["params"] = None
        fake_completion.return_value = _response(["x"])
        sample("q", n=1, model="gpt-4o", temperature=0.7, max_tokens=9)
        sent = fake_completion.call_args.kwargs
        assert (sent["model"], sent["temperature"], sent["max_tokens"]) == (
            "gpt-4o",
            0.7,
            9,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [dict(n=0), dict(n=2, concurrency=0), dict(n=2, strategy="nope"), dict(n=True)],
    )
    def test_bad_args(self, fake_completion, kwargs):
        with pytest.raises(ValueError):
            sample("q", **kwargs)
        fake_completion.assert_not_called()

    def test_independent_calls_run_concurrently(self, fake_completion, params_support):
        params_support["params"] = None
        barrier = threading.Barrier(3, timeout=5)

        def blocking(**kw):
            barrier.wait()
            return _response(["x"])

        fake_completion.side_effect = blocking
        assert len(sample("q", n=3, concurrency=3)) == 3

    def test_default_concurrency_from_config(self, fake_completion, params_support):
        params_support["params"] = None
        fake_completion.return_value = _response(["x"])
        with aix.using(chat_sample_concurrency=1):
            assert len(sample("q", n=2)) == 2

    def test_exports(self):
        assert aix.sample is sample and aix.Sample is Sample and aix.Samples is Samples


class TestConstrainedAnswerConcurrent:
    def test_n_calls_run_concurrently(self, fake_completion):
        barrier = threading.Barrier(3, timeout=5)

        def blocking(**kw):
            barrier.wait()
            return _response(['{"answer": "yes"}'])

        fake_completion.side_effect = blocking
        out = aix.constrained_answer("q", ["yes", "no"], n=3, concurrency=3)
        assert out == ["yes", "yes", "yes"]


def test_end_to_end_through_litellm_mock_response():
    """Real litellm (no network): real param lookup + litellm's mock_response."""
    pytest.importorskip("litellm")
    out = sample("q", n=3, model="gpt-4o-mini", mock_response="yes")
    assert out.strategy == "provider_n"  # litellm reports n support for OpenAI
    assert out.texts == ("yes", "yes", "yes")
    assert out.usage is not None and out.usage.total_tokens > 0
