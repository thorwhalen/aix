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
from aix.sampling import (
    Sample,
    Samples,
    SamplingError,
    _derived_seed,
    concurrent_map,
    sample,
)
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
        expected = [_derived_seed(10, j) for j in range(4)]
        assert sorted(
            c.kwargs["seed"] for c in fake_completion.call_args_list
        ) == sorted(expected)
        assert [s.seed for s in out] == expected
        assert len(set(expected)) == 4
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
        assert [s.seed for s in out] == [0, _derived_seed(0, 1), _derived_seed(0, 2)]
        assert out.usage == Usage(6, 9, 15)
        assert (out.n_requests, out.n_usage_reports) == (3, 3)

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
        [
            dict(n=0),
            dict(n=2, concurrency=0),
            dict(n=2, strategy="nope"),
            dict(n=True),
            dict(n=2, on_error="ignore"),
            dict(n=2, messages=[{"role": "user", "content": "other"}]),
            dict(n=2, stream=True),
        ],
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
        assert aix.SamplingError is SamplingError

    def test_seeds_do_not_overlap_across_base_seeds(self):
        run0 = {_derived_seed(0, j) for j in range(5)}
        run1 = {_derived_seed(1, j) for j in range(5)}
        assert not run0 & run1
        assert _derived_seed(7, 0) == 7

    def test_n1_forced_provider_is_a_plain_request(
        self, fake_completion, params_support
    ):
        params_support["params"] = ["n"]
        fake_completion.side_effect = _provider_side_effect
        out = sample("q", n=1, strategy="provider")
        assert out.strategy == "independent"
        assert "n" not in fake_completion.call_args.kwargs

    def test_auto_falls_back_when_provider_refuses_n(
        self, fake_completion, params_support
    ):
        class UnsupportedParamsError(Exception):  # litellm's name, matched by name
            pass

        params_support["params"] = ["n"]

        def refuse_n(**kw):
            if kw.get("n", 1) > 1:
                raise UnsupportedParamsError("n not supported")
            return _response(["x"], usage=_usage())

        fake_completion.side_effect = refuse_n
        out = sample("q", n=3)
        assert out.strategy == "independent" and len(out) == 3
        assert out.n_requests == 3  # the refused request is not counted
        with pytest.raises(UnsupportedParamsError):
            sample("q", n=3, strategy="provider")

    def test_auto_does_not_swallow_other_errors(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        fake_completion.side_effect = RuntimeError("rate limited")
        with pytest.raises(RuntimeError):
            sample("q", n=3)

    def _one_failure(self, fake_completion, params_support):
        params_support["params"] = None
        calls = {"k": 0}
        lock = threading.Lock()

        def flaky(**kw):
            with lock:
                calls["k"] += 1
                k = calls["k"]
            if k == 2:
                raise RuntimeError("boom")
            return _response(["x"], usage=_usage())

        fake_completion.side_effect = flaky

    def test_request_failure_raises_with_partial_batch(
        self, fake_completion, params_support
    ):
        self._one_failure(fake_completion, params_support)
        with pytest.raises(SamplingError) as info:
            sample("q", n=4, concurrency=2)
        partial = info.value.samples
        # Requests stop launching after the failure; what was made is all kept.
        assert len(partial) == partial.n_requests == fake_completion.call_count
        assert 2 <= partial.n_requests <= 4
        assert len(partial.errors) == 1
        assert isinstance(info.value.__cause__, RuntimeError)
        assert partial.usage.total_tokens == 5 * (partial.n_requests - 1)

    def test_raise_stops_launching_after_first_failure(
        self, fake_completion, params_support
    ):
        params_support["params"] = None
        fake_completion.side_effect = RuntimeError("invalid key")
        with pytest.raises(SamplingError) as info:
            sample("q", n=20, concurrency=2)
        assert fake_completion.call_count <= 3  # 2 in flight (+1 race), not 20
        assert info.value.samples.n_requests == fake_completion.call_count

    def test_record_draws_everything(self, fake_completion, params_support):
        params_support["params"] = None
        fake_completion.side_effect = RuntimeError("invalid key")
        out = sample("q", n=5, concurrency=2, on_error="record")
        assert fake_completion.call_count == 5 and len(out.errors) == 5

    def test_sampling_error_pickles(self):
        import pickle

        err = SamplingError("x", samples=Samples([], strategy="independent"))
        assert str(pickle.loads(pickle.dumps(err))) == "x"

    def test_n1_sends_base_seed(self, fake_completion, params_support):
        params_support["params"] = None
        fake_completion.return_value = _response(["x"])
        sample("q", n=1, seed=42)
        assert fake_completion.call_args.kwargs["seed"] == 42

    def test_request_failure_recorded(self, fake_completion, params_support):
        self._one_failure(fake_completion, params_support)
        out = sample("q", n=4, on_error="record")
        failed = [s for s in out if not s.ok]
        assert len(failed) == 1 and failed[0].payload is None and failed[0].text == ""

    def test_missing_key_raises_itself_once(self, monkeypatch, fake_completion):
        from aix.credentials import (
            MissingCredentialError,
            PROVIDER_ENV_VARS,
            provider_env_vars,
        )

        for provider in PROVIDER_ENV_VARS:
            for env_name in provider_env_vars(provider):
                monkeypatch.delenv(env_name, raising=False)
        monkeypatch.setattr("aix.credentials._lookup_store", lambda name: None)
        monkeypatch.setattr("aix.credentials._dotenv_loaded", True)
        with pytest.raises(MissingCredentialError):
            sample("q", n=3, model="anthropic/claude-x", strategy="independent")
        fake_completion.assert_not_called()


class TestDotenvLoadIsThreadSafe:
    def test_concurrent_first_load_waits_for_the_load(self, monkeypatch):
        """A thread must not see the 'loaded' flag before the .env is loaded."""
        import aix.credentials as creds

        dotenv = pytest.importorskip("dotenv")
        monkeypatch.setattr(creds, "_dotenv_loaded", False)
        monkeypatch.delenv("AIX_TEST_DOTENV_KEY", raising=False)
        started = threading.Event()

        def slow_load(*args, **kwargs):
            started.set()
            time.sleep(0.2)
            monkeypatch.setenv("AIX_TEST_DOTENV_KEY", "loaded")
            return True

        monkeypatch.setattr(dotenv, "load_dotenv", slow_load)
        monkeypatch.setattr(dotenv, "find_dotenv", lambda **kw: "")
        seen = {}

        def second():
            started.wait(5)
            creds._maybe_load_dotenv()
            import os

            seen["value"] = os.environ.get("AIX_TEST_DOTENV_KEY")

        t = threading.Thread(target=second)
        t.start()
        creds._maybe_load_dotenv()
        t.join(5)
        assert seen["value"] == "loaded"


class TestConstrainedAnswerConcurrent:
    def test_n_calls_run_concurrently(self, fake_completion):
        barrier = threading.Barrier(3, timeout=5)

        def blocking(**kw):
            barrier.wait()
            return _response(['{"answer": "yes"}'])

        fake_completion.side_effect = blocking
        out = aix.constrained_answer("q", ["yes", "no"], n=3, concurrency=3)
        assert out == ["yes", "yes", "yes"]

    def test_fail_fast_does_not_start_queued_calls(self, fake_completion):
        fake_completion.return_value = _response(['{"answer": "maybe"}'])
        with pytest.raises(aix.ConstraintViolation):
            aix.constrained_answer("q", ["yes", "no"], n=20, concurrency=2)
        assert fake_completion.call_count < 20

    @pytest.mark.parametrize("bad", [0, -1, "5"])
    def test_bad_concurrency(self, fake_completion, bad):
        with pytest.raises(ValueError):
            aix.constrained_answer("q", ["yes", "no"], n=2, concurrency=bad)


def test_end_to_end_through_litellm_mock_response():
    """Real litellm (no network): real param lookup + litellm's mock_response."""
    pytest.importorskip("litellm")
    out = sample("q", n=3, model="gpt-4o-mini", mock_response="yes")
    assert out.strategy == "provider_n"  # litellm reports n support for OpenAI
    assert out.texts == ("yes", "yes", "yes")
    assert out.usage is not None and out.usage.total_tokens > 0


class TestServerSideNRefusal:
    """Issue #55: fall back when the provider refuses n with a 400."""

    class BadRequestError(Exception):  # litellm's name, matched by name
        pass

    def _refuse_n(self, fake_completion, *, plain_ok=True):
        calls = []

        def respond(**kw):
            calls.append(kw)
            if kw.get("n", 1) > 1 or not plain_ok:
                raise self.BadRequestError("'n' must be 1 for this model")
            return _response(["x"], usage=_usage())

        fake_completion.side_effect = respond
        return calls

    def test_falls_back_with_probe(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        calls = self._refuse_n(fake_completion)
        out = sample("q", n=4, concurrency=4)
        assert out.strategy == "independent" and len(out) == 4
        assert isinstance(out.fallback_error, self.BadRequestError)
        assert out.n_requests == 5  # the refused (sent) n request + 4 singles
        assert calls[0]["n"] == 4 and all("n" not in c for c in calls[1:])

    def test_unrelated_400_raises_original(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        calls = self._refuse_n(fake_completion, plain_ok=False)
        with pytest.raises(self.BadRequestError):
            sample("q", n=4, concurrency=4)
        assert len(calls) == 2  # the n request + one probe, not a batch of copies

    def test_forced_provider_does_not_fall_back(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        self._refuse_n(fake_completion)
        with pytest.raises(self.BadRequestError):
            sample("q", n=3, strategy="provider")

    def test_no_fallback_error_on_normal_paths(self, fake_completion, params_support):
        params_support["params"] = ["n"]
        fake_completion.side_effect = _provider_side_effect
        assert sample("q", n=2).fallback_error is None
        params_support["params"] = None
        assert sample("q", n=2).fallback_error is None

    def test_litellm_refusal_recorded_too(self, fake_completion, params_support):
        class UnsupportedParamsError(Exception):
            pass

        params_support["params"] = ["n"]

        def refuse(**kw):
            if kw.get("n", 1) > 1:
                raise UnsupportedParamsError("n")
            return _response(["x"])

        fake_completion.side_effect = refuse
        out = sample("q", n=2)
        assert isinstance(out.fallback_error, UnsupportedParamsError)
        assert out.n_requests == 2


def test_real_litellm_bad_request_is_recognised():
    litellm = pytest.importorskip("litellm")
    from aix.sampling import _is_bad_request_error

    err = litellm.BadRequestError("n must be 1", model="m", llm_provider="groq")
    assert _is_bad_request_error(err)
