"""Tests for aix.chat.complete / Completion / Usage (issue #41)."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import aix
from aix.chat import Completion, Usage, chat, complete

_aix_chat = sys.modules["aix.chat"]


def _fake_response(texts, *, reasons=None, usage=None, model="served-model"):
    """A LiteLLM-shaped response with one choice per text."""
    reasons = reasons or ["stop"] * len(texts)
    choices = [
        SimpleNamespace(message=SimpleNamespace(content=t), finish_reason=r)
        for t, r in zip(texts, reasons)
    ]
    return SimpleNamespace(choices=choices, usage=usage, model=model)


@pytest.fixture
def fake_completion():
    with patch.object(_aix_chat, "_litellm_completion") as mock:
        yield mock


class TestUsage:
    def test_from_response_object(self):
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        )
        assert Usage.from_response(resp) == Usage(3, 4, 7)

    def test_total_derived_when_missing(self):
        resp = {"usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        assert Usage.from_response(resp).total_tokens == 7

    def test_absent_usage_is_none(self):
        assert Usage.from_response(SimpleNamespace(usage=None)) is None
        assert Usage.from_response({"usage": {}}) is None

    def test_non_integer_counts_are_ignored(self):
        resp = {"usage": {"prompt_tokens": "3", "completion_tokens": True}}
        assert Usage.from_response(resp) is None

    def test_breakdowns_openai_shape(self):
        resp = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 8},
                "completion_tokens_details": {"reasoning_tokens": 4},
            }
        }
        u = Usage.from_response(resp)
        assert (u.cached_prompt_tokens, u.reasoning_tokens) == (8, 4)

    def test_breakdowns_anthropic_shape(self):
        resp = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_input_tokens": 6,
                "cache_creation_input_tokens": 3,
            }
        }
        u = Usage.from_response(resp)
        assert (u.cached_prompt_tokens, u.cache_creation_tokens) == (6, 3)

    def test_integral_float_counts_accepted(self):
        resp = {"usage": {"prompt_tokens": 3.0, "completion_tokens": 4.0}}
        assert Usage.from_response(resp).total_tokens == 7

    def test_add_zero_both_sides_and_empty_sum(self):
        u = Usage(1, 1, 2)
        assert u + 0 == u and 0 + u == u
        assert sum([], Usage()) == Usage()
        with pytest.raises(TypeError):
            u + 1

    def test_usages_sum(self):
        assert sum([Usage(1, 2, 3), Usage(10, 20, 30)]) == Usage(11, 22, 33)
        assert Usage(1, 1, 2) + Usage(1, 1, 2) == Usage(2, 2, 4)


class TestCompletionFromResponse:
    def test_all_choices_kept(self):
        resp = _fake_response(
            ["a", "b", "c"],
            reasons=["stop", "length", "content_filter"],
            usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        )
        c = Completion.from_response(resp)
        assert c.texts == ("a", "b", "c")
        assert c.finish_reasons == ("stop", "length", "content_filter")
        assert c.usage == Usage(5, 6, 11)
        assert c.model == "served-model"
        assert c.raw is resp
        assert c.text == "a" and c.finish_reason == "stop" and str(c) == "a"

    def test_none_content_is_empty_string(self):
        c = Completion.from_response(_fake_response([None]))
        assert c.texts == ("",)

    def test_list_of_parts_content_joined(self):
        parts = [{"type": "text", "text": "a"}, {"type": "image_url"}, {"text": "b"}]
        assert Completion.from_response(_fake_response([parts])).texts == ("ab",)

    def test_no_choices(self):
        c = Completion.from_response(
            SimpleNamespace(choices=[], usage=None, model=None)
        )
        assert c.texts == () and c.text == "" and c.finish_reason is None

    def test_dict_shaped_response(self):
        resp = {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "model": "m",
        }
        c = Completion.from_response(resp)
        assert (c.texts, c.finish_reasons, c.model, c.usage) == (
            ("x",),
            ("stop",),
            "m",
            None,
        )

    def test_real_litellm_model_response(self):
        litellm = pytest.importorskip("litellm")
        resp = litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                    "index": 0,
                },
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "length",
                    "index": 1,
                },
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            model="gpt-x",
        )
        c = Completion.from_response(resp)
        assert c.texts == ("hi", "")
        assert c.finish_reasons == ("stop", "length")
        assert c.usage == Usage(3, 4, 7)
        assert c.model == "gpt-x"


class TestComplete:
    def test_returns_completion_with_all_choices(self, fake_completion):
        fake_completion.return_value = _fake_response(["x", "y"])
        c = complete("hi", n=2)
        assert isinstance(c, Completion)
        assert c.texts == ("x", "y")
        assert fake_completion.call_args.kwargs["n"] == 2

    def test_optional_params_only_sent_when_set(self, fake_completion):
        fake_completion.return_value = _fake_response(["x"])
        complete("hi")
        sent = fake_completion.call_args.kwargs
        assert "n" not in sent and "seed" not in sent and "response_format" not in sent
        assert "stream" not in sent

    def test_seed_and_response_format_forwarded(self, fake_completion):
        fake_completion.return_value = _fake_response(["{}"])
        fmt = {"type": "json_object"}
        complete("hi", seed=7, response_format=fmt, top_p=0.5)
        sent = fake_completion.call_args.kwargs
        assert (
            sent["seed"] == 7
            and sent["response_format"] == fmt
            and sent["top_p"] == 0.5
        )

    def test_shares_defaults_and_messages_with_chat(self, fake_completion):
        fake_completion.return_value = _fake_response(["x"])
        chat([{"role": "user", "content": "q"}], model="gpt-4o", temperature=0.3)
        chat_kwargs = dict(fake_completion.call_args.kwargs)
        complete([{"role": "user", "content": "q"}], model="gpt-4o", temperature=0.3)
        complete_kwargs = dict(fake_completion.call_args.kwargs)
        assert chat_kwargs.pop("stream") is False
        assert chat_kwargs == complete_kwargs

    @pytest.mark.parametrize("bad_n", [0, -1, 1.5, True, "2"])
    def test_rejects_bad_n(self, fake_completion, bad_n):
        with pytest.raises(ValueError, match="positive integer"):
            complete("hi", n=bad_n)
        fake_completion.assert_not_called()

    def test_bad_n_reported_before_missing_credentials(self, monkeypatch):
        from aix.credentials import PROVIDER_ENV_VARS, provider_env_vars

        for provider in PROVIDER_ENV_VARS:
            for env_name in provider_env_vars(provider):
                monkeypatch.delenv(env_name, raising=False)
        monkeypatch.setattr("aix.credentials._lookup_store", lambda name: None)
        monkeypatch.setattr("aix.credentials._dotenv_loaded", True)
        with pytest.raises(ValueError, match="positive integer"):
            complete("hi", model="anthropic/claude-x", n=0)

    def test_numpy_style_integral_n_accepted(self, fake_completion):
        import numbers

        class Int64(int):  # stands in for np.int64 (an Integral, not a bool)
            pass

        assert isinstance(Int64(2), numbers.Integral)
        fake_completion.return_value = _fake_response(["x", "y"])
        complete("hi", n=Int64(2))
        assert type(fake_completion.call_args.kwargs["n"]) is int

    def test_rejects_stream(self, fake_completion):
        with pytest.raises(ValueError, match="does not stream"):
            complete("hi", stream=True)

    def test_exported_at_top_level(self):
        assert aix.complete is complete
        assert aix.Completion is Completion and aix.Usage is Usage

    def test_end_to_end_through_litellm_mock_response(self):
        """Real litellm (no network): litellm's own mock_response path."""
        pytest.importorskip("litellm")
        c = complete("hi", model="gpt-4o-mini", n=2, mock_response="mocked!")
        assert c.texts == ("mocked!", "mocked!")
        assert c.finish_reasons == ("stop", "stop")
        assert isinstance(c.usage, Usage) and c.usage.total_tokens > 0


class TestChatUnchanged:
    def test_chat_still_returns_first_text(self, fake_completion):
        fake_completion.return_value = _fake_response(["first", "second"])
        assert chat("hi", n=2) == "first"
