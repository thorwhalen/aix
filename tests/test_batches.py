"""Tests for aix.batches module."""

import pytest
from unittest.mock import patch

from aix.batches import batch_chat, batch_process, BatchError


def _flaky_chat(prompt, model=None, **kwargs):
    """Answer every prompt except 'b', which raises a rate-limit error."""
    if prompt == "b":
        raise RuntimeError("rate limit exceeded")
    return "real answer"


class TestBatchChatErrorSlots:
    """Tests for how batch_chat reports a failed prompt."""

    @patch("aix.batches.chat", side_effect=_flaky_chat)
    def test_error_slot_is_typed(self, mock_chat):
        """A failed slot carries the real exception, not just its text."""
        results = list(batch_chat(["a", "b", "c"]))

        assert isinstance(results[1], BatchError)
        assert isinstance(results[1].exception, RuntimeError)
        assert results[1].index == 1

    @patch("aix.batches.chat", side_effect=_flaky_chat)
    def test_error_slot_is_still_a_plain_string(self, mock_chat):
        """Backcompat pin: the slot is byte-for-byte what it always was."""
        results = list(batch_chat(["a", "b", "c"]))

        assert len(results) == 3
        assert isinstance(results[1], str)
        assert results[1] == "ERROR: rate limit exceeded"
        assert results[0] == "real answer"
        assert results[2] == "real answer"

    @patch("aix.batches.chat", side_effect=_flaky_chat)
    def test_on_error_raise(self, mock_chat):
        """on_error='raise' surfaces the underlying exception."""
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            list(batch_chat(["a", "b", "c"], on_error="raise"))

    @patch("aix.batches.chat", side_effect=_flaky_chat)
    def test_on_error_skip(self, mock_chat):
        """on_error='skip' omits the failed slots from the stream."""
        assert list(batch_chat(["a", "b", "c"], on_error="skip")) == [
            "real answer",
            "real answer",
        ]

    @patch("aix.batches.chat", side_effect=_flaky_chat)
    def test_unknown_on_error_rejected(self, mock_chat):
        """An unsupported on_error value is an error, not a silent no-op."""
        with pytest.raises(ValueError, match="on_error"):
            list(batch_chat(["a"], on_error="ignore"))

    @patch("aix.batches.chat", side_effect=_flaky_chat)
    def test_default_is_return(self, mock_chat):
        """The default keeps every slot, in order, with no exception raised."""
        results = list(batch_chat(["a", "b", "c"], on_error="return"))
        assert len(results) == 3
        assert results[1] == "ERROR: rate limit exceeded"


class TestBatchProcessErrorSlots:
    """Same guarantees for the generic batch_process."""

    @staticmethod
    def _flaky(item):
        if item == "b":
            raise RuntimeError("rate limit exceeded")
        return "real answer"

    def test_error_slot_is_typed(self):
        """A failed slot carries the real exception, not just its text."""
        results = list(batch_process(["a", "b", "c"], self._flaky, retry_attempts=1))

        assert isinstance(results[1], BatchError)
        assert isinstance(results[1].exception, RuntimeError)
        assert results[1].index == 1

    def test_error_slot_is_still_a_plain_string(self):
        """Backcompat pin: the slot is byte-for-byte what it always was."""
        results = list(batch_process(["a", "b", "c"], self._flaky, retry_attempts=1))

        assert len(results) == 3
        assert isinstance(results[1], str)
        assert results[1] == "ERROR: rate limit exceeded"

    def test_on_error_raise(self):
        """on_error='raise' surfaces the underlying exception."""
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            list(
                batch_process(
                    ["a", "b", "c"], self._flaky, retry_attempts=1, on_error="raise"
                )
            )

    def test_on_error_skip(self):
        """on_error='skip' omits the failed slots from the stream."""
        results = list(
            batch_process(
                ["a", "b", "c"], self._flaky, retry_attempts=1, on_error="skip"
            )
        )
        assert results == ["real answer", "real answer"]


class TestBatchErrorType:
    """BatchError must behave as the string it replaces."""

    def test_is_a_str_subclass(self):
        """Every string use of the old value keeps working."""
        err = BatchError(RuntimeError("boom"), index=2)

        assert isinstance(err, str)
        assert err == "ERROR: boom"
        assert err.startswith("ERROR:")
        assert err[7:] == "boom"

    def test_json_serialisable(self):
        """A results list can still be dumped to JSON."""
        import json

        assert json.dumps([BatchError(RuntimeError("boom"))]) == '["ERROR: boom"]'

    def test_survives_a_pickle_round_trip(self):
        """A str subclass with a custom __new__ must not corrupt on unpickle."""
        import pickle

        err = BatchError(RuntimeError("boom"), index=2)
        revived = pickle.loads(pickle.dumps(err))

        assert revived == "ERROR: boom"
        assert isinstance(revived, BatchError)
        assert revived.index == 2

    def test_exported_from_package_root(self):
        """BatchError is part of the public surface."""
        import aix

        assert aix.BatchError is BatchError
        assert "BatchError" in aix.__all__
