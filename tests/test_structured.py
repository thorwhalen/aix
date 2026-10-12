"""Tests for provider-enforced, validated structured output (issue #43)."""

import json
import sys
from types import SimpleNamespace
from typing import Literal, Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import aix
import aix.structured
from aix.chat import Usage, complete
from aix.sampling import sample
from aix.structured import ResponseSchema, SchemaValidationError, _is_strict_compatible

_aix_chat = sys.modules["aix.chat"]


class Span(BaseModel):
    start: int
    end: int


class Judgement(BaseModel):
    value: Literal["yes", "no", "unsure"]
    confidence: float
    evidence: list[Span]
    note: Optional[str] = None


NESTED_DICT_SCHEMA = {
    "title": "judgement",
    "type": "object",
    "properties": {
        "value": {"enum": ["yes", "no"]},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"start": {"type": "integer"}},
                "required": ["start"],
            },
        },
    },
    "required": ["value"],
}

GOOD = json.dumps(
    {"value": "yes", "confidence": 0.9, "evidence": [{"start": 1, "end": 4}]}
)


def _response(*texts, usage=None):
    choices = [
        SimpleNamespace(message=SimpleNamespace(content=t), finish_reason="stop")
        for t in texts
    ]
    return SimpleNamespace(choices=choices, usage=usage, model="m")


def _usage(total=5):
    return {"prompt_tokens": total - 1, "completion_tokens": 1, "total_tokens": total}


@pytest.fixture
def schema_support():
    """Set what the (mocked) litellm supports_response_schema reports."""
    state = {"value": True}
    with patch.object(
        aix.structured,
        "_litellm_supports_response_schema",
        lambda model: state["value"],
    ):
        yield state


@pytest.fixture
def fake_completion():
    with patch.object(_aix_chat, "_litellm_completion") as mock:
        yield mock


class TestResponseSchema:
    def test_pydantic_returns_instance_and_is_strict(self):
        spec = ResponseSchema.from_schema(Judgement)
        assert spec.strict_compatible
        out = spec.validate(GOOD)
        assert isinstance(out, Judgement) and out.evidence[0].end == 4

    def test_pydantic_rejects_bad_enum_and_nested(self):
        spec = ResponseSchema.from_schema(Judgement)
        with pytest.raises(SchemaValidationError):
            spec.validate(GOOD.replace('"yes"', '"maybe"'))
        with pytest.raises(SchemaValidationError):
            spec.validate(GOOD.replace('"end": 4', '"end": "four"'))

    def test_dict_schema_nested_enum_required(self):
        spec = ResponseSchema.from_schema(NESTED_DICT_SCHEMA)
        assert spec.name == "judgement" and not spec.strict_compatible
        assert (
            spec.validate('{"value": "no", "evidence": [{"start": 2}]}')["value"]
            == "no"
        )
        for bad in [
            '{"value": "maybe"}',
            '{"evidence": []}',
            '{"value": "no", "evidence": [{}]}',
        ]:
            with pytest.raises(SchemaValidationError) as info:
                spec.validate(bad)
            assert info.value.text == bad and info.value.detail

    def test_not_json(self):
        with pytest.raises(SchemaValidationError, match="not valid JSON"):
            ResponseSchema.from_schema(Judgement).validate("Sure! Here you go")

    def test_code_fence_tolerated(self):
        spec = ResponseSchema.from_schema(NESTED_DICT_SCHEMA)
        assert spec.validate('```json\n{"value": "yes"}\n```') == {"value": "yes"}

    def test_malformed_schema_fails_at_definition(self):
        import jsonschema

        with pytest.raises(jsonschema.SchemaError):
            ResponseSchema.from_schema({"type": "not-a-type"})

    def test_bad_schema_type(self):
        with pytest.raises(TypeError):
            ResponseSchema.from_schema(int)

    def test_strict_compatibility(self):
        closed = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        assert _is_strict_compatible(closed)
        assert not _is_strict_compatible({**closed, "required": []})
        nested_open = {
            **closed,
            "properties": {"a": {"type": "object", "properties": {}}},
        }
        assert not _is_strict_compatible(nested_open)

    def test_schema_name_sanitized(self):
        spec = ResponseSchema.from_schema({"title": "my schema!", "type": "object"})
        assert spec.name == "my_schema_"


class TestCompleteRouting:
    def test_strict_mode(self, fake_completion, schema_support):
        fake_completion.return_value = _response(GOOD)
        c = complete("q", response_schema=Judgement)
        fmt = fake_completion.call_args.kwargs["response_format"]
        assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
        assert c.schema_mode == "strict" and isinstance(c.value, Judgement)
        assert c.errors == (None,) and c.repairs == 0

    def test_non_strict_schema_mode(self, fake_completion, schema_support):
        fake_completion.return_value = _response('{"value": "yes"}')
        c = complete("q", response_schema=NESTED_DICT_SCHEMA)
        fmt = fake_completion.call_args.kwargs["response_format"]
        assert fmt["json_schema"]["strict"] is False
        assert fmt["json_schema"]["schema"] == NESTED_DICT_SCHEMA
        assert c.schema_mode == "schema" and c.value == {"value": "yes"}

    def test_json_mode_fallback_adds_instruction(self, fake_completion, schema_support):
        schema_support["value"] = False
        fake_completion.return_value = _response(GOOD)
        messages = [{"role": "user", "content": "q"}]
        c = complete(messages, response_schema=Judgement)
        sent = fake_completion.call_args.kwargs
        assert sent["response_format"] == {"type": "json_object"}
        assert "JSON Schema" in sent["messages"][-1]["content"]
        assert sent["messages"][-1]["content"].startswith("q")
        assert messages == [{"role": "user", "content": "q"}]  # caller's not mutated
        assert c.schema_mode == "json_mode" and isinstance(c.value, Judgement)

    def test_json_mode_appends_turn_after_non_user_message(
        self, fake_completion, schema_support
    ):
        schema_support["value"] = False
        fake_completion.return_value = _response(GOOD)
        complete([{"role": "system", "content": "s"}], response_schema=Judgement)
        last = fake_completion.call_args.kwargs["messages"][-1]
        assert last["role"] == "user" and "JSON Schema" in last["content"]

    def test_without_schema_unchanged(self, fake_completion):
        fake_completion.return_value = _response("x")
        c = complete("q")
        assert c.schema_mode is None and c.values == ()
        with pytest.raises(ValueError, match="No structured value"):
            c.value

    def test_schema_and_format_exclusive(self, fake_completion):
        with pytest.raises(ValueError, match="not both"):
            complete(
                "q", response_schema=Judgement, response_format={"type": "json_object"}
            )
        fake_completion.assert_not_called()

    @pytest.mark.parametrize("bad", [-1, 1.5, True])
    def test_bad_max_repairs(self, fake_completion, bad):
        with pytest.raises(ValueError, match="max_repairs"):
            complete("q", response_schema=Judgement, max_repairs=bad)


class TestRepair:
    def test_invalid_then_repaired(self, fake_completion, schema_support):
        fake_completion.side_effect = [
            _response("not json", usage=_usage(5)),
            _response(GOOD, usage=_usage(7)),
        ]
        c = complete("q", response_schema=Judgement, max_repairs=1)
        assert isinstance(c.value, Judgement)
        assert c.repairs == 1 and c.texts == (GOOD,)
        assert c.usage == Usage(10, 2, 12)
        repair_messages = fake_completion.call_args.kwargs["messages"]
        assert repair_messages[-2] == {"role": "assistant", "content": "not json"}
        assert "invalid" in repair_messages[-1]["content"]

    def test_repairs_exhausted_records_error(self, fake_completion, schema_support):
        fake_completion.return_value = _response("nope")
        c = complete("q", response_schema=Judgement, max_repairs=2)
        assert c.repairs == 2 and fake_completion.call_count == 3
        assert c.values == (None,)
        assert isinstance(c.errors[0], SchemaValidationError)
        with pytest.raises(SchemaValidationError):
            c.value

    def test_zero_repairs(self, fake_completion, schema_support):
        fake_completion.return_value = _response("nope")
        c = complete("q", response_schema=Judgement, max_repairs=0)
        assert fake_completion.call_count == 1 and c.errors[0] is not None

    def test_default_repairs_from_config(self, fake_completion, schema_support):
        fake_completion.return_value = _response("nope")
        with aix.using(chat_schema_max_repairs=3):
            complete("q", response_schema=Judgement)
        assert fake_completion.call_count == 4

    def test_only_invalid_choice_repaired(self, fake_completion, schema_support):
        fake_completion.side_effect = [
            _response(GOOD, "bad"),
            _response(GOOD),
        ]
        c = complete("q", n=2, response_schema=Judgement, max_repairs=1)
        assert c.repairs == 1 and c.errors == (None, None)
        assert "n" not in fake_completion.call_args.kwargs


class TestSampleWithSchema:
    def test_payloads_are_validated_values(self, fake_completion, schema_support):
        with patch(
            "aix.capabilities._litellm_get_supported_openai_params", lambda model: ["n"]
        ):
            fake_completion.side_effect = [
                _response(GOOD, "bad"),
                _response("still bad"),
            ]
            out = sample("q", n=2, response_schema=Judgement, max_repairs=1)
        assert isinstance(out[0].payload, Judgement) and out[0].ok
        assert out[1].payload is None and isinstance(
            out[1].error, SchemaValidationError
        )

    def test_parse_and_schema_exclusive(self, fake_completion):
        with pytest.raises(ValueError, match="not both"):
            sample("q", n=2, parse=json.loads, response_schema=Judgement)


class TestPromptFunc:
    def test_response_schema_returns_value(self, fake_completion, schema_support):
        fake_completion.return_value = _response(GOOD)
        judge = aix.prompt_func("Judge {claim}", response_schema=Judgement)
        out = judge(claim="the sky is blue")
        assert isinstance(out, Judgement)
        assert fake_completion.call_args.kwargs["messages"][0]["content"] == (
            "Judge the sky is blue"
        )
        assert judge.response_schema is Judgement

    def test_invalid_raises(self, fake_completion, schema_support):
        fake_completion.return_value = _response("nope")
        judge = aix.prompt_func("Judge {claim}", response_schema=Judgement)
        with pytest.raises(SchemaValidationError):
            judge(claim="x")

    def test_both_schemas_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            aix.prompt_func(
                "x {y}", output_schema={"a": str}, response_schema=Judgement
            )


def test_end_to_end_through_litellm_mock_response():
    """Real litellm (no network): real support lookup + mock_response."""
    pytest.importorskip("litellm")
    c = complete(
        "q", model="gpt-4o-mini", response_schema=Judgement, mock_response=GOOD
    )
    assert c.schema_mode == "strict"
    assert c.value == Judgement.model_validate_json(GOOD)


def test_exports():
    assert aix.ResponseSchema is ResponseSchema
    assert aix.SchemaValidationError is SchemaValidationError
