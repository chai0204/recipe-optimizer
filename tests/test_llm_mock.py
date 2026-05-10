"""MockLLMClient unit tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from recipe_optimizer.llm import LLMOutputError, MockLLMClient


class _Sample(BaseModel):
    value: int
    name: str


class _Other(BaseModel):
    value: int
    name: str


def test_register_dict_returns_validated_model():
    client = MockLLMClient()
    client.register("parser", {"value": 42, "name": "hello"})

    result = client.generate_structured(
        prompt="ignored",
        output_schema=_Sample,
        label="parser",
    )

    assert isinstance(result, _Sample)
    assert result.value == 42
    assert result.name == "hello"


def test_register_basemodel_instance_returns_same():
    client = MockLLMClient()
    canned = _Sample(value=7, name="x")
    client.register("parser", canned)

    result = client.generate_structured(
        prompt="ignored",
        output_schema=_Sample,
        label="parser",
    )
    assert result == canned


def test_register_basemodel_of_different_class_is_coerced():
    """If a different BaseModel subclass is registered, it should be
    coerced through dict round-trip into the requested schema."""
    client = MockLLMClient()
    client.register("parser", _Other(value=3, name="abc"))

    result = client.generate_structured(
        prompt="ignored",
        output_schema=_Sample,
        label="parser",
    )
    assert isinstance(result, _Sample)
    assert result.value == 3
    assert result.name == "abc"


def test_register_callable_is_invoked_with_prompt():
    client = MockLLMClient()
    client.register("parser", lambda prompt: {"value": len(prompt), "name": prompt[:3]})

    result = client.generate_structured(
        prompt="hello",
        output_schema=_Sample,
        label="parser",
    )
    assert result.value == 5
    assert result.name == "hel"


def test_unregistered_label_raises():
    client = MockLLMClient()
    with pytest.raises(LLMOutputError, match="No mock handler"):
        client.generate_structured(
            prompt="x",
            output_schema=_Sample,
            label="missing",
        )


def test_invalid_dict_raises_output_error():
    client = MockLLMClient()
    client.register("parser", {"value": "not-an-int", "name": "x"})
    with pytest.raises(LLMOutputError, match="failed validation"):
        client.generate_structured(
            prompt="x",
            output_schema=_Sample,
            label="parser",
        )


def test_logs_capture_success_and_failure():
    client = MockLLMClient()
    client.register("ok", {"value": 1, "name": "a"})
    client.register("bad", {"value": "nope", "name": "a"})

    client.generate_structured(prompt="p1", output_schema=_Sample, label="ok")
    with pytest.raises(LLMOutputError):
        client.generate_structured(prompt="p2", output_schema=_Sample, label="bad")
    with pytest.raises(LLMOutputError):
        client.generate_structured(prompt="p3", output_schema=_Sample, label="missing")

    logs = client.call_logs
    assert len(logs) == 3
    assert [log.success for log in logs] == [True, False, False]
    assert [log.label for log in logs] == ["ok", "bad", "missing"]
    assert logs[0].parsed_output == {"value": 1, "name": "a"}
    assert logs[1].error is not None
    assert logs[2].attempts == 0  # never invoked the handler


def test_dump_logs_jsonl(tmp_path):
    client = MockLLMClient()
    client.register("parser", {"value": 1, "name": "a"})
    client.generate_structured(prompt="p", output_schema=_Sample, label="parser")

    out = tmp_path / "logs.jsonl"
    client.dump_logs_jsonl(out)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json

    record = json.loads(lines[0])
    assert record["label"] == "parser"
    assert record["success"] is True
    assert record["parsed_output"] == {"value": 1, "name": "a"}


def test_clear_logs():
    client = MockLLMClient()
    client.register("parser", {"value": 1, "name": "a"})
    client.generate_structured(prompt="p", output_schema=_Sample, label="parser")
    assert len(client.call_logs) == 1
    client.clear_logs()
    assert len(client.call_logs) == 0


def test_is_registered():
    client = MockLLMClient()
    assert not client.is_registered("parser")
    client.register("parser", {"value": 1, "name": "a"})
    assert client.is_registered("parser")
