"""Tests for the central configuration module (aix.config)."""

import tempfile

import pytest

from aix import config
from aix.config import (
    AixConfig,
    ChatConfig,
    DEFAULT_ALIASES,
    load_config,
    get_config,
    set_config,
    configure,
    using,
    resolve_model,
)


def test_shipped_defaults():
    """A freshly built config exposes the shipped defaults."""
    c = AixConfig()
    assert c.chat.model == ChatConfig().model
    assert c.chat.temperature == 1.0
    assert c.embeddings.model == "text-embedding-3-small"
    assert c.image.model == "dall-e-3"
    assert c.audio.tts_model == "gpt-4o-mini-tts"
    assert c.audio.transcription_model == "whisper-1"
    assert c.aliases == DEFAULT_ALIASES


def test_env_layer_overrides_defaults():
    """Environment variables override shipped defaults, with casting."""
    environ = {"AIX_CHAT_MODEL": "env/model", "AIX_CHAT_TEMPERATURE": "0.25"}
    c = load_config(path="/does/not/exist.toml", environ=environ)
    assert c.chat.model == "env/model"
    assert c.chat.temperature == 0.25  # cast to float
    # untouched fields keep their defaults
    assert c.embeddings.model == "text-embedding-3-small"


def test_toml_layer_and_env_precedence():
    """TOML file fills values; env vars beat the TOML file."""
    toml = (
        b'[chat]\nmodel = "toml/model"\ntemperature = 0.9\n'
        b'[image]\nmodel = "toml/image"\n'
        b'[aliases]\nbest = "x/y"\n'
    )
    with tempfile.NamedTemporaryFile("wb", suffix=".toml", delete=False) as f:
        f.write(toml)
        path = f.name

    # env beats toml for chat.model; toml still provides image.model + aliases
    c = load_config(path=path, environ={"AIX_CHAT_MODEL": "env/model"})
    assert c.chat.model == "env/model"
    assert c.chat.temperature == 0.9  # from toml (no env override)
    assert c.image.model == "toml/image"
    # TOML aliases overlay the shipped defaults (override 'best', keep the rest)
    assert c.aliases == {**DEFAULT_ALIASES, "best": "x/y"}


def test_missing_toml_is_ignored():
    c = load_config(path="/no/such/file.toml", environ={})
    assert isinstance(c, AixConfig)
    assert c.chat.model == ChatConfig().model


@pytest.fixture
def restore_active_config():
    """Restore the active config after a test mutates it."""
    saved = get_config()
    yield
    set_config(saved)


def test_configure_flat_and_nested(restore_active_config):
    configure(chat_model="a/b")
    assert get_config().chat.model == "a/b"

    configure(chat={"temperature": 0.2})
    assert get_config().chat.temperature == 0.2
    # flat override above is preserved (immutable replace, not reset)
    assert get_config().chat.model == "a/b"


def test_configure_aliases(restore_active_config):
    configure(aliases={"fast": "a/b"})
    assert get_config().aliases["fast"] == "a/b"


def test_configure_rejects_unknown_key(restore_active_config):
    with pytest.raises(TypeError):
        configure(not_a_real_key=1)


def test_using_is_scoped(restore_active_config):
    before = get_config().chat.temperature
    with using(chat_temperature=0.0) as scoped:
        assert scoped.chat.temperature == 0.0
        assert get_config().chat.temperature == 0.0
    # restored on exit
    assert get_config().chat.temperature == before


def test_using_restores_on_exception(restore_active_config):
    before = get_config().chat.model
    with pytest.raises(ValueError):
        with using(chat_model="temp/model"):
            assert get_config().chat.model == "temp/model"
            raise ValueError("boom")
    assert get_config().chat.model == before


def test_config_is_immutable():
    c = AixConfig()
    with pytest.raises(Exception):
        c.chat.model = "nope"  # frozen dataclass


def test_set_config_type_checked(restore_active_config):
    with pytest.raises(TypeError):
        set_config({"chat": {"model": "x"}})  # not an AixConfig


def test_resolve_model_aliases(restore_active_config):
    # Shipped aliases resolve to concrete ids
    assert resolve_model("fast") == DEFAULT_ALIASES["fast"]
    assert resolve_model("best") == DEFAULT_ALIASES["best"]
    # Non-alias strings (literal model ids) pass through unchanged
    assert resolve_model("gpt-4o") == "gpt-4o"
    assert resolve_model("openrouter/anthropic/claude-3.5-sonnet") == (
        "openrouter/anthropic/claude-3.5-sonnet"
    )
    # None passes through
    assert resolve_model(None) is None


def test_resolve_model_custom_alias(restore_active_config):
    configure(aliases={"smart": "anthropic/claude-sonnet-4"})
    assert resolve_model("smart") == "anthropic/claude-sonnet-4"
    # configure merges: shipped aliases are preserved
    assert resolve_model("fast") == DEFAULT_ALIASES["fast"]


def test_resolve_model_chain_and_cycle(restore_active_config):
    # Chains resolve transitively
    configure(aliases={"a": "b", "b": "gpt-4o"})
    assert resolve_model("a") == "gpt-4o"
    # Cycles terminate instead of looping forever
    configure(aliases={"x": "y", "y": "x"})
    assert resolve_model("x") in {"x", "y"}


def test_alias_used_in_using_block(restore_active_config):
    with using(aliases={"fast": "custom/fast"}):
        assert resolve_model("fast") == "custom/fast"
    # restored to shipped default outside the block
    assert resolve_model("fast") == DEFAULT_ALIASES["fast"]


def test_active_config_drives_function_defaults(restore_active_config):
    """chat()/embeddings() read the active config for their model default."""
    import sys

    configure(chat_model="active/chat", embedding_model="active/embed")
    # The DFLT_* constants stay pinned to shipped defaults...
    chat_mod = sys.modules["aix.chat"]
    assert chat_mod.DFLT_CHAT_MODEL == ChatConfig().model
    # ...while the active config reflects the override used at call time.
    assert get_config().chat.model == "active/chat"
    assert get_config().embeddings.model == "active/embed"


def test_using_is_isolated_across_threads(restore_active_config):
    import threading

    both_inside = threading.Barrier(2, timeout=10)
    seen, errors = {}, []

    def worker(name, model):
        try:
            with using(chat_model=model):
                both_inside.wait()  # both threads are now inside their own block
                seen[name] = get_config().chat.model
                both_inside.wait()  # neither exits until both have read
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(n, m))
        for n, m in (("a", "x/a"), ("b", "x/b"))
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert seen == {"a": "x/a", "b": "x/b"}


def test_using_does_not_leak_into_other_thread(restore_active_config):
    import threading

    outside = get_config().chat.model
    seen = []
    with using(chat_model="scoped/model"):
        t = threading.Thread(target=lambda: seen.append(get_config().chat.model))
        t.start()
        t.join()
    assert seen == [outside]


def test_using_is_isolated_across_asyncio_tasks(restore_active_config):
    import asyncio

    async def task(model, delay):
        with using(chat_model=model):
            await asyncio.sleep(delay)
            return get_config().chat.model

    async def main():
        return await asyncio.gather(task("x/a", 0.02), task("x/b", 0.01))

    assert asyncio.run(main()) == ["x/a", "x/b"]


def test_configure_inside_using_is_scoped(restore_active_config):
    before = get_config().chat.model
    with using(chat_temperature=0.0):
        configure(chat_model="inner/model")
        assert get_config().chat.model == "inner/model"
        assert get_config().chat.temperature == 0.0
    assert get_config().chat.model == before


def test_nested_using_restores_outer(restore_active_config):
    with using(chat_model="outer/m"):
        with using(chat_temperature=0.5):
            assert get_config().chat.model == "outer/m"
        assert get_config().chat.temperature != 0.5
        assert get_config().chat.model == "outer/m"


def test_using_reaches_aix_worker_pools(restore_active_config):
    from unittest.mock import patch

    from aix.batches import batch_chat
    from aix.sampling import concurrent_map

    with using(chat_model="scoped/model"):
        seen = concurrent_map(
            lambda _: get_config().chat.model, range(3), concurrency=3
        )
        assert seen == ["scoped/model"] * 3

        with patch(
            "aix.batches.chat", side_effect=lambda *a, **k: get_config().chat.model
        ):
            out = list(batch_chat(["a", "b", "c"], max_workers=3))
        assert out == ["scoped/model"] * 3


def test_using_exited_from_another_thread_does_not_raise(restore_active_config):
    import threading

    def gen():
        with using(chat_model="gen/model"):
            yield

    g, errors = gen(), []
    next(g)

    def close():
        try:
            g.close()  # a different context than the one that entered the block
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=close)
    t.start()
    t.join()
    assert not errors
