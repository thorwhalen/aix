"""`import aix` must not import the optional `oa` package (issue #54).

Importing `oa` resolves an OpenAI key as a side effect: it raises without one,
and prompts on stdin in an interactive session. So `import aix` must not touch
it; the legacy backend imports it on first call.
"""

import subprocess
import sys
import textwrap

FAKE_OA = textwrap.dedent(
    """
    raise RuntimeError("fake oa: imported at import time")
    """
)


def _run(tmp_path, code):
    pkg = tmp_path / "oa"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(FAKE_OA)
    env = {
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{tmp_path}",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_aix_does_not_import_oa(tmp_path):
    result = _run(
        tmp_path,
        "import sys, aix; assert 'oa' not in sys.modules; "
        "assert 'gpt-4o' in aix.chat_models; print('ok')",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


def test_oa_imported_on_first_legacy_call(tmp_path):
    result = _run(
        tmp_path,
        "import aix.gen_ai as g\n"
        "try:\n"
        "    g.chat('hi', model='gpt-4o')\n"
        "except RuntimeError as e:\n"
        "    print('lazy:', e)\n",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "fake oa: imported at import time" in result.stdout


def test_passthrough_arguments(monkeypatch):
    """The lazy wrapper forwards every argument to oa.chat unchanged."""
    import types

    from aix.gen_ai import openai_genai

    if not hasattr(openai_genai, "chat"):
        import pytest

        pytest.skip("oa not installed")
    seen = {}
    fake = types.ModuleType("oa")
    fake.chat = lambda *a, **k: seen.update(args=a, kwargs=k) or "ok"
    monkeypatch.setitem(__import__("sys").modules, "oa", fake)
    assert openai_genai.chat(messages=[{"role": "user", "content": "x"}]) == "ok"
    assert seen == {
        "args": (),
        "kwargs": {"messages": [{"role": "user", "content": "x"}]},
    }
