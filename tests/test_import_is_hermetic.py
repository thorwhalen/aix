"""Importing ``aix`` must not touch the network.

``litellm`` fetches its model-cost map over HTTPS **as a side effect of being
imported**, so any eager ``import litellm`` inside ``aix`` makes every consumer
pay an unbounded outbound request just to ``import aix`` — a server cold start
blocks on it, an offline or restricted-egress deployment eats a connect
timeout, and a downstream test that only does ``importorskip("aix.audio")``
reaches the network.  See https://github.com/thorwhalen/aix/issues/38 and the
module docstring of :mod:`aix._litellm`.

Two independent guards, because either alone can go quietly false-green:

* :func:`test_import_opens_no_connection` runs the import in a **subprocess**
  under a socket guard.  A subprocess is required: by the time this module
  runs, pytest has long since imported ``aix``, so an in-process check could
  only observe an import that already happened.  The child's environment has
  ``LITELLM_LOCAL_MODEL_COST_MAP`` **removed** — with it set, litellm reads a
  bundled copy and never dials out, so the guard would pass even against a
  fully eager import.  (``tests/conftest.py`` sets it for the rest of the
  suite; this test must not inherit it.)
* :func:`test_import_does_not_import_litellm` asserts the mechanism directly.
  It is the guard that survives a machine where the network happens to be
  unreachable, or where some future litellm release stops fetching.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import aix

#: Modules a consumer plausibly imports directly.  ``aix`` itself is the
#: strongest case (its ``__init__`` pulls in every litellm-backed submodule);
#: ``aix.audio`` is the exact downstream reproducer from issue #38.
IMPORT_TARGETS = [
    "aix",
    "aix.audio",
    "aix.chat",
    "aix.embeddings",
    "aix.image",
    "aix.vision",
]

#: Child program: install a socket guard, import the target, report as JSON.
#:
#: ``connect``/``create_connection`` are recorded *and* blocked, so a
#: regression fails fast instead of hanging on an unreachable host.  DNS
#: lookups are recorded but allowed through: they are reported as context on
#: failure rather than asserted on, since a stray resolution elsewhere in the
#: stack would be a flaky reason to fail a build.  Blocking at the connect
#: layer already prevents the round trip either way.
_PROBE_SOURCE = r"""
import json, socket, sys

target = sys.argv[1]
connects, lookups = [], []

_real_getaddrinfo = socket.getaddrinfo


def _guard_getaddrinfo(host, port, *args, **kwargs):
    lookups.append(str(host))
    return _real_getaddrinfo(host, port, *args, **kwargs)


def _guard_connect(self, address, *args, **kwargs):
    connects.append(str(address))
    raise OSError("blocked by the aix import-hermeticity guard")


def _guard_create_connection(address, *args, **kwargs):
    connects.append(str(address))
    raise OSError("blocked by the aix import-hermeticity guard")


socket.getaddrinfo = _guard_getaddrinfo
socket.socket.connect = _guard_connect
socket.create_connection = _guard_create_connection

__import__(target)

try:
    import importlib.util

    litellm_installed = importlib.util.find_spec("litellm") is not None
except Exception:
    litellm_installed = False

print(
    "AIX_PROBE_RESULT "
    + json.dumps(
        {
            "connects": connects,
            "lookups": lookups,
            "litellm_imported": "litellm" in sys.modules,
            "litellm_installed": litellm_installed,
        }
    )
)
"""

_RESULT_MARKER = "AIX_PROBE_RESULT "


def _run_probe(source: str, *args: str) -> dict:
    """Run *source* in a child interpreter and return its JSON result.

    The child inherits this process's environment minus
    ``LITELLM_LOCAL_MODEL_COST_MAP`` (see the module docstring), and is pointed
    at the same ``aix`` this process imported, so the test can never
    accidentally probe a different installation.
    """
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    aix_parent = str(Path(aix.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [aix_parent, env.get("PYTHONPATH", "")])
    )

    proc = subprocess.run(
        [sys.executable, "-c", source, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"probe subprocess failed ({proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER) :])
    raise AssertionError(
        f"probe produced no result line\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


@pytest.fixture(scope="module")
def import_probe():
    """Probe each target once; both guards read the same run."""
    cache: dict = {}

    def probe(target: str) -> dict:
        if target not in cache:
            cache[target] = _run_probe(_PROBE_SOURCE, target)
        return cache[target]

    return probe


@pytest.mark.parametrize("target", IMPORT_TARGETS)
def test_import_opens_no_connection(target, import_probe):
    """``import <target>`` must open no outbound connection."""
    result = import_probe(target)

    if not result["litellm_installed"]:
        pytest.skip(
            "litellm is not installed, so this guard cannot distinguish a lazy "
            "import from an eager one"
        )

    assert result["connects"] == [], (
        f"`import {target}` opened {len(result['connects'])} outbound "
        f"connection(s): {result['connects']}. DNS lookups seen: "
        f"{result['lookups']}. Something on the import path reaches the "
        f"network -- most likely an eager `import litellm`, whose model-cost "
        f"map fetch runs at import time (see aix/_litellm.py)."
    )


@pytest.mark.parametrize("target", IMPORT_TARGETS)
def test_import_does_not_import_litellm(target, import_probe):
    """``import <target>`` must leave ``litellm`` out of ``sys.modules``.

    The structural half of the guard: it holds even offline, and even if a
    future litellm release drops the fetch.
    """
    result = import_probe(target)

    if not result["litellm_installed"]:
        pytest.skip("litellm is not installed; nothing to defer")

    assert not result["litellm_imported"], (
        f"`import {target}` imported litellm. litellm must be resolved on "
        f"first use instead (see the loaders in aix/_litellm.py and its "
        f"per-module callers), because importing it fetches a model-cost map "
        f"over HTTPS."
    )


def test_litellm_entry_points_resolve_on_first_use():
    """Deferring the import must not break it: every entry point still resolves.

    Runs in a child so the resolution (and its cost-map fetch) stays out of
    this process, and with the bundled cost map so it costs no network.
    """
    source = r"""
import json, os, sys

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "1"

import aix, aix.audio, aix.chat, aix.embeddings, aix.image, aix.vision

modules = sys.modules
before = "litellm" in modules
resolved = {
    "chat.completion": modules["aix.chat"]._completion(),
    "vision.completion": modules["aix.vision"]._completion(),
    "embeddings.embedding": modules["aix.embeddings"]._embedding(),
    "image.image_generation": modules["aix.image"]._image_generation(),
    "audio.speech": modules["aix.audio"]._speech(),
    "audio.transcription": modules["aix.audio"]._transcription(),
}
print("AIX_PROBE_RESULT " + json.dumps({
    "imported_before_use": before,
    "unresolved": sorted(k for k, v in resolved.items() if not callable(v)),
    "imported_after_use": "litellm" in modules,
    "cached": modules["aix.chat"]._completion() is resolved["chat.completion"],
}))
"""
    result = _run_probe(source)

    assert result["imported_before_use"] is False
    assert result["unresolved"] == [], (
        f"these litellm entry points did not resolve on first use: "
        f"{result['unresolved']}"
    )
    assert result["imported_after_use"] is True
    assert result["cached"], "a resolved entry point must be cached, not re-imported"
