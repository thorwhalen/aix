"""Root pytest configuration — keeps every pytest invocation off the network.

Importing ``litellm`` fetches its model-cost map over HTTPS.  ``import aix`` no
longer triggers that (see :mod:`aix._litellm`), but anything that reaches the
*first call* still does — including a doctest, and including
:func:`aix._litellm.litellm_attr`'s own example.

This lives at the repo root rather than in ``tests/`` because a ``conftest.py``
only applies to what is collected beneath it, and CI collects **two** trees:
``pytest tests/`` and ``pytest --doctest-modules aix/`` (the wads ``run-tests``
action adds ``--doctest-modules`` by default).  A ``tests/conftest.py`` would
leave the doctest run reaching out.

``setdefault`` leaves an explicitly-set value alone; litellm treats any
non-empty value as "use the bundled map".  Deliberately *not* done inside the
library itself: ``os.environ`` is process-global, so that choice belongs to the
operator, not to ``aix`` (see :mod:`aix._litellm`).
``tests/test_import_is_hermetic.py`` strips this from its subprocesses — with
it set, its socket guard could not fail.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "1")
