"""sklearn (scikit-learn)"""
from contextlib import suppress
from guide.tools import submodule_callables

import sklearn

for _ in list(dict.fromkeys(submodule_callables(sklearn))):
    if hasattr(_, '__name__'):
        locals()[_.__name__] = _

for _subpackage_name in sklearn.__all__:
    with suppress(ModuleNotFoundError):
        _subpackage = __import__(f"sklearn.{_subpackage_name}")
        for _ in submodule_callables(sklearn):
            if hasattr(_, '__name__'):
                locals()[_.__name__] = _
