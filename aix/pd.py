"""pandas"""

from guide.tools import submodule_callables

import pandas

for _ in list(dict.fromkeys(submodule_callables(pandas))):
    if hasattr(_, '__name__'):
        locals()[_.__name__] = _