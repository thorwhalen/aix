"""numpy"""

from guide.tools import submodule_callables

import numpy

for _ in list(dict.fromkeys(submodule_callables(numpy))):
    if hasattr(_, '__name__'):
        locals()[_.__name__] = _
