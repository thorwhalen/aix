"""Quick access to AI tools goodies directly (without the need to remember the import path)"""

from aix.pd import *
from aix.np import *
from aix.sk import *

from aix import pd
from aix import np
from aix import sk

#
# from contextlib import suppress
#
# preferred_order = ['sk', 'np', 'pd']
#
# with suppress(ModuleNotFoundError):
#     from aix import sk
#
# with suppress(ModuleNotFoundError):
#     from aix import np
#
# with suppress(ModuleNotFoundError):
#     from aix import pd
#
# for _module_name in preferred_order[::-1]:
#     print(f"------ {_module_name}")
#     _module = __import__(f'aix.{_module_name}')
#     for _name in filter(lambda x: not x.startswith('__'), dir(_module)):
#         print(_name, _module)
#         locals()[_name] = getattr(_module, _name)
