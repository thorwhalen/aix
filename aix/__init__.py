"""
Facade to key AI tools.

Get a list of available chat functions (this will depend on the AI packages
you have installed locally):

>>> from aix import chat_funcs
>>> list(chat_funcs)  # doctest: +SKIP
['gemini-1.5-flash',
 'gpt-4',
 'gpt-4-32k',
 'gpt-4-turbo',
 'gpt-3.5-turbo',
 'o1-preview',
 'o1-mini',
 'gpt-4o',
 'gpt-4o-mini']

Choose a chat function and chat with it:
>>> google_ai_chat = chat_funcs['gemini-1.5-flash']  # doctest: +SKIP
>>> google_ai_chat("What is the meaning of life? Respond with a number.")  # doctest: +SKIP
'42'
>>> openai_chat = chat_funcs['gpt-3.5-turbo']  # doctest: +SKIP
>>> openai_chat("What is the meaning of life? Respond with a number.")  # doctest: +SKIP
'42'

"""

from aix.gen_ai import chat, chat_models, chat_funcs

# TODO: Change this so that there's a load_pkg function that loads the packages dynamically
#   if and when use wants.

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
