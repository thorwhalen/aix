"""
Generative AI modules.
"""

import re
from collections import UserDict
from functools import partial

from aix.gen_ai import google_genai, openai_genai

genai_modules = [google_genai, openai_genai]

chat_models = UserDict()
chat_funcs = UserDict()

for _gen_ai_module in genai_modules:
    _name = getattr(_gen_ai_module, "name", None)
    _chat = getattr(_gen_ai_module, "chat", None)
    _chat_models = getattr(_gen_ai_module, "chat_models", None)

    # add provider to the _chat_models values
    for _model in _chat_models.keys():
        _chat_models[_model]["provider"] = _name

    if _chat_models:
        locals()[f"{_name}_chat_models"] = _chat_models
    else:
        print(f"Warning: {_name} module is missing chat or chat_models.")

    if _chat:
        locals()[f"{_name}_chat"] = _chat
        chat_models.update(_chat_models)
        chat_funcs.update(
            {model: partial(_chat, model=model) for model in _chat_models.keys()}
        )

for _model in chat_models.keys():
    # get a version of _model (str) that is a valid python identifier
    _model_identifier = re.compile(r"\W+").sub("_", _model)
    setattr(chat_models, _model_identifier, _model)
    setattr(chat_funcs, _model_identifier, chat_funcs[_model])


# for _model in chat_models.keys():
#     chat_functions[_model] = chat_function_for_model(_model)


def chat_function_for_model(model: str):
    """Return the chat function for the specified model."""
    for _gen_ai_module in genai_modules:
        if model in getattr(_gen_ai_module, "chat_models", {}):
            chat_func = getattr(_gen_ai_module, "chat", None)
            if chat_func is not None:
                return chat_func
            else:
                raise ValueError(
                    f"Chat function for model '{model}' requires missing packages: "
                    f"{getattr(_gen_ai_module, 'required_packages', [])}. "
                    f"Please install them and try again."
                )

    raise ValueError(
        f"Model '{model}' wasn't found. Available models "
        "(given what you have installed in your environment): {chat_models.keys()}"
    )


def chat(prompt: str, *, model: str, **kwargs):
    """Chat with the specified model."""
    chat_func = chat_function_for_model(model)
    return chat_func(prompt, model=model, **kwargs)
