"""Google GenAI API functionality."""

from contextlib import suppress
from aix.credentials import resolve_api_key

name = "google"
required_packages = ["google.generativeai"]


class Const:
    GOOGLE_API_KEY = "GOOGLE_API_KEY"  # the reference (key) for the Google API key


chat_models = {
    "gemini-1.5-flash": {},  # TODO: Add metadata
    # Add more models as needed
}

with suppress(ModuleNotFoundError, ImportError):
    import google.generativeai as genai
    from i2 import Sig

    # Resolve via the unified credential path (env / .env / AIX config store).
    # Non-interactive at import time so importing the package never blocks.
    genai.configure(api_key=resolve_api_key("google"))

    # TODO: Add Literal for model
    _sig = Sig(genai.GenerativeModel.generate_content)
    _sig = _sig[2:]  # don't keep the first two (self, and contents (which is prompt))

    @Sig.replace_kwargs_using(_sig)
    def chat(prompt: str, *, model="gemini-1.5-flash", **kwargs):
        """Chat with Google's Gemini model."""
        response = genai.GenerativeModel(model).generate_content(prompt)
        return response.text
