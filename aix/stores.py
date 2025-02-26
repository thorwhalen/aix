"""Storage layers for AIX.

TODO: Finish this module
"""

import re
from contextlib import suppress

# Registry for extension-based "to string" decoders.
_extension_to_text_decoder = {}


def get_extension(string):
    m = re.search(r"\.([a-zA-Z0-9]+)$", string)
    return m.group(1).lower() if m else None


def add_extension_based_decoder(ext, decoder):
    """
    Register a decoder function for a given file extension.

    Args:
        ext (str): The file extension (without the leading dot), e.g., "pdf".
        decoder (Callable[[bytes], str]): A function that takes bytes and returns a decoded string.
    """
    _extension_to_text_decoder[ext.lower()] = decoder


def extension_based_decode_to_text(k, v):
    """
    Decode the given bytes `v` to a string based on the file extension extracted from `key`.

    If a decoder is registered for the extension, it is used.
    Otherwise, the bytes are decoded using v.decode('utf-8').

    Args:
        key (str): The key (typically a filename) from which to extract the extension.
        v (bytes): The value to decode.

    Returns:
        str: The decoded text.
    """
    # Extract the file extension (e.g., "txt", "pdf", etc.)
    ext = get_extension(k)

    if ext is not None and ext in _extension_to_text_decoder:
        return _extension_to_text_decoder[ext](v)
    else:
        # Fallback to default UTF-8 decoding
        return v.decode("utf-8")


# --- Register decoders for common text-based file extensions ---
# For plain text files, markdown files, and HTML, we can use the default decode.
add_extension_based_decoder("txt", lambda v: v.decode("utf-8"))
add_extension_based_decoder("md", lambda v: v.decode("utf-8"))


# # --- Optionally register a PDF decoder if the necessary package is available ---
# ignore_import_errors = suppress(ImportError, ModuleNotFoundError)
# with ignore_import_errors:
#     # Replace 'SOME_PACKAGE_THAT_HAS_A_TOOL_TO_CONVERT_PDF_TO_TEXT' with the actual package name.
#     from SOME_PACKAGE_THAT_HAS_A_TOOL_TO_CONVERT_PDF_TO_TEXT import pdf_bytes_to_string

#     add_extension_based_decoder('pdf', pdf_bytes_to_string)

# Example usage:
if __name__ == "__main__":
    # Example: decoding a plain text file (assuming the bytes represent UTF-8 text)
    key_txt = "example.txt"
    data_txt = b"Hello, world!"
    print(extension_based_decode_to_text(key_txt, data_txt))  # Output: Hello, world!

    # Example: decoding an unregistered extension will fall back to utf-8 decoding.
    key_unknown = "data.unknown"
    data_unknown = b"Fallback to UTF-8"
    print(extension_based_decode_to_text(key_unknown, data_unknown))
