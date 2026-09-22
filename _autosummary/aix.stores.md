# aix.stores

Storage layers for AIX.

### Functions

| [`add_extension_based_decoder`](#aix.stores.add_extension_based_decoder)(ext, decoder)   | Register a decoder function for a given file extension.                                  |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [`extension_based_decode_to_text`](#aix.stores.extension_based_decode_to_text)(k, v)        | Decode the given bytes `v` to a string based on the file extension extracted from `key`. |
| `get_extension`(string)                                                                      |                                                                                          |

### aix.stores.add_extension_based_decoder(ext, decoder)

Register a decoder function for a given file extension.

* **Parameters:**
  * **ext** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The file extension (without the leading dot), e.g., “pdf”.
  * **decoder** (*Callable* *[* *[*[*bytes*](https://docs.python.org/3/builtins/stdtypes.html#bytes) *]* *,* [*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *]*) – A function that takes bytes and returns a decoded string.

### aix.stores.extension_based_decode_to_text(k, v)

Decode the given bytes `v` to a string based on the file extension extracted from `key`.

If a decoder is registered for the extension, it is used.
Otherwise, the bytes are decoded using v.decode(‘utf-8’).

* **Parameters:**
  * **key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The key (typically a filename) from which to extract the extension.
  * **v** ([*bytes*](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The value to decode.
* **Returns:**
  The decoded text.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)
