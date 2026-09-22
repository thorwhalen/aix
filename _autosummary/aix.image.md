# aix.image

Image generation interface for AIX.

Generate images from text descriptions using various AI models.

### Examples

Simple image generation:

```pycon
>>> from aix.image import generate_image
>>> image = generate_image("A serene mountain landscape at sunset")
>>> image.save("landscape.png")
```

Multiple images:

```pycon
>>> images = generate_images(
...     "A cute cat wearing a hat",
...     n=3,
...     size="1024x1024"
... )
>>> for i, img in enumerate(images):
...     img.save(f"cat_{i}.png")
```

With specific model:

```pycon
>>> image = generate_image(
...     "Abstract art with vibrant colors",
...     model="dall-e-3"
... )
```

### Functions

| [`create_variation`](#aix.image.create_variation)(image_path, \*[, model, ...])     | Create variations of an existing image.      |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------|
| [`edit_image`](#aix.image.edit_image)(image_path, prompt, \*[, ...])          | Edit an existing image based on a prompt.    |
| [`generate_image`](#aix.image.generate_image)(prompt, \*[, model, size, ...])     | Generate a single image from a text prompt.  |
| [`generate_images`](#aix.image.generate_images)(prompt, \*[, n, model, size, ...]) | Generate multiple images from a text prompt. |

### Classes

| [`GeneratedImage`](#aix.image.GeneratedImage)([url, b64_json, model, ...])   | Wrapper for generated images.   |
|------------------------------------------------------------------------------------------------|---------------------------------|

### *class* aix.image.GeneratedImage(url=None, b64_json=None, model=None, prompt=None, revised_prompt=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper for generated images.

Provides convenient access to image data in various formats.

### Examples

```pycon
>>> img = GeneratedImage(url="https://...", model="dall-e-3")
>>> img.save("output.png")
>>> img.show()
>>> data = img.as_bytes()
```

#### as_bytes()

Get image as bytes.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
* **Returns:**
  Image data as bytes

#### as_pil_image()

Get image as PIL Image object.

* **Returns:**
  PIL.Image object
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If PIL is not installed

#### save(path, format=None)

Save image to file.

* **Parameters:**
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image format (e.g., ‘PNG’, ‘JPEG’). Auto-detected from path if None.

### Examples

```pycon
>>> img.save("output.png")
>>> img.save("output.jpg", format="JPEG")
```

#### show()

Display the image.

Requires PIL (Pillow) to be installed.

### Examples

```pycon
>>> img.show()
```

### aix.image.create_variation(image_path, , model=None, size=None, n=1, api_key=None, \*\*kwargs)

Create variations of an existing image.

* **Parameters:**
  * **image_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the source image
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use (typically ‘dall-e-2’)
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output image size
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of variations to generate
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  `Union`[[`GeneratedImage`](#aix.image.GeneratedImage), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`GeneratedImage`](#aix.image.GeneratedImage)]]
* **Returns:**
  GeneratedImage or list of GeneratedImage objects

### Examples

```pycon
>>> from aix.image import create_variation
>>> variations = create_variation(
...     "original.png",
...     n=3,
...     size="512x512"
... )
>>> for i, var in enumerate(variations):
...     var.save(f"variation_{i}.png")
```

### aix.image.edit_image(image_path, prompt, , mask_path=None, model=None, size=None, n=1, api_key=None, \*\*kwargs)

Edit an existing image based on a prompt.

* **Parameters:**
  * **image_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the image to edit
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Description of the desired edit
  * **mask_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Optional path to mask image (transparent areas will be edited)
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use (typically ‘dall-e-2’ for edits)
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output image size
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of variations to generate
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  `Union`[[`GeneratedImage`](#aix.image.GeneratedImage), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`GeneratedImage`](#aix.image.GeneratedImage)]]
* **Returns:**
  GeneratedImage or list of GeneratedImage objects

### Examples

```pycon
>>> from aix.image import edit_image
>>> edited = edit_image(
...     "photo.png",
...     "Add a rainbow in the sky",
...     mask_path="sky_mask.png"
... )
>>> edited.save("edited_photo.png")
```

### aix.image.generate_image(prompt, , model=None, size=None, quality=None, style=None, response_format='url', api_key=None, \*\*kwargs)

Generate a single image from a text prompt.

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description of the image to generate
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use (e.g., ‘dall-e-2’, ‘dall-e-3’, ‘stable-diffusion’)
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image size (e.g., ‘1024x1024’, ‘512x512’, ‘1792x1024’)
  * **quality** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image quality (‘standard’ or ‘hd’ for DALL-E 3)
  * **style** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image style (‘vivid’ or ‘natural’ for DALL-E 3)
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format of response (‘url’ or ‘b64_json’)
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explicit API key. If None, resolved from the environment / .env
    / AIX config store for the model’s provider (see aix.credentials).
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedImage`](#aix.image.GeneratedImage)
* **Returns:**
  GeneratedImage object
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed

### Examples

```pycon
>>> from aix.image import generate_image
>>> image = generate_image("A serene mountain landscape")
>>> image.save("landscape.png")
```

```pycon
>>> # High quality with DALL-E 3
>>> image = generate_image(
...     "Abstract art with vibrant colors",
...     model="dall-e-3",
...     quality="hd",
...     style="vivid"
... )
```

```pycon
>>> # Specific size
>>> image = generate_image(
...     "A futuristic city",
...     size="1792x1024"
... )
```

### aix.image.generate_images(prompt, , n=None, model=None, size=None, quality=None, style=None, response_format='url', api_key=None, \*\*kwargs)

Generate multiple images from a text prompt.

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description of images to generate
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of images to generate
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image size
  * **quality** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image quality
  * **style** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image style
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format of response
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`GeneratedImage`](#aix.image.GeneratedImage)]
* **Returns:**
  List of GeneratedImage objects

### Examples

```pycon
>>> from aix.image import generate_images
>>> images = generate_images(
...     "A cute robot",
...     n=3,
...     size="512x512"
... )
>>> for i, img in enumerate(images):
...     img.save(f"robot_{i}.png")
```
