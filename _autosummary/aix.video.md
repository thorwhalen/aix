# aix.video

Video generation interface for AIX.

Generate videos from text descriptions or images.

#### NOTE
Video generation support varies by provider and may require additional
API keys and configuration. This module provides a unified interface but
implementation depends on available providers.

### Examples

Text to video:

```pycon
>>> from aix.video import generate_video
>>> video = generate_video(
...     "A cat walking through a garden",
...     duration=5
... )
>>> video.save("cat_video.mp4")
```

Image to video:

```pycon
>>> video = animate_image(
...     "static_image.jpg",
...     prompt="Make the clouds move gently"
... )
```

### Functions

| [`animate_image`](#aix.video.animate_image)(image_path[, prompt, model, ...])    | Animate a static image into a video.                        |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`estimate_cost`](#aix.video.estimate_cost)(duration[, resolution, provider])    | Estimate the cost of video generation.                      |
| [`extend_video`](#aix.video.extend_video)(video_path[, prompt, ...])            | Extend an existing video with additional generated content. |
| [`generate_video`](#aix.video.generate_video)(prompt, \*[, model, duration, ...]) | Generate a video from a text prompt.                        |
| [`generate_video_pika`](#aix.video.generate_video_pika)(prompt, \*[, duration])        | Generate video using Pika Labs.                             |
| [`generate_video_runway`](#aix.video.generate_video_runway)(prompt, \*[, duration])      | Generate video using Runway ML.                             |
| [`generate_video_stable_diffusion`](#aix.video.generate_video_stable_diffusion)(prompt, \*[, ...]) | Generate video using Stable Diffusion Video.                |
| [`get_available_providers`](#aix.video.get_available_providers)()                          | Get list of available video generation providers.           |
| [`interpolate_frames`](#aix.video.interpolate_frames)(video_path, \*[, ...])          | Interpolate frames to create smoother video.                |

### Classes

| [`GeneratedVideo`](#aix.video.GeneratedVideo)([url, data, model, prompt, ...])   | Wrapper for generated videos.   |
|----------------------------------------------------------------------------------------------------|---------------------------------|

### *class* aix.video.GeneratedVideo(url=None, data=None, model=None, prompt=None, duration=None, resolution=None, status='completed', task_id=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper for generated videos.

Provides convenient access to video data and metadata.

### Examples

```pycon
>>> video = GeneratedVideo(url="https://...")
>>> video.save("output.mp4")
>>> print(video.duration)
5.0
```

#### as_bytes()

Get video as bytes.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
* **Returns:**
  Video data as bytes

#### save(path)

Save video to file.

* **Parameters:**
  **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path

### Examples

```pycon
>>> video.save("output.mp4")
```

#### wait_until_complete(max_wait=300, poll_interval=5)

Wait for video generation to complete (for async operations).

* **Parameters:**
  * **max_wait** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum time to wait in seconds
  * **poll_interval** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Time between status checks in seconds
* **Raises:**
  * [**TimeoutError**](https://docs.python.org/3/builtins/exceptions.html#TimeoutError) – If generation doesn’t complete within max_wait
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – If generation fails

### aix.video.animate_image(image_path, prompt=None, , model=None, duration=3.0, motion_strength=0.5, \*\*kwargs)

Animate a static image into a video.

* **Parameters:**
  * **image_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the source image
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional text prompt to guide the animation
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video generation model to use
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Animation duration in seconds
  * **motion_strength** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Strength of motion (0.0 to 1.0)
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedVideo`](#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object

### Examples

```pycon
>>> from aix.video import animate_image
>>> video = animate_image(
...     "landscape.jpg",
...     prompt="Gentle camera pan across the scene",
...     duration=4
... )
>>> video.save("animated_landscape.mp4")
```

### aix.video.estimate_cost(duration, resolution='1280x720', provider=None)

Estimate the cost of video generation.

* **Parameters:**
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Video duration in seconds
  * **resolution** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video resolution
  * **provider** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Provider name (or None for all)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict with cost estimates

### Examples

```pycon
>>> from aix.video import estimate_cost
>>> cost = estimate_cost(duration=5, resolution="1920x1080")
>>> print(cost)
{'runway': 0.05, 'pika': 0.03}
```

### aix.video.extend_video(video_path, prompt=None, , extend_duration=2.0, model=None, \*\*kwargs)

Extend an existing video with additional generated content.

* **Parameters:**
  * **video_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the source video
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional text prompt to guide the extension
  * **extend_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How many seconds to add
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video generation model
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedVideo`](#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object with extended content

### Examples

```pycon
>>> from aix.video import extend_video
>>> extended = extend_video(
...     "original.mp4",
...     prompt="Continue the same scene",
...     extend_duration=3
... )
```

### aix.video.generate_video(prompt, , model=None, duration=5.0, resolution='1280x720', fps=24, aspect_ratio=None, style=None, seed=None, api_key=None, \*\*kwargs)

Generate a video from a text prompt.

#### NOTE
This is a high-level interface. Actual implementation depends on
available video generation providers (Runway, Pika, etc.) and may require
provider-specific API keys.

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description of the video to generate
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video generation model to use
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Video duration in seconds (typically 2-10 seconds)
  * **resolution** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video resolution (‘1280x720’, ‘1920x1080’, etc.)
  * **fps** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Frames per second
  * **aspect_ratio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Aspect ratio (‘16:9’, ‘9:16’, ‘1:1’, etc.)
  * **style** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video style hint (provider-specific)
  * **seed** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explicit API key for the video provider. If None, resolved from
    the environment / .env / AIX config store (see aix.credentials).
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedVideo`](#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object
* **Raises:**
  * [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) – If no video provider is configured
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If required provider SDK is not installed

### Examples

```pycon
>>> from aix.video import generate_video
>>> video = generate_video(
...     "A serene ocean sunset with gentle waves",
...     duration=5,
...     resolution="1920x1080"
... )
>>> video.save("sunset.mp4")
```

```pycon
>>> # Specific style
>>> video = generate_video(
...     "A futuristic city",
...     style="cyberpunk",
...     duration=4
... )
```

### aix.video.generate_video_pika(prompt, , duration=3.0, \*\*kwargs)

Generate video using Pika Labs.

Requires: PIKA_API_KEY environment variable

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Video duration
  * **\*\*kwargs** – Pika-specific parameters
* **Return type:**
  [`GeneratedVideo`](#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object

### aix.video.generate_video_runway(prompt, , duration=5.0, \*\*kwargs)

Generate video using Runway ML.

Requires: RUNWAY_API_KEY environment variable

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Video duration
  * **\*\*kwargs** – Runway-specific parameters
* **Return type:**
  [`GeneratedVideo`](#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object

### aix.video.generate_video_stable_diffusion(prompt, , duration=2.0, \*\*kwargs)

Generate video using Stable Diffusion Video.

Requires: Stable Diffusion Video model setup

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Video duration
  * **\*\*kwargs** – SD-specific parameters
* **Return type:**
  [`GeneratedVideo`](#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object

### aix.video.get_available_providers()

Get list of available video generation providers.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of provider names that are configured

### Examples

```pycon
>>> from aix.video import get_available_providers
>>> providers = get_available_providers()
>>> print(providers)
['runway', 'pika']
```

### aix.video.interpolate_frames(video_path, , target_fps=60, model=None, \*\*kwargs)

Interpolate frames to create smoother video.

* **Parameters:**
  * **video_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the source video
  * **target_fps** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target frames per second
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Frame interpolation model
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedVideo`](#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo with interpolated frames

### Examples

```pycon
>>> from aix.video import interpolate_frames
>>> smooth_video = interpolate_frames(
...     "choppy.mp4",
...     target_fps=60
... )
```
