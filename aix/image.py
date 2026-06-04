"""Image generation interface for AIX.

Generate images from text descriptions using various AI models.

Examples:
    Simple image generation:
    >>> from aix.image import generate_image
    >>> image = generate_image("A serene mountain landscape at sunset")  # doctest: +SKIP
    >>> image.save("landscape.png")  # doctest: +SKIP

    Multiple images:
    >>> images = generate_images(
    ...     "A cute cat wearing a hat",
    ...     n=3,
    ...     size="1024x1024"
    ... )  # doctest: +SKIP
    >>> for i, img in enumerate(images):  # doctest: +SKIP
    ...     img.save(f"cat_{i}.png")

    With specific model:
    >>> image = generate_image(
    ...     "Abstract art with vibrant colors",
    ...     model="dall-e-3"
    ... )  # doctest: +SKIP
"""

from collections.abc import Iterable
from typing import Union, Any
from pathlib import Path
import base64
from io import BytesIO

# Import LiteLLM but keep it private
try:
    from litellm import image_generation as _litellm_image_generation
except ImportError:
    _litellm_image_generation = None

# Try to import PIL for image handling
try:
    from PIL import Image as PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    PILImage = None


# Shipped-default constants, kept for backward compatibility. The *active*
# defaults are resolved from ``aix.config`` at call time (see aix/config.py).
from aix.config import get_config as _get_config, ImageConfig as _ImageConfig

DFLT_IMAGE_MODEL = _ImageConfig().model
DFLT_IMAGE_SIZE = _ImageConfig().size
DFLT_IMAGE_QUALITY = _ImageConfig().quality
DFLT_NUM_IMAGES = _ImageConfig().num_images


class GeneratedImage:
    """Wrapper for generated images.

    Provides convenient access to image data in various formats.

    Examples:
        >>> img = GeneratedImage(url="https://...", model="dall-e-3")  # doctest: +SKIP
        >>> img.save("output.png")  # doctest: +SKIP
        >>> img.show()  # doctest: +SKIP
        >>> data = img.as_bytes()  # doctest: +SKIP
    """

    def __init__(
        self,
        url: str = None,
        b64_json: str = None,
        model: str = None,
        prompt: str = None,
        revised_prompt: str = None,
    ):
        """Initialize generated image.

        Args:
            url: URL of the generated image
            b64_json: Base64-encoded image data
            model: Model used for generation
            prompt: Original prompt
            revised_prompt: Revised prompt (if model modified it)
        """
        self.url = url
        self.b64_json = b64_json
        self.model = model
        self.prompt = prompt
        self.revised_prompt = revised_prompt
        self._image_data = None
        self._pil_image = None

    def as_bytes(self) -> bytes:
        """Get image as bytes.

        Returns:
            Image data as bytes
        """
        if self._image_data is not None:
            return self._image_data

        if self.b64_json:
            self._image_data = base64.b64decode(self.b64_json)
            return self._image_data

        if self.url:
            import requests

            response = requests.get(self.url)
            response.raise_for_status()
            self._image_data = response.content
            return self._image_data

        raise ValueError("No image data available")

    def as_pil_image(self):
        """Get image as PIL Image object.

        Returns:
            PIL.Image object

        Raises:
            ImportError: If PIL is not installed
        """
        if not _PIL_AVAILABLE:
            raise ImportError(
                "PIL (Pillow) is required for image operations. "
                "Install it with: pip install Pillow"
            )

        if self._pil_image is not None:
            return self._pil_image

        image_bytes = self.as_bytes()
        self._pil_image = PILImage.open(BytesIO(image_bytes))
        return self._pil_image

    def save(self, path: Union[str, Path], format: str = None):
        """Save image to file.

        Args:
            path: Output file path
            format: Image format (e.g., 'PNG', 'JPEG'). Auto-detected from path if None.

        Examples:
            >>> img.save("output.png")  # doctest: +SKIP
            >>> img.save("output.jpg", format="JPEG")  # doctest: +SKIP
        """
        if _PIL_AVAILABLE:
            pil_img = self.as_pil_image()
            pil_img.save(path, format=format)
        else:
            # Fallback: save raw bytes
            path = Path(path)
            with open(path, "wb") as f:
                f.write(self.as_bytes())

    def show(self):
        """Display the image.

        Requires PIL (Pillow) to be installed.

        Examples:
            >>> img.show()  # doctest: +SKIP
        """
        pil_img = self.as_pil_image()
        pil_img.show()

    def __repr__(self) -> str:
        """String representation."""
        return f"GeneratedImage(model={self.model}, url={bool(self.url)})"


def generate_image(
    prompt: str,
    *,
    model: str = None,
    size: str = None,
    quality: str = None,
    style: str = None,
    response_format: str = "url",
    **kwargs,
) -> GeneratedImage:
    """Generate a single image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        model: Model to use (e.g., 'dall-e-2', 'dall-e-3', 'stable-diffusion')
        size: Image size (e.g., '1024x1024', '512x512', '1792x1024')
        quality: Image quality ('standard' or 'hd' for DALL-E 3)
        style: Image style ('vivid' or 'natural' for DALL-E 3)
        response_format: Format of response ('url' or 'b64_json')
        **kwargs: Additional provider-specific parameters

    Returns:
        GeneratedImage object

    Raises:
        ImportError: If LiteLLM is not installed

    Examples:
        >>> from aix.image import generate_image
        >>> image = generate_image("A serene mountain landscape")  # doctest: +SKIP
        >>> image.save("landscape.png")  # doctest: +SKIP

        >>> # High quality with DALL-E 3
        >>> image = generate_image(
        ...     "Abstract art with vibrant colors",
        ...     model="dall-e-3",
        ...     quality="hd",
        ...     style="vivid"
        ... )  # doctest: +SKIP

        >>> # Specific size
        >>> image = generate_image(
        ...     "A futuristic city",
        ...     size="1792x1024"
        ... )  # doctest: +SKIP
    """
    if _litellm_image_generation is None:
        raise ImportError(
            "LiteLLM is required for image generation. "
            "Install it with: pip install litellm"
        )

    # Apply defaults from the active config (explicit args still win)
    _img_cfg = _get_config().image
    model = model or _img_cfg.model
    size = size or _img_cfg.size

    # Build parameters
    params = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": response_format,
    }

    if quality:
        params["quality"] = quality
    if style:
        params["style"] = style

    # Add additional kwargs
    params.update(kwargs)

    # Call LiteLLM
    response = _litellm_image_generation(**params)

    # Extract image data
    image_data = response.data[0]

    return GeneratedImage(
        url=getattr(image_data, "url", None),
        b64_json=getattr(image_data, "b64_json", None),
        model=model,
        prompt=prompt,
        revised_prompt=getattr(image_data, "revised_prompt", None),
    )


def generate_images(
    prompt: str,
    *,
    n: int = None,
    model: str = None,
    size: str = None,
    quality: str = None,
    style: str = None,
    response_format: str = "url",
    **kwargs,
) -> list[GeneratedImage]:
    """Generate multiple images from a text prompt.

    Args:
        prompt: Text description of images to generate
        n: Number of images to generate
        model: Model to use
        size: Image size
        quality: Image quality
        style: Image style
        response_format: Format of response
        **kwargs: Additional provider-specific parameters

    Returns:
        List of GeneratedImage objects

    Examples:
        >>> from aix.image import generate_images
        >>> images = generate_images(
        ...     "A cute robot",
        ...     n=3,
        ...     size="512x512"
        ... )  # doctest: +SKIP
        >>> for i, img in enumerate(images):  # doctest: +SKIP
        ...     img.save(f"robot_{i}.png")
    """
    if _litellm_image_generation is None:
        raise ImportError(
            "LiteLLM is required for image generation. "
            "Install it with: pip install litellm"
        )

    # Apply defaults from the active config (explicit args still win)
    _img_cfg = _get_config().image
    model = model or _img_cfg.model
    size = size or _img_cfg.size
    n = n or _img_cfg.num_images

    # Build parameters
    params = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": response_format,
    }

    if quality:
        params["quality"] = quality
    if style:
        params["style"] = style

    # Add additional kwargs
    params.update(kwargs)

    # Call LiteLLM
    response = _litellm_image_generation(**params)

    # Extract all images
    images = []
    for image_data in response.data:
        images.append(
            GeneratedImage(
                url=getattr(image_data, "url", None),
                b64_json=getattr(image_data, "b64_json", None),
                model=model,
                prompt=prompt,
                revised_prompt=getattr(image_data, "revised_prompt", None),
            )
        )

    return images


def edit_image(
    image_path: Union[str, Path],
    prompt: str,
    *,
    mask_path: Union[str, Path] = None,
    model: str = None,
    size: str = None,
    n: int = 1,
    **kwargs,
) -> Union[GeneratedImage, list[GeneratedImage]]:
    """Edit an existing image based on a prompt.

    Args:
        image_path: Path to the image to edit
        prompt: Description of the desired edit
        mask_path: Optional path to mask image (transparent areas will be edited)
        model: Model to use (typically 'dall-e-2' for edits)
        size: Output image size
        n: Number of variations to generate
        **kwargs: Additional provider-specific parameters

    Returns:
        GeneratedImage or list of GeneratedImage objects

    Examples:
        >>> from aix.image import edit_image
        >>> edited = edit_image(
        ...     "photo.png",
        ...     "Add a rainbow in the sky",
        ...     mask_path="sky_mask.png"
        ... )  # doctest: +SKIP
        >>> edited.save("edited_photo.png")  # doctest: +SKIP
    """
    if _litellm_image_generation is None:
        raise ImportError(
            "LiteLLM is required for image editing. "
            "Install it with: pip install litellm"
        )

    # Read image file
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Read mask if provided
    mask_data = None
    if mask_path:
        with open(mask_path, "rb") as f:
            mask_data = f.read()

    # Build parameters
    params = {
        "model": model or "dall-e-2",
        "prompt": prompt,
        "image": image_data,
        "n": n,
    }

    if mask_data:
        params["mask"] = mask_data
    if size:
        params["size"] = size

    params.update(kwargs)

    # Call LiteLLM (using custom_llm_provider for edits)
    response = _litellm_image_generation(**params)

    # Extract images
    images = []
    for img_data in response.data:
        images.append(
            GeneratedImage(
                url=getattr(img_data, "url", None),
                b64_json=getattr(img_data, "b64_json", None),
                model=params["model"],
                prompt=prompt,
            )
        )

    return images[0] if n == 1 else images


def create_variation(
    image_path: Union[str, Path],
    *,
    model: str = None,
    size: str = None,
    n: int = 1,
    **kwargs,
) -> Union[GeneratedImage, list[GeneratedImage]]:
    """Create variations of an existing image.

    Args:
        image_path: Path to the source image
        model: Model to use (typically 'dall-e-2')
        size: Output image size
        n: Number of variations to generate
        **kwargs: Additional provider-specific parameters

    Returns:
        GeneratedImage or list of GeneratedImage objects

    Examples:
        >>> from aix.image import create_variation
        >>> variations = create_variation(
        ...     "original.png",
        ...     n=3,
        ...     size="512x512"
        ... )  # doctest: +SKIP
        >>> for i, var in enumerate(variations):  # doctest: +SKIP
        ...     var.save(f"variation_{i}.png")
    """
    if _litellm_image_generation is None:
        raise ImportError(
            "LiteLLM is required for image variations. "
            "Install it with: pip install litellm"
        )

    # Read image file
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Build parameters
    params = {
        "model": model or "dall-e-2",
        "image": image_data,
        "n": n,
    }

    if size:
        params["size"] = size

    params.update(kwargs)

    # Call LiteLLM
    response = _litellm_image_generation(**params)

    # Extract images
    images = []
    for img_data in response.data:
        images.append(
            GeneratedImage(
                url=getattr(img_data, "url", None),
                b64_json=getattr(img_data, "b64_json", None),
                model=params["model"],
            )
        )

    return images[0] if n == 1 else images
