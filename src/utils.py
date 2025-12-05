from typing import Optional
from PIL import Image
import numpy as np
import torch
import random


def load_image(file) -> Image.Image:
    """Load image from a file-like object into a Pillow image (RGB)."""
    image = Image.open(file).convert("RGB")
    return image


def resize_for_model(image: Image.Image, max_side: int = 768) -> Image.Image:
    """
    Resize image while keeping aspect ratio so that the longest side is <= max_side.
    Stable Diffusion often works well with 512 or 768.
    """
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)  # don't upscale
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def set_seed(seed: Optional[int] = None) -> int:
    """
    Set random seed for reproducibility. If seed is None, generate one.
    Returns the seed used.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
