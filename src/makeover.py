from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from .styles import STYLE_PRESETS, BASE_PROMPT, NEGATIVE_PROMPT
from .utils import resize_for_model, set_seed


class RoomMakeoverModel:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize the Stable Diffusion Img2Img pipeline.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # optional: disable if you want faster inference
        )

        self.pipe = self.pipe.to(device)
        self.device = device

    def build_prompt(self, style_name: str) -> str:
        """Combine base prompt with style description."""
        style = STYLE_PRESETS.get(style_name)
        if not style:
            return BASE_PROMPT

        style_text = style["description"]
        return f"{BASE_PROMPT}, {style_text}"

    def restyle_room(
        self,
        image: Image.Image,
        style_name: str,
        strength: float = 0.6,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Apply style to a room image using Stable Diffusion Img2Img.

        - strength: how strongly to transform the image (0.0–1.0, typical 0.4–0.8)
        - guidance_scale: how strongly the prompt guides the result (higher = more prompt)
        """
        seed = set_seed(seed)

        # Prepare image
        image = resize_for_model(image)

        prompt = self.build_prompt(style_name)

        result = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=NEGATIVE_PROMPT,
        )

        out_image = result.images[0]
        return out_image
