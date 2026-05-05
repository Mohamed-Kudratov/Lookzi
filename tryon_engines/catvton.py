from model.cloth_masker import vis_mask
import numpy as np
from utils import infer_garment_style, resize_and_crop, resize_and_padding

from .base import TryOnEngine, TryOnRequest, TryOnResult


class CatVTONEngine(TryOnEngine):
    name = "catvton"
    supported_categories = {"upper", "lower", "overall", "inner", "outer"}

    def __init__(self, pipeline, automasker, mask_processor, device):
        self.pipeline = pipeline
        self.automasker = automasker
        self.mask_processor = mask_processor
        self.device = device

    def run(self, request: TryOnRequest) -> TryOnResult:
        import torch

        person_image = resize_and_crop(request.person_image, (request.width, request.height))
        garment_image = resize_and_padding(request.garment_image, (request.width, request.height))
        garment_style, covers_lower_legs, style_debug = infer_garment_style(
            request.garment_image,
            request.category,
        )

        mask = request.metadata.get("mask")
        if mask is not None:
            mask = resize_and_crop(mask, (request.width, request.height))
        else:
            mask = self.automasker(
                person_image,
                request.category,
                garment_style=garment_style,
                covers_lower_legs=covers_lower_legs,
            )["mask"]
        mask = self.mask_processor.blur(mask, blur_factor=9)

        generator = None
        if request.seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)

        result_image = self.pipeline(
            image=person_image,
            condition_image=garment_image,
            mask=mask,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )[0]

        mask_coverage = float((np.array(mask) > 127).mean() * 100)
        return TryOnResult(
            image=result_image,
            engine_name=self.name,
            category=request.category,
            mask=mask,
            masked_person=vis_mask(person_image, mask),
            diagnostics={
                "garment_style": garment_style,
                "covers_lower_legs": covers_lower_legs,
                "style_debug": style_debug,
                "mask_coverage": round(mask_coverage, 2),
            },
        )
