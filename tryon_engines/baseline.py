from PIL import Image, ImageOps

from .base import TryOnEngine, TryOnRequest, TryOnResult


class IdentityBaselineEngine(TryOnEngine):
    name = "identity_baseline"
    supported_categories = {"upper", "lower", "overall", "inner", "outer"}

    def run(self, request: TryOnRequest) -> TryOnResult:
        image = ImageOps.fit(
            request.person_image.convert("RGB"),
            (request.width, request.height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return TryOnResult(
            image=image,
            engine_name=self.name,
            category=request.category,
            mask=None,
            masked_person=image,
            diagnostics={
                "baseline": True,
                "note": "Returns the resized person image without try-on.",
            },
        )
