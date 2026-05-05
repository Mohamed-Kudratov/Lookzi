from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image


@dataclass
class TryOnRequest:
    person_image: Image.Image
    garment_image: Image.Image
    category: str
    seed: int = 42
    steps: int = 50
    guidance_scale: float = 2.5
    width: int = 768
    height: int = 1024
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TryOnResult:
    image: Image.Image
    engine_name: str
    category: str
    mask: Optional[Image.Image] = None
    masked_person: Optional[Image.Image] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TryOnEngine:
    name = "base"
    supported_categories: set[str] = set()

    def can_handle(self, category: str) -> bool:
        return category in self.supported_categories

    def run(self, request: TryOnRequest) -> TryOnResult:
        raise NotImplementedError
