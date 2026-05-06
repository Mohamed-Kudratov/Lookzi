from .base import TryOnEngine, TryOnRequest, TryOnResult
from .baseline import IdentityBaselineEngine
from .router import EngineRouter

__all__ = [
    "CatVTONEngine",
    "EngineRouter",
    "IdentityBaselineEngine",
    "TryOnEngine",
    "TryOnRequest",
    "TryOnResult",
]


def __getattr__(name):
    if name == "CatVTONEngine":
        from .catvton import CatVTONEngine
        return CatVTONEngine
    raise AttributeError(name)
