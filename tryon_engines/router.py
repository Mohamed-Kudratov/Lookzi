from .base import TryOnEngine, TryOnRequest, TryOnResult


class EngineRouter:
    def __init__(self, engines: list[TryOnEngine]):
        self.engines = engines

    def choose(self, category: str) -> TryOnEngine:
        for engine in self.engines:
            if engine.can_handle(category):
                return engine
        raise ValueError(f"No try-on engine is registered for category: {category}")

    def run(self, request: TryOnRequest) -> TryOnResult:
        engine = self.choose(request.category)
        return engine.run(request)
