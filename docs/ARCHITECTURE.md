# Lookzi AI Stylist Commerce Platform

Lookzi is no longer treated as a single virtual try-on model. The product direction is an AI stylist commerce platform where try-on is one rendering capability inside a larger system.

## Current Scope

The current build focuses on store-provided garments only.

- A store/admin provides garment images and metadata.
- A user uploads a person image.
- The user selects one store garment.
- Lookzi generates a visual preview.
- The system records which garment categories work reliably with which engine.

Recommendation AI is intentionally out of scope for now. It should be added only after the rendering engines can produce reliable previews for the target garment categories.

## Future Scope

Later, Lookzi can recommend full outfits from store catalog items:

- T-shirt + jeans + overshirt + cap.
- Layered looks.
- Occasion-aware styling.
- Budget and size-aware product matching.

That future system should build structured outfit plans first, then render previews. It should not depend on one try-on model doing every task.

## Architecture

```text
User Photo
  -> Person Understanding
  -> Garment Selection
  -> Category Router
  -> Try-On Engine
  -> Preview Image
  -> Quality/Benchmark Log
```

## Engine Strategy

Lookzi must support multiple engines behind one interface.

```text
TryOnEngine
  - CatVTONEngine
  - IDMEngine
  - CreativeEditEngine
  - FutureEngine
```

The user should not choose an engine manually. The system should route internally by category and reliability.

## Category Policy

Initial policy:

| Category | Preferred engine | Status |
| --- | --- | --- |
| upper/t-shirt/shirt | CatVTON | Candidate |
| cardigan/jacket | CatVTON + mask tuning | Candidate |
| pants | CatVTON or alternative | Needs benchmark |
| shorts | CatVTON or alternative | Needs benchmark |
| skirt | CreativeEdit or alternative VTON | CatVTON weak |
| dress/overall | CreativeEdit or alternative VTON | CatVTON weak |
| layered outfit | Future creative/styling pipeline | Not current scope |

## Accurate vs Creative Preview

Lookzi should support two preview types:

### Accurate Try-On

- Goal: keep the selected product close to the original garment.
- Best for simple categories that the engine handles reliably.
- Current candidate: CatVTON for upper garments.

### Creative Preview

- Goal: produce a realistic style visualization.
- May alter garment details.
- Useful for skirts, dresses, layered outfits, and recommendation previews.
- Not a strict product-accurate proof.

## Benchmark First

No engine should be considered production-ready without category benchmarks.

Each test should record:

- input person image
- garment image
- category
- engine
- output image
- mask preview
- garment similarity
- body preservation
- visual realism
- notes/failure reason

## Product Rule

Do not keep patching masks indefinitely when the model cannot perform the category. If a category repeatedly fails with a clean mask, mark the engine as unsuitable for that category and route to another engine.
