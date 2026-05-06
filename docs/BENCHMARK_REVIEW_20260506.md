# Lookzi Benchmark Review - 2026-05-06

## Human Result

Latest submitted human review file:

```text
lookzi_human_review_20260506_115215.json
```

Summary:

| Rating | Count |
|---|---:|
| GOOD | 1 |
| OK | 0 |
| BAD | 6 |
| MODEL_FAIL | 0 |
| MASK_FAIL | 1 |
| Total | 8 |

Pair-level result:

| Pair | Rating |
|---|---|
| C01 | MASK_FAIL |
| C02 | GOOD |
| C03 | BAD |
| C04 | BAD |
| C05 | BAD |
| C06 | BAD |
| C07 | BAD |
| C08 | BAD |

## Decision

This CatVTON-based setup is not production-ready for the current Lookzi catalog benchmark.

The result is not a small threshold issue. With only 1/8 acceptable human outcomes, continuing to tune masks alone is unlikely to produce a reliable product path. C01 is explicitly a mask failure, but the rest being BAD means the generation engine and category/model fit are also failing.

## Next Engineering Step

Stop treating this as only a mask-engine problem.

The next milestone should be an engine gate:

1. Keep CatVTON as one candidate engine, not the default product decision.
2. Add a second candidate engine behind the existing `tryon_engines` abstraction.
3. Run the same 8-pair benchmark against both engines.
4. Promote only categories that pass human review, for example `>= 6/8 GOOD or OK` with no repeated severe artifact pattern.
5. Disable or mark unsupported any category that cannot pass the gate.

For the current evidence, Lookzi should not promise reliable lower, skirt, or overall try-on from this CatVTON setup.
