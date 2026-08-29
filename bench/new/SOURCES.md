# Where the benchmark material comes from

Nothing here has been through our system before. That is the whole point: the
first attempt at this set was assembled from our own uploads, which is the set
the product has already been tuned against, and a score from it would have
flattered us.

## Tier 1 -- studio

`forgeml/viton_hd`, the VITON-HD mirror. 768x1024 garments photographed flat on
white, ghost-mannequin style. Upper body only, which is a real limit of this
tier: no trousers, no dresses.

It also carries `image` -- a photograph of a model actually wearing that
garment. That is ground truth, and it is worth more than the garment shot: a
try-on result can be put beside the real photograph of the real person in the
real garment instead of beside somebody's opinion.

Research licence. Fine for measuring ourselves; not a source of anything we
ship.

## Tier 2 -- a competent phone photograph

`wargoninnovation/clothingdatasetsecondhand`, CC-BY-4.0. 43,100 photographs of
second-hand clothing laid out and shot from above -- trousers, shirts, dresses,
cardigans, jackets. Real listing photographs: creased, folded, sleeves crossed,
some of them bunched. Carries condition, damage, stain and material fields.

Attribution: Wargön Innovation, CC-BY-4.0.

## Tier 3 -- not gathered

Nothing found. Public datasets are curated by definition, and the tier that
matters most to this product -- a garment on a patterned bedspread under a
yellow bulb, half out of frame, shot at an angle -- is exactly what nobody
publishes. It has to be photographed, and it has to be photographed where the
customers are, because "bad light" in Tashkent is not the same light.

## A dead end, recorded so nobody repeats it

`Codatta/Fashion-1K` describes itself as flat-lay and ghost-mannequin with no
human figures. The images are styling collages: a dress next to handbags,
shoes, perfume and jewellery, several products per frame. The card is wrong.
Forty were downloaded and looked at before this was noticed.
