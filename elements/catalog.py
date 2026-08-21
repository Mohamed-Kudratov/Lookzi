#!/usr/bin/env python
"""Generate every prompt the Elements library needs, with coverage guaranteed.

The competitor's Elements library has eight categories -- Models, Portraits,
Poses, Backgrounds, Photography Styles, Clothing, Accessories, Shoes. Writing
those prompts by hand is several hundred lines of near-duplicate text, and the
mistakes it produces stay invisible until a LoRA trains badly weeks later.

So the taxonomy is declared once and the prompts are derived from it. Coverage
is allocated by round-robin rather than sampled at random, because a random draw
over 30 images leaves gaps -- and a gap in the angle axis is exactly what makes
an identity LoRA work only at the pose it was trained on (see CURATION.md).

    python elements/catalog.py                   # write manifest.csv, print summary
    python elements/catalog.py --category poses  # one category
    python elements/catalog.py --dry-run         # counts and coverage only
"""
import argparse
import csv
import os
import sys

# --------------------------------------------------------------------------
# Ethnicity wording, chosen by measurement -- see elements/ethnicity_probe.py.
#
# "Central Asian" does not work. Run it through the generator and it returns
# East and Southeast Asian faces, because that is what "Asian" overwhelmingly
# means in the training corpus and Central Asia is rare enough to be rounded to
# the nearest neighbour. A bare nationality does not work either, in the other
# direction: "Uzbek" and "Tajik" alone pull toward Persian and Middle Eastern.
#
# What worked in the probe was a nationality PLUS the bone structure. The name
# selects the region and the features hold it there. Z-Image-Turbo is
# CFG-distilled and runs at guidance_scale=0.0, so there is no negative prompt
# to push back with -- the positive description is the only lever there is.
#
# The clauses below carry ethnic structure only. Everything that separates one
# roster member from another stays in their own skin/hair/detail fields, or the
# roster collapses back into one face.
ETHNICITY = {
    "uzbek": dict(label="Uzbek Central Asian",
                  features="Turkic features, high wide cheekbones and a low nose bridge"),
    "kazakh": dict(label="Kazakh Central Asian",
                   features="Turkic features, prominent wide cheekbones and a "
                            "pronounced epicanthic fold"),
    "tajik": dict(label="Tajik Central Asian",
                  features="Persian features, a straight narrow nose and strong dark brows"),
    "slavic": dict(label="Slavic European", features=""),
}

# Which roster member is which. The mix is deliberate: Uzbekistan is not
# ethnically uniform, and a roster that is reads as a stock library rather than
# a local one. Mostly Uzbek, some Kazakh and Tajik, and the Slavic entries the
# roster already had.
ETHNICITY_BY_FACE = {
    "f_cauz_20s_slim": "tajik",
    "f_cauz_20s_avg": "uzbek",
    "f_cauz_30s_avg": "uzbek",
    "f_cauz_40s_full": "uzbek",
    "f_cauz_50s_avg": "kazakh",
    "f_cauz_20s_hijab": "uzbek",
    "f_cauz_40s_hijab": "tajik",
    "f_slav_20s_slim": "slavic",
    "f_slav_30s_avg": "slavic",
    "m_cauz_20s_slim": "uzbek",
    "m_cauz_30s_avg": "kazakh",
    "m_cauz_40s_avg": "uzbek",
    "m_slav_30s_avg": "slavic",
}

# The roster, from ROSTER.md
# --------------------------------------------------------------------------
# Every entry carries distinguishing features, not just demographics. Without
# them "Central Asian woman, mid 20s, average build" describes half the roster,
# the generator returns near-identical faces, and the library reads as one
# person in different clothes -- which is the opposite of a roster.
#
# Vary skin tone, hair and one memorable facial detail. Those three carry almost
# all of the perceived difference between two people at catalogue size.
ROSTER = [
    dict(id="f_cauz_20s_slim", gender="woman", appearance=None,
         age="early 20s", build="slim", modest=False,
         skin="light warm ivory skin", hair="long straight black hair",
         detail="high cheekbones and a narrow face"),
    dict(id="f_cauz_20s_avg", gender="woman", appearance=None,
         age="mid 20s", build="average", modest=False,
         skin="medium olive skin", hair="shoulder-length dark brown hair",
         detail="a round face and full lips"),
    dict(id="f_cauz_30s_avg", gender="woman", appearance=None,
         age="early 30s", build="average", modest=False,
         skin="tan golden skin", hair="dark chestnut hair in a low bun",
         detail="a strong jaw and arched brows"),
    dict(id="f_cauz_40s_full", gender="woman", appearance=None,
         age="late 30s", build="fuller", modest=False,
         skin="warm beige skin", hair="short bobbed black hair",
         detail="a soft oval face and wide-set eyes"),
    dict(id="f_cauz_50s_avg", gender="woman", appearance=None,
         age="early 50s", build="average", modest=False,
         skin="deeper tan skin with visible fine lines",
         hair="greying dark hair pulled back",
         detail="a lined face and deep-set eyes"),
    dict(id="f_cauz_20s_hijab", gender="woman", appearance=None,
         age="mid 20s", build="average", modest=True,
         skin="fair olive skin", hair="hair fully covered by the headscarf",
         detail="large dark almond eyes and thin brows"),
    dict(id="f_cauz_40s_hijab", gender="woman", appearance=None,
         age="late 30s", build="average", modest=True,
         skin="warm brown skin", hair="hair fully covered by the headscarf",
         detail="a broad face and a wide smile line"),
    dict(id="f_slav_20s_slim", gender="woman", appearance=None,
         age="early 20s", build="slim", modest=False,
         skin="very fair pale skin with freckles", hair="long light blonde hair",
         detail="pale grey-blue eyes and a small straight nose"),
    dict(id="f_slav_30s_avg", gender="woman", appearance=None,
         age="early 30s", build="average", modest=False,
         skin="fair rosy skin", hair="mid-length auburn hair",
         detail="green eyes and a square jaw"),
    dict(id="m_cauz_20s_slim", gender="man", appearance=None,
         age="early 20s", build="slim", modest=False,
         skin="light olive skin", hair="short black hair, clean shaven",
         detail="a narrow face and sharp cheekbones"),
    dict(id="m_cauz_30s_avg", gender="man", appearance=None,
         age="early 30s", build="average", modest=False,
         skin="tan skin", hair="short dark hair with a trimmed beard",
         detail="a square face and heavy brows"),
    dict(id="m_cauz_40s_avg", gender="man", appearance=None,
         age="late 40s", build="average", modest=False,
         skin="weathered brown skin", hair="greying short hair and a moustache",
         detail="a lined forehead and a broad nose"),
    dict(id="m_slav_30s_avg", gender="man", appearance=None,
         age="early 30s", build="average", modest=False,
         skin="fair skin", hair="light brown hair, short beard",
         detail="blue eyes and a long face"),
]


# Filled in from ETHNICITY rather than written inline, so the wording lives in
# exactly one place. It was already changed once when the probe showed the old
# phrasing missed the region entirely; thirteen inline copies would have made
# that a thirteen-line edit with one of them silently left behind.
for _f in ROSTER:
    _f["ethnicity"] = ETHNICITY_BY_FACE[_f["id"]]
    _f["appearance"] = ETHNICITY[_f["ethnicity"]]["label"]
    # Kept as its own clause rather than folded into `appearance`. Inlined, it
    # landed between the article and the noun -- "a Uzbek Central Asian, Turkic
    # features, high wide cheekbones ... woman" -- and it also collided with the
    # per-face `detail`, so two different face shapes were asserted in one
    # sentence. As a separate clause after the build it reads as a sentence and
    # the two never contradict, because the ethnic clause describes bone
    # structure and `detail` describes the individual.
    _f["features"] = ETHNICITY[_f["ethnicity"]]["features"]
assert all(f["appearance"] for f in ROSTER)

# --------------------------------------------------------------------------
# Coverage axes for identity datasets
# --------------------------------------------------------------------------
ANGLES = [
    "front view, facing camera",
    "three-quarter view turned to the left",
    "three-quarter view turned to the right",
    "near profile view",
    "shot from slightly above eye level",
    "shot from slightly below eye level",
]

# Skewed to full body on purpose: these models wear clothes, so the LoRA has to
# lock proportions and not only a face. See ROSTER.md.
DISTANCE_MIX = (["headshot"] * 4
                + ["half body, from the waist up"] * 9
                + ["full body, head to feet"] * 17)

LIGHTING = [
    "soft diffused daylight",
    "hard directional sunlight",
    "strong side light",
    # Not "rim light from behind". That phrasing produced a bright outline
    # traced around the head and shoulders on every image it touched -- the
    # look of a badly cut-out sticker rather than a backlit photograph. "Rim
    # light" correlates with hard edge highlights in product photography, so
    # the model drew the edge instead of the light. Describing the source and
    # ruling out the halo explicitly fixes it.
    "gentle backlight from a low sun, soft haze, no glowing outline",
]

BACKGROUNDS_TRAIN = [
    "plain light grey studio backdrop",
    "warm beige wall",
    "white seamless studio",
    "blurred outdoor street",
    "blurred plain interior",
    "soft gradient backdrop",
]

# Varied and plain. A single repeated outfit gets learned as part of the
# identity and then fights every garment put on the model afterwards.
CLOTHING_NEUTRAL = [
    "a plain white t-shirt and dark grey trousers",
    "a plain black t-shirt and blue jeans",
    "a plain navy long-sleeve top and beige trousers",
    "a plain grey knit top and black trousers",
    "a plain olive shirt and dark jeans",
]
CLOTHING_MODEST = [
    "a plain long-sleeve beige tunic and wide dark trousers with a matching headscarf",
    "a plain navy long dress with long sleeves and a matching headscarf",
    "a plain grey long-sleeve top and long black skirt with a matching headscarf",
    "a plain olive long tunic and loose trousers with a matching headscarf",
    "a plain white long-sleeve blouse and long navy skirt with a matching headscarf",
]

EXPRESSION_MIX = ["neutral"] * 20 + ["a slight natural smile"] * 10

# The tail that keeps output away from the plastic AI look. Three of them,
# because "visible pores and fine skin detail" is nonsense on a photograph of a
# shoe -- it wastes conditioning and invites artefacts on a surface that has no
# skin. Match the tail to what is actually in frame.
REALISM_PERSON = ("natural skin texture, visible pores and fine skin detail, "
                  "realistic fabric drape, shot on 85mm lens, shallow depth of "
                  "field, photorealistic, unretouched")
REALISM_PRODUCT = ("sharp material texture, visible fabric weave and stitching, "
                   "accurate colour, soft natural contact shadow, "
                   "photorealistic product photography, high detail")
REALISM_SCENE = ("realistic surface texture, natural light falloff, "
                 "photorealistic, high detail, no people")

# Kept for readability in the person-facing builders below.
REALISM = REALISM_PERSON

# --------------------------------------------------------------------------
# Other Elements categories
# --------------------------------------------------------------------------
POSES = [
    "standing straight, arms relaxed at the sides",
    "standing with weight on one hip, one hand on the hip",
    "standing with both hands in pockets",
    "standing with arms crossed",
    "walking towards the camera mid-stride",
    "walking away, glancing back over the shoulder",
    "standing in three-quarter turn, looking over the shoulder",
    "leaning back against a wall",
    "leaning sideways against a wall, one foot flexed",
    "sitting on a low stool, back straight",
    "sitting on the floor, legs extended to one side",
    "sitting cross-legged on the floor",
    "sitting on the floor, knees drawn up, arms resting on the knees",
    "crouching on the balls of the feet",
    "kneeling upright on both knees",
    "one hand raised to the hair, elbow out",
    "both hands raised to the back of the head",
    "hands clasped in front, standing tall",
    "one arm extended down, the other hand touching the collar",
    "half turned away, head turned back to camera",
    "seated on a chair, leaning forward, elbows on the knees",
    "seated on a chair, legs crossed",
    "standing on tiptoe, arms slightly out for balance",
    "mid-twist, torso turned away and face to camera",
    "standing with one leg crossed over the other",
    "hands adjusting a cuff at the wrist",
    "hands adjusting a collar",
    "arms loose, caught mid-laugh",
    "standing square to camera, chin slightly lifted",
    "reclining on one elbow on the floor",
]

BACKGROUNDS = [
    "seamless white studio cyclorama",
    "seamless light grey studio backdrop",
    "seamless warm beige backdrop",
    "seamless deep charcoal backdrop",
    "seamless terracotta backdrop",
    "seamless sage green backdrop",
    "textured plaster wall in a warm tone",
    "textured concrete wall in a cool tone",
    "muted painted brick wall",
    "cream fabric drape backdrop",
    "grey fabric drape backdrop",
    "minimal interior with a window and daylight",
    "minimal interior with a wooden floor and plain walls",
    "sunlit room with long window shadows",
    "modern lobby with clean lines",
    "blurred city street in daytime",
    "blurred city street at golden hour",
    "blurred park greenery",
    "desert sand and open sky",
    "old town stone architecture",
    "marble surface with a soft shadow",
    "gradient backdrop from light to dark grey",
    "gradient backdrop from warm cream to tan",
    "outdoor courtyard in midday shade",
    "plain studio with a visible floor line",
]

PHOTOGRAPHY_STYLES = [
    "clean e-commerce catalogue lighting, even and shadowless",
    "editorial fashion photography with dramatic contrast",
    "soft natural window light, lifestyle feel",
    "high-key, bright and airy",
    "low-key and moody with deep shadows",
    "golden hour warm backlight",
    "overcast diffuse daylight",
    "hard direct on-camera flash",
    "film photography grain with muted colour",
    "black and white fine art",
    "polished studio three-point lighting",
    "candid documentary feel",
    "warm vintage colour grade",
    "cool desaturated modern grade",
    "beauty lighting, soft and frontal with minimal shadow",
]

CLOTHING = [
    ("tops", ["a plain white cotton t-shirt", "a plain black t-shirt",
              "a white cotton shirt", "a light blue oxford shirt",
              "a grey marl sweatshirt", "a cream knit jumper",
              "a navy long-sleeve top", "a black ribbed tank top",
              "a beige linen blouse", "a striped long-sleeve tee"]),
    ("bottoms", ["straight blue jeans", "dark wash slim jeans",
                 "black tailored trousers", "beige chino trousers",
                 "wide-leg cream trousers", "a black midi skirt",
                 "a denim mini skirt", "a pleated grey skirt"]),
    ("outerwear", ["a black leather jacket", "a beige trench coat",
                   "a navy wool overcoat", "a denim jacket", "a grey blazer",
                   "a quilted puffer jacket", "an olive utility jacket"]),
    ("dresses", ["a plain white midi dress", "a black slip dress",
                 "a floral summer dress", "a knitted long dress",
                 "a linen shirt dress"]),
    ("modest", ["a long-sleeve maxi dress", "a loose tunic with wide trousers",
                "a long cardigan over a maxi skirt", "a plain black abaya",
                "a hijab in plain neutral tones"]),
]

ACCESSORIES = [
    "a tan structured leather handbag", "a small black crossbody bag",
    "a natural canvas tote bag", "a leather belt with a simple buckle",
    "gold hoop earrings", "a silver pendant necklace",
    "layered fine chain necklaces", "an analogue wristwatch with a leather strap",
    "a steel wristwatch with a metal bracelet", "a thin gold bangle",
    "a wide brim felt hat", "a knitted beanie", "a patterned silk scarf",
    "a plain wool scarf", "round wire-frame glasses",
    "rectangular acetate glasses", "aviator sunglasses", "oversized sunglasses",
    "black leather gloves", "a minimal backpack",
]

SHOES = [
    "white leather sneakers", "black running trainers", "canvas plimsolls",
    "black leather ankle boots", "tan suede chelsea boots",
    "knee-high leather boots", "black leather loafers", "brown brogues",
    "black court heels", "strappy heeled sandals", "flat leather sandals",
    "espadrilles", "chunky platform sneakers", "ballet flats", "hiking boots",
    "black derby shoes", "mule slides", "combat boots",
]



def article(face):
    """"a" or "an" plus the ethnic label.

    "a Uzbek" reads as a typo in a caption and the model has seen very few of
    them; the corpus writes "an Uzbek".
    """
    label = face["appearance"]
    return ("an " if label[0].upper() in "AEIOU" else "a ") + label


def feature_clause(face):
    """The ethnic bone structure, or nothing.

    Empty for the Slavic entries: the probe showed "Slavic European" already
    lands, and an unnecessary anatomical clause only competes with the per-face
    detail for the same slot in the sentence.
    """
    return f"{face['features']}, " if face["features"] else ""


def _clothing_for(modest):
    return CLOTHING_MODEST if modest else CLOTHING_NEUTRAL


def roster_prompts(per_face=30):
    """Identity datasets. Coverage is allocated, not sampled.

    Round-robin over each axis independently guarantees every angle, light,
    background and distance bucket is hit. Sampling at random over 30 draws
    leaves holes, and a hole in the angle axis is what makes a LoRA work only
    at the angle it saw.
    """
    rows = []
    for face in ROSTER:
        clothes = _clothing_for(face["modest"])
        pronoun = "her" if face["gender"] == "woman" else "his"
        for i in range(per_face):
            angle = ANGLES[i % len(ANGLES)]
            distance = DISTANCE_MIX[i % len(DISTANCE_MIX)]
            light = LIGHTING[i % len(LIGHTING)]
            bg = BACKGROUNDS_TRAIN[i % len(BACKGROUNDS_TRAIN)]
            outfit = clothes[i % len(clothes)]
            expression = EXPRESSION_MIX[i % len(EXPRESSION_MIX)]
            rows.append({
                "category": "models",
                "id": "%s__%03d" % (face["id"], i),
                "group": face["id"],
                "prompt": (f"photorealistic photograph of {article(face)} "
                           f"{face['gender']} in {pronoun} {face['age']}, "
                           f"{face['build']} build, {feature_clause(face)}"
                           f"{face['skin']}, {face['hair']}, "
                           f"{face['detail']}, wearing {outfit}, "
                           f"{distance}, {angle}, {light}, {bg}, "
                           f"{expression} expression, {REALISM}"),
                "angle": angle.split(",")[0],
                "distance": distance.split(",")[0],
                "lighting": light,
                "background": bg,
                "expression": expression,
            })
    return rows


def portrait_prompts(per_face=6):
    """Headshots of the same roster faces, listed separately by the competitor."""
    rows = []
    for face in ROSTER:
        clothes = _clothing_for(face["modest"])
        pronoun = "her" if face["gender"] == "woman" else "his"
        for i in range(per_face):
            angle = ANGLES[i % len(ANGLES)]
            light = LIGHTING[i % len(LIGHTING)]
            bg = BACKGROUNDS_TRAIN[i % len(BACKGROUNDS_TRAIN)]
            rows.append({
                "category": "portraits",
                "id": "%s__portrait_%02d" % (face["id"], i),
                "group": face["id"],
                "prompt": (f"photorealistic head and shoulders portrait of "
                           f"{article(face)} {face['gender']} in {pronoun} "
                           f"{face['age']}, {feature_clause(face)}"
                           f"{face['skin']}, {face['hair']}, "
                           f"{face['detail']}, wearing "
                           f"{clothes[i % len(clothes)]}, {angle}, {light}, {bg}, "
                           f"neutral expression, {REALISM}"),
                "angle": angle.split(",")[0],
                "distance": "headshot",
                "lighting": light,
                "background": bg,
                "expression": "neutral",
            })
    return rows


def pose_prompts():
    """Pose references.

    DWPose reads a skeleton out of these at generation time, so what matters is
    that the body is unambiguous: plain clothes, plain background, whole figure
    in frame including the feet.
    """
    rows = []
    for gender in ("woman", "man"):
        outfit = ("a plain white top and blue jeans" if gender == "woman"
                  else "a plain white t-shirt and blue jeans")
        for i, pose in enumerate(POSES):
            rows.append({
                "category": "poses",
                "id": "pose_%s_%03d" % (gender, i),
                "group": "poses_" + gender,
                "prompt": (f"photorealistic full body photograph of a {gender} wearing "
                           f"{outfit}, {pose}, plain light grey studio backdrop, "
                           f"even soft lighting, whole figure in frame with the feet "
                           f"visible, {REALISM}"),
                "angle": "", "distance": "full body", "lighting": "even soft",
                "background": "plain grey studio", "expression": "neutral",
            })
    return rows


def simple_prompts(category, items, template, tail, group=None):
    rows = []
    for i, item in enumerate(items):
        rows.append({
            "category": category,
            "id": "%s_%03d" % (category, i),
            "group": group or category,
            "prompt": template.format(item=item) + ", " + tail,
            "angle": "", "distance": "", "lighting": "",
            "background": "", "expression": "",
        })
    return rows


def clothing_prompts():
    rows, n = [], 0
    for sub, items in CLOTHING:
        for item in items:
            rows.append({
                "category": "clothing",
                "id": "clothing_%03d" % n,
                "group": sub,
                "prompt": (f"professional e-commerce product photograph of {item}, "
                           f"laid flat on a plain white background, centred, "
                           f"even shadowless lighting, no model, {REALISM_PRODUCT}"),
                "angle": "", "distance": "", "lighting": "even shadowless",
                "background": "white", "expression": "",
            })
            n += 1
    return rows


BUILDERS = {
    "models": roster_prompts,
    "portraits": portrait_prompts,
    "poses": pose_prompts,
    "backgrounds": lambda: simple_prompts(
        "backgrounds", BACKGROUNDS,
        "photorealistic empty scene, {item}, no people and no objects, "
        "wide framing suitable as a photographic backdrop",
        REALISM_SCENE),
    "photography_styles": lambda: simple_prompts(
        "photography_styles", PHOTOGRAPHY_STYLES,
        "photorealistic full body fashion photograph of a person in plain "
        "clothing on a plain backdrop, {item}",
        REALISM_PERSON),
    "clothing": clothing_prompts,
    "accessories": lambda: simple_prompts(
        "accessories", ACCESSORIES,
        "professional e-commerce product photograph of {item} on a plain white "
        "background, centred, even shadowless lighting, no model",
        REALISM_PRODUCT),
    "shoes": lambda: simple_prompts(
        "shoes", SHOES,
        "professional e-commerce product photograph of a pair of {item}, "
        "three-quarter view on a plain white background, even shadowless "
        "lighting, no model",
        REALISM_PRODUCT),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "manifest.csv"))
    ap.add_argument("--category", default=None, help="one of: " + ", ".join(BUILDERS))
    ap.add_argument("--per-face", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cats = [args.category] if args.category else list(BUILDERS)
    for c in cats:
        if c not in BUILDERS:
            ap.error("unknown category %r; choose from %s" % (c, ", ".join(BUILDERS)))

    rows = []
    for c in cats:
        rows += roster_prompts(args.per_face) if c == "models" else BUILDERS[c]()

    print("%-22s%9s%9s" % ("category", "prompts", "groups"))
    for c in cats:
        sub = [r for r in rows if r["category"] == c]
        print("  %-20s%9d%9d" % (c, len(sub), len(set(r["group"] for r in sub))))
    print("  %-20s%9d" % ("TOTAL", len(rows)))

    # Coverage check on the identity datasets -- the one that fails silently.
    models = [r for r in rows if r["category"] == "models"]
    if models:
        one = [r for r in models if r["group"] == ROSTER[0]["id"]]
        print("\ncoverage per face (%d images):" % len(one))
        for axis in ("angle", "distance", "lighting", "background", "expression"):
            counts = {}
            for r in one:
                counts[r[axis]] = counts.get(r[axis], 0) + 1
            expected = {"angle": len(ANGLES), "distance": 3, "lighting": len(LIGHTING),
                        "background": len(BACKGROUNDS_TRAIN), "expression": 2}[axis]
            gap = "" if len(counts) >= expected else "   <-- GAP, only %d of %d" % (
                len(counts), expected)
            print("  %-12s%d values  %s%s" % (
                axis, len(counts), sorted(counts.values(), reverse=True), gap))

    if args.dry_run:
        return 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  -> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
