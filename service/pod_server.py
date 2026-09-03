#!/usr/bin/env python3
"""The model, behind HTTP, on the pod.

    python -m service.pod_server        # on the pod

One process holding one loaded model, answering one request at a time. It knows
nothing about jobs, credits, users or the queue -- those live where they can be
backed up, and this lives where the GPU is.

Why HTTP rather than the queue-polling worker in gpu_worker.py. That one needs
to reach Postgres, which means Postgres has to be reachable from a rented
machine on someone else's network, which means either exposing the database to
the internet or paying for a managed one before the product has a single user.
This inverts it: the pod exposes one endpoint on loopback, we reach it through
the ssh connection we already have, and nothing about our side becomes public.

    ssh -N -L 18000:127.0.0.1:8000 root@<pod-ip> -p <pod-port>

Bound to 127.0.0.1 deliberately, and that is the whole security model. The pod
is a rented machine with a public address; a service on 0.0.0.0 there is a
service on the internet. On loopback it is reachable only by something already
inside the pod, and an ssh tunnel is exactly that.

Images travel as bytes in the request. That is the right trade at this size and
the wrong one later: at scale the bytes should go straight between the customer
and object storage with only signed links crossing the wire, which is what
runpod_handler.py does. Doing it that way now would mean a public bucket and an
account, before knowing whether anyone wants the product.
"""
import io
import os
import sys
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH = os.environ.get("MODEL_PATH", "ovedrive/Qwen-Image-Edit-2509-4bit")
LORA_DIR = os.environ.get("LORA_DIR", os.path.join(ROOT, "weights"))
LIGHTNING = int(os.environ.get("LIGHTNING", "8"))
MAX_BYTES = int(os.environ.get("MAX_INPUT_BYTES", str(32 * 1024 * 1024)))

# Where the cutout model's weights live. On the container disk they are fetched
# again on every pod; on the volume they are fetched once.
os.environ.setdefault("U2NET_HOME", "/workspace/.u2net")
# The finished packshot. Three by four to match everything else we produce, so
# a seller's catalogue does not come back in two shapes.
PACKSHOT_SIZE = (int(os.environ.get("PACKSHOT_W", "768")),
                 int(os.environ.get("PACKSHOT_H", "1024")))
PACKSHOT_MARGIN = float(os.environ.get("PACKSHOT_MARGIN", "0.07"))

app = FastAPI(title="Lookzi pod server")

_pipe = None
_load_seconds = None
_error = None
# One card, one model, one image at a time. Two requests sampling at once would
# not go twice as fast; they would contend for the same VRAM and make both slow
# and occasionally kill one with an allocation failure. Queueing here keeps the
# failure mode boring.
_gpu = threading.Lock()
_stats = {"served": 0, "failed": 0, "seconds": 0.0}


def load():
    global _pipe, _load_seconds, _error
    from pipeline import LayeringVTONPipeline
    t = time.time()
    print(f"[pod] loading {MODEL_PATH}", flush=True)
    try:
        _pipe = LayeringVTONPipeline(MODEL_PATH, LORA_DIR, lightning=LIGHTNING)
        _load_seconds = round(time.time() - t, 1)
        print(f"[pod] ready in {_load_seconds}s", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        # Held rather than raised, so /health can say what went wrong. A worker
        # that dies at startup leaves whoever is waiting to guess between a
        # crash, a slow load and a wrong address.
        _error = f"{type(exc).__name__}: {exc}"
        print(f"[pod] failed to load: {_error}", flush=True)


@app.on_event("startup")
def _startup():
    # In a thread, so the port is listening while the weights load. Otherwise
    # every health check during the three minutes of loading is a connection
    # refused, which reads as "the server is not running".
    threading.Thread(target=load, daemon=True).start()


@app.get("/health")
def health():
    return {"ready": _pipe is not None, "error": _error,
            "model": MODEL_PATH, "lightning": LIGHTNING,
            "load_seconds": _load_seconds, "busy": _gpu.locked(),
            "served": _stats["served"], "failed": _stats["failed"],
            "mean_seconds": round(_stats["seconds"] / _stats["served"], 2)
            if _stats["served"] else None}


def _read(upload: UploadFile, name: str) -> Image.Image:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{name} is larger than {MAX_BYTES} bytes")
    if not data:
        raise HTTPException(400, f"{name} is empty")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(400, f"{name} is not an image: {exc}")


@app.post("/generate")
def generate(person: UploadFile = File(...),
             garment: UploadFile = File(...),
             mode: str = Form("upper"),
             description: str = Form(""),
             seed: int = Form(42)):
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    if _pipe is None:
        raise HTTPException(503, "the model is still loading")

    from utils import process_inputs

    person_img = _read(person, "person")
    garment_img = _read(garment, "garment")
    p, g, pose = process_inputs(person_img, garment_img, None)

    started = time.time()
    with _gpu:
        try:
            out = _pipe(person_img=p, garment_img=g, pose_img=pose,
                        description=description or "the garment",
                        mode=mode, seed=int(seed))
        except Exception as exc:                              # noqa: BLE001
            _stats["failed"] += 1
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed

    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(
        content=buf.getvalue(), media_type="image/png",
        # In headers rather than a JSON envelope, so the body stays a plain
        # image that anything can open, including a browser pointed at it.
        headers={"X-Seconds": str(elapsed),
                 "X-Width": str(out.width), "X-Height": str(out.height)})


# ---------------------------------------------------------------------------
# packshot
#
# Not a generation. A packshot is the garment on its own, cleanly, and the
# honest way to produce one from a photograph is to cut the garment out and
# present it -- not to ask a diffusion model to imagine it again and hope the
# buttons survive. It is also a hundred times cheaper and it cannot hallucinate
# a different collar.

_cutters = {}
_cutter_lock = threading.Lock()
DEFAULT_CUTTER = os.environ.get("PACKSHOT_MODEL", "u2net")

# What rembg will build a session for. Named here so a caller cannot ask the
# pod to fetch something arbitrary, and so the list of what is worth comparing
# is written down rather than remembered.
#
# u2net is what has shipped so far: 2020, trained for salient object detection
# in general, not for product photography. On a hundred real listing
# photographs it cut the background cleanly and kept the coat hanger about a
# third of the time, because a hanger is a salient object.
#
# The birefnet family is the current state of the art for this and rembg
# already carries it, so trying them costs a download and no new code.
CUTTERS = ("u2net", "u2netp", "isnet-general-use", "u2net_cloth_seg",
           "birefnet-general", "birefnet-general-lite", "birefnet-dis",
           "birefnet-massive", "birefnet-hrsod")


def cutter(name=None):
    """A segmentation session, made once per model and reused.

    Built on first use rather than at start-up: the try-on model is what this
    machine is for, and a pod should not spend its first two minutes fetching
    a matting model it may never be asked to run.
    """
    name = name or DEFAULT_CUTTER
    if name not in CUTTERS:
        raise HTTPException(400, f"unknown cutter {name}; one of {CUTTERS}")
    with _cutter_lock:
        if name not in _cutters:
            from rembg import new_session
            t = time.time()
            print(f"[pod] loading the cutout model {name}", flush=True)
            _cutters[name] = new_session(name)
            print(f"[pod] {name} ready in {time.time() - t:.1f}s", flush=True)
    return _cutters[name]


def _correct(img, mask=None):
    """Fix what a phone under a ceiling light gets wrong.

    Not retouching -- nothing is invented. A dim yellow room throws the white
    balance and dulls the contrast, and both are measurable properties of the
    pixels rather than judgements about the garment. Generative retouching was
    tried and does not work in this configuration: the model kept the bedroom
    and turned a navy t-shirt orange. See docs/CONTROLS.md.

    `mask` is which pixels are the garment. It matters more than it sounds: run
    on a cut-out without it, the transparent surround reads as black, those
    blacks own the low end, and the stretch lifts a navy t-shirt to flat grey.
    """
    import numpy as np
    a = np.asarray(img).astype(np.float32)
    if mask is None:
        sel = a.reshape(-1, 3)
    else:
        m = np.asarray(mask) > 24
        sel = a[m]
        if sel.size < 300:            # too little garment to measure anything by
            return img

    # Grey world, damped. A garment is not a grey scene -- a genuinely red
    # jumper should stay red -- so it is applied at half strength, which lifts a
    # colour cast without arguing with the product.
    means = sel.mean(axis=0)
    if means.min() > 1:
        gain = np.clip(means.mean() / means, 0.8, 1.25)
        a *= 1 + (gain - 1) * 0.5
        sel = sel * (1 + (gain - 1) * 0.5)

    # Contrast from the garment's own range, and gently: a dark jumper is
    # supposed to be dark, and stretching it to touch white is how navy became
    # grey the first time this ran.
    lo, hi = np.percentile(sel, 2), np.percentile(sel, 98)
    if hi - lo > 20:
        stretched = (a - lo) * (255.0 / (hi - lo))
        a = a + (stretched - a) * 0.45

    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


# ---------------------------------------------------------------------------
# retouch
#
# The packshot a seller actually wants: the garment alone, straight, on white,
# with the hanger and the room gone. A cut-out cannot do it -- a crooked dress
# cut out is a crooked dress on white -- and everything generative tried before
# this went through the try-on pipeline, which takes three image slots and was
# trained to put a garment on a person. Asked for a product photograph it gave
# back a woman; asked again it turned a multicoloured print flat pink.
#
# The model was never the problem. Qwen-Image-Edit is an instruction editor and
# this is exactly an instruction edit; it was being called the wrong way. Given
# one image and one sentence it removes the hanger, drops the wall, straightens
# the garment and keeps the print. Measured against the cut-out of the same
# dress: colour 0.31, print 0.20, well inside the gate.
#
# It costs no extra VRAM. The editor is assembled from the components the
# try-on model already has loaded -- same weights, same 16 GB -- and the
# adapters come off for the duration, because the try-on LoRA is what makes
# this model deaf to instructions.

RETOUCH_STEPS = int(os.environ.get("RETOUCH_STEPS", "12"))
# Four, with Lightning. Eight was the adapter's design point and four was
# indistinguishable on five garments while taking half as long.
RETOUCH_FAST_STEPS = int(os.environ.get("RETOUCH_FAST_STEPS", "4"))
RETOUCH_SIDE = int(os.environ.get("RETOUCH_SIDE", "768"))
RETOUCH_CFG = float(os.environ.get("RETOUCH_CFG", "4.0"))
# "Keep the garment exactly as it is" was read as keeping the creases too, and
# it did: the dress came back straight and still crumpled off the hanger. What
# a seller needs is the garment pressed -- that is what a catalogue photograph
# is -- with its design untouched. Asking for both gives a fuller, evener skirt
# and scores slightly better on fidelity as well.
# Every detail this names was one the model changed when it was not named.
# "Keep the garment's own design exactly" is not enough: on a blue dress it
# straightened a sweetheart neckline into a bandeau, turned two straps into
# one and lengthened the hem, and every fidelity number stayed inside the
# limits, because those measure colour and texture and a neckline is neither.
#
# Naming them fixed it. The instruction is long and it is long on purpose.
RETOUCH_PROMPT = (
    "Turn this into a professional e-commerce product photograph of the garment "
    "on its own. Remove the hanger, the wall and everything else; plain pure "
    "white background. Press the fabric so it hangs smooth and even. Do not "
    "redesign anything: keep the neckline shape exactly, keep every strap "
    "exactly as many and as wide as they are, keep the hem at exactly the same "
    "length, keep every seam, band and panel where it is, keep the colours and "
    "the fabric including any sheer or mesh layer. Soft even studio lighting, "
    "sharp focus. No person, no mannequin, no hanger.")

_editor = None
_editor_lock = threading.Lock()


def editor():
    """The plain image editor, built from the try-on model's own parts."""
    global _editor
    if _pipe is None:
        raise HTTPException(503, "the model is still loading")
    with _editor_lock:
        if _editor is None:
            from diffusers import QwenImageEditPlusPipeline
            t = time.time()
            print("[pod] assembling the editor from the loaded components",
                  flush=True)
            _pipe._load_text_encoder()
            _editor = QwenImageEditPlusPipeline(
                scheduler=_pipe.noise_scheduler, vae=_pipe.vae,
                text_encoder=_pipe.text_encoder,
                tokenizer=_pipe.processor.tokenizer,
                processor=_pipe.processor, transformer=_pipe.transformer)
            print(f"[pod] editor ready in {time.time() - t:.1f}s", flush=True)
    return _editor


# ---------------------------------------------------------------------------
# presentation
#
# The last step of every packshot, and none of it is generative. Three things
# a seller asked for, and each was a real fault in what shipped before.
#
# **One size.** A garment photographed from across the room came back small in
# the frame and one photographed close came back filling it, so a catalogue was
# a jumble. The old code refused to enlarge -- "blowing a small photograph up
# turns a usable image into a soft one" -- which is true and was the wrong
# call. Consistency is worth more than the last of the sharpness, and the
# honest fix for sharpness is an upscaler, which is a separate job.
#
# **A shadow.** A white t-shirt on a white ground has no edge at all. Every
# catalogue photograph has a contact shadow and this is why. Computed from the
# mask rather than asked of the model: the model puts it in odd places, a
# blurred offset copy of the mask cannot.
#
# **A soft edge.** Tulle, chiffon and mesh are genuinely half-transparent, so
# the alpha ramps rather than switching. Cut hard, a tulle hem loses its shape.

# 1536x2048. Every marketplace a seller here uses wants at least a thousand on
# the short side -- Ozon a thousand square, Wildberries nine hundred by twelve
# and asks for 1200x1600 -- and this clears all of them with room. It is only
# worth asking for because the detail is now really there: without the
# super-resolution step below this would be a megapixel generation stretched
# over three, which is a bigger file and not a better picture.
PRESENT_SIZE = (int(os.environ.get("PRESENT_W", "1536")),
                int(os.environ.get("PRESENT_H", "2048")))
# Off by an env var rather than by editing, because it costs time and the
# trade may look different on a busy day.
UPSCALE = os.environ.get("PACKSHOT_UPSCALE", "1") not in ("", "0", "false")
PRESENT_FILL = float(os.environ.get("PRESENT_FILL", "0.88"))
PRESENT_SHADOW = float(os.environ.get("PRESENT_SHADOW", "0.30"))


def _ground_alpha(img, slack=20.0):
    """Where the garment is, on a picture that already has a clean ground."""
    import numpy as np
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    m = max(2, int(min(h, w) * 0.04))
    edge = np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3),
                           a[:, :m].reshape(-1, 3), a[:, -m:].reshape(-1, 3)])
    ground = edge.mean(axis=0)
    dist = np.linalg.norm(a - ground, axis=2)
    return np.clip((dist - slack) / slack, 0.0, 1.0)


def _present(img, alpha=None, shadow=PRESENT_SHADOW, size=None, fill=None):
    import numpy as np
    from PIL import ImageFilter

    size = size or PRESENT_SIZE
    fill = PRESENT_FILL if fill is None else fill
    a = _ground_alpha(img) if alpha is None else alpha
    ys, xs = np.nonzero(a > 0.25)
    if len(ys) < 50:
        return img.convert("RGB")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    rgb = img.convert("RGB").crop((int(x0), int(y0), int(x1), int(y1)))
    mask = Image.fromarray((a[y0:y1, x0:x1] * 255).astype("uint8"))
    scale = min(size[0] * fill / rgb.width, size[1] * fill / rgb.height)
    wide = (max(8, int(rgb.width * scale)), max(8, int(rgb.height * scale)))
    rgb = rgb.resize(wide, Image.LANCZOS)
    mask = mask.resize(wide, Image.LANCZOS)

    out = Image.new("RGB", size, (255, 255, 255))
    left, top = (size[0] - wide[0]) // 2, (size[1] - wide[1]) // 2
    if shadow > 0:
        blur = max(6, int(min(wide) * 0.045))
        soft = mask.filter(ImageFilter.GaussianBlur(blur))
        soft = Image.fromarray(
            (np.asarray(soft).astype(np.float32) * shadow).astype("uint8"))
        out.paste(Image.new("RGB", wide, (120, 120, 125)),
                  (left + int(blur * 0.35), top + int(blur * 0.9)), soft)
    out.paste(rgb, (left, top), mask)
    return out


@app.post("/retouch")
def retouch(garment: UploadFile = File(...),
            instruction: str = Form(""),
            steps: int = Form(0), seed: int = Form(11),
            side: int = Form(0), fast: str = Form("")):
    """`side` is the long edge the edit runs at.

    It is a parameter because it is a trade and the trade is not settled: a
    seller's listing wants detail, and every pixel is time and VRAM on a card
    that already holds two models.
    """
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    import torch

    img = _read(garment, "garment")
    # Resized, not thumbnailed. thumbnail() only ever shrinks, so a seller's
    # 400x711 listing thumbnail stayed 400x711 and the edit ran at whatever the
    # model chose -- and the print came back muddy because there was nothing to
    # work from. Small inputs are the common case here, so they are enlarged to
    # the working size first and the model reconstructs the detail.
    long_edge = max(int(side) or RETOUCH_SIDE, 512)
    scale = long_edge / max(img.size)
    if abs(scale - 1.0) > 0.02:
        img = img.resize((max(64, int(img.width * scale)),
                          max(64, int(img.height * scale))), Image.LANCZOS)
    pipe = editor()
    started = time.time()
    with _gpu:
        # The try-on adapter reads the garment image and ignores text, which is
        # the opposite of what is wanted here. Lightning goes too: it is
        # distilled for guidance 1.0, and with no negative branch nothing pulls
        # the picture toward the instruction.
        # Two ways to take the try-on adapter off, and they cost very
        # differently.
        #
        # The slow one turns everything off, including Lightning, and pays for
        # guidance: twelve steps, each run twice -- once with the instruction
        # and once without -- because the difference between those two passes
        # is what pulls the picture toward the words. Fifty-two seconds.
        #
        # The fast one keeps Lightning, which is distilled to work in eight
        # steps with no second pass at all. A sixth of the arithmetic. Whether
        # the instruction still lands is the open question: with the try-on
        # adapter *and* Lightning the model ignored text entirely, but that was
        # measured through the try-on pipeline and with the adapter that makes
        # this model a garment compositor. Lightning alone, in the plain
        # editor, has never been tried.
        # Fast by default, and that is a measurement rather than a preference.
        # On the same two dresses the Lightning path came back as good and on
        # the flowered one better -- colour 0.35 at four steps against 0.60 at
        # twelve with guidance -- in a quarter of the time. Checked again on a
        # white tee, a lace shirt and a skirt: no visible loss. The undistilled
        # path is still there behind fast=0.
        want_fast = str(fast).lower() not in ("0", "false", "no")
        scale = getattr(_pipe, "lightning_scale", 1.0)
        toggled = None
        if want_fast and getattr(_pipe, "lightning", 0) and                 hasattr(pipe.transformer, "set_adapters"):
            pipe.transformer.set_adapters(["lightning"], [scale])
            toggled = "lightning"
        elif hasattr(pipe.transformer, "disable_adapters"):
            pipe.transformer.disable_adapters()
            toggled = "all"
        try:
            # The size is *not* asked for, and that was measured the hard
            # way. Forcing it made the picture worse: at 704x1280 and 576x1024
            # the print smeared, the leaf motifs ran together, while the
            # pipeline's own choice -- 768x1376, about a megapixel -- came back
            # with every motif separate. It picks a shape the model was trained
            # on, and a different one is off the distribution.
            out = pipe(image=[img],
                       prompt=(instruction or "").strip() or RETOUCH_PROMPT,
                       num_inference_steps=(int(steps) or
                                            (RETOUCH_FAST_STEPS if want_fast
                                             else RETOUCH_STEPS)),
                       true_cfg_scale=1.0 if want_fast else RETOUCH_CFG,
                       negative_prompt=" ",
                       generator=torch.Generator("cuda").manual_seed(int(seed))
                       ).images[0]
        except Exception as exc:                              # noqa: BLE001
            _stats["failed"] += 1
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
        finally:
            if toggled == "lightning":
                pipe.transformer.set_adapters(["default", "lightning"],
                                              [1.0, scale])
            elif toggled == "all":
                pipe.transformer.enable_adapters()
    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed

    # Twice the size before it is framed, so the garment is built from real
    # detail rather than from a megapixel stretched over three.
    if UPSCALE:
        try:
            from .upscale import upscale as _up
            t = time.time()
            out = _up(out)
            print(f"[pod] upscaled to {out.size} in {time.time() - t:.1f}s",
                  flush=True)
        except Exception as exc:                              # noqa: BLE001
            # A packshot without the extra detail is still a packshot.
            print(f"[pod] upscale skipped: {type(exc).__name__}: {exc}",
                  flush=True)
    out = _present(out)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Seconds": str(elapsed), "X-Width": str(out.width),
                             "X-Height": str(out.height)})


@app.post("/packshot")
def packshot(garment: UploadFile = File(...),
             background: str = Form("#FFFFFF"),
             correct: str = Form("1"),
             width: int = Form(0), height: int = Form(0),
             model: str = Form("")):
    from rembg import remove

    img = _read(garment, "garment")
    size = (width or PACKSHOT_SIZE[0], height or PACKSHOT_SIZE[1])
    started = time.time()
    try:
        cut = remove(img, session=cutter(model or None))
    except Exception as exc:                                  # noqa: BLE001
        _stats["failed"] += 1
        raise HTTPException(500, f"could not separate the garment: "
                                 f"{type(exc).__name__}: {exc}")

    # Trim to what is actually left. A photograph of a jumper on a large table
    # is mostly table; centring the original frame would centre the table.
    box = cut.getbbox()
    if box:
        cut = cut.crop(box)

    # Corrected after the cut, not before. Correcting first brightens the room
    # along with the garment, which makes the clutter beside it look more like
    # foreground -- a bag and a phone that the cut-out had dropped came back.
    # Afterwards the correction sees the garment and nothing else.
    if correct != "0":
        alpha = cut.getchannel("A")
        rgb = _correct(cut.convert("RGB"), mask=alpha)
        rgb.putalpha(alpha)
        cut = rgb

    # The cut-out carries its own alpha, which beats anything read back off a
    # rendered ground. The rest -- one size for every garment, and a shadow so
    # a white shirt on white has an edge -- is the same presentation the
    # retouched version gets, so the two are never a different shape.
    import numpy as np
    canvas = _present(
        cut.convert("RGB"),
        alpha=np.asarray(cut.getchannel("A")).astype(np.float32) / 255.0,
        size=size)

    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Seconds": str(elapsed),
                             "X-Width": str(size[0]), "X-Height": str(size[1])})


# ---------------------------------------------------------------------------
# enhance
#
# A packshot is not only a cut-out. A seller photographs a jumper on a bed
# under a ceiling light and wants back something that belongs in a catalogue:
# creases gone, colour honest, lighting even. Segmentation cannot do that.
#
# The base model can. Underneath the try-on adapter is Qwen-Image-Edit, an
# instruction-following editor -- the adapter is what makes it a garment
# compositor and deaf to everything else. So this turns the adapter off, sends
# the garment in all three image slots, and gives the model a plain instruction
# instead of the try-on template.

ENHANCE_PROMPT = (
    "Retouch this garment into a clean catalogue product photograph. Remove "
    "creases and wrinkles, even out the lighting, keep the colour, the fabric "
    "texture, the cut and every detail exactly as they are. Do not change the "
    "shape of the garment and do not add or remove any part of it."
)


@app.post("/enhance")
def enhance(garment: UploadFile = File(...),
            instruction: str = Form(""),
            seed: int = Form(42),
            steps: int = Form(0),
            fast: str = Form("0")):
    """fast=1 keeps Lightning: eight steps, and the instruction is ignored.

    It is left reachable only because it is the honest comparison. The first
    version of this endpoint ran that way by default and returned the picture
    it was given, wall and coat hook included.
    """
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    if _pipe is None:
        raise HTTPException(503, "the model is still loading")

    from utils import pad_to_aspect_ratio

    img = _read(garment, "garment")
    # The same picture in every slot: the model is being asked to edit one
    # image, and the three-slot shape is the pipeline's, not the task's.
    padded = pad_to_aspect_ratio(img, target_size=(512, 896), pad_color=(255, 255, 255))

    quick = str(fast).lower() in ("1", "true", "yes")
    started = time.time()
    with _gpu:
        try:
            out = _pipe(person_img=padded, garment_img=padded, pose_img=padded,
                        description="", mode=None, seed=int(seed),
                        num_inference_steps=int(steps) or None,
                        raw_prompt=instruction.strip() or ENHANCE_PROMPT,
                        adapters=False, lightning=quick)
        except Exception as exc:                              # noqa: BLE001
            _stats["failed"] += 1
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed

    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Seconds": str(elapsed),
                             "X-Width": str(out.width), "X-Height": str(out.height)})



def main():
    import uvicorn
    port = int(os.environ.get("POD_SERVER_PORT", "8000"))
    host = os.environ.get("POD_SERVER_HOST", "127.0.0.1")
    if host != "127.0.0.1":
        print(f"[pod] WARNING: binding {host}, not loopback. This machine has "
              "a public address and this endpoint has no authentication.",
              flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
