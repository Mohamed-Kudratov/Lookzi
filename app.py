import argparse
import os
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import LookziPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="Lookzi virtual try-on demo.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="hf_models/stable-diffusion-inpainting",  # Local download avoids Windows symlink privileges.
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="hf_models/sd-vae-ft-mse",
        help="The path or model identifier for the VAE checkpoint.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="hf_models/lookzi-vton",
        help=(
            "The path to the virtual try-on model files."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run the demo on. Auto uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link. Use this in Colab.",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="127.0.0.1",
        help="Host for the Gradio server. Use 0.0.0.0 when needed in hosted notebooks.",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Port for the Gradio server.",
    )
    parser.add_argument(
        "--enable_safety_checker",
        action="store_true",
        help="Enable the inherited NSFW safety checker. Disabled by default to avoid false positives on try-on outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="",
        help="Root path for reverse proxy (e.g. Colab proxy URL). Fixes static file URLs.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "Input and output width used by the generation pipeline."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "Input and output height used by the generation pipeline."
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up generation. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10 and an Nvidia Ampere GPU."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def infer_garment_style(cloth_image: Image.Image, cloth_type: str) -> tuple:
    """Returns (style_str, covers_lower_legs: bool, debug_info_str)."""
    image = np.array(cloth_image.convert("RGB").resize((256, 256))).astype(np.int16)
    edge_pixels = np.concatenate([image[:8].reshape(-1, 3), image[-8:].reshape(-1, 3), image[:, :8].reshape(-1, 3), image[:, -8:].reshape(-1, 3)])
    bg = np.median(edge_pixels, axis=0)
    distance = np.linalg.norm(image - bg, axis=2)
    foreground = distance > 28
    foreground[:4, :] = False
    foreground[-4:, :] = False
    foreground[:, :4] = False
    foreground[:, -4:] = False
    ys, xs = np.where(foreground)
    if len(xs) < 100:
        return "auto", False, "fg_pixels<100 → auto"

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h = max(1, y1 - y0 + 1)
    w = max(1, x1 - x0 + 1)

    # Garment covers lower legs if it's tall enough (pants, maxi dress, jumpsuit)
    covers_lower_legs = (h / 256.0) > 0.65

    def band_width(start, end):
        yy0 = y0 + int(h * start)
        yy1 = y0 + int(h * end)
        band = foreground[yy0:yy1 + 1]
        by, bx = np.where(band)
        if len(bx) < 10:
            return 0
        return bx.max() - bx.min() + 1

    shoulder_w = band_width(0.12, 0.32)
    chest_w = band_width(0.34, 0.56)
    hem_w = band_width(0.72, 0.95)
    shoulder_ratio = shoulder_w / max(chest_w, 1)
    hem_ratio = hem_w / max(chest_w, 1)

    # Upper side mass: shoulder/upper-arm zone (y 12%–38%)
    upper = foreground[y0 + int(h * 0.12):y0 + int(h * 0.38), x0:x1 + 1]
    side_band = max(1, int(w * 0.25))
    left_mass = upper[:, :side_band].mean() if upper.size else 0
    right_mass = upper[:, -side_band:].mean() if upper.size else 0
    side_mass = (left_mass + right_mass) / 2

    # Lower side mass: elbow/forearm zone (y 38%–65%)
    # Sleeve fabric must extend here; wide sleeveless bodies do not
    lower = foreground[y0 + int(h * 0.38):y0 + int(h * 0.65), x0:x1 + 1]
    lower_left = lower[:, :side_band].mean() if lower.size else 0
    lower_right = lower[:, -side_band:].mean() if lower.size else 0
    lower_side_mass = (lower_left + lower_right) / 2

    dbg = (f"side={side_mass:.3f} lower_side={lower_side_mass:.3f} "
           f"sho_ratio={shoulder_ratio:.3f} hem_ratio={hem_ratio:.3f} "
           f"covers_legs={covers_lower_legs}")

    # Long sleeve: fabric on sides in BOTH upper AND lower zone
    if side_mass > 0.55 and lower_side_mass > 0.42:
        return "sleeved", covers_lower_legs, dbg + " → sleeved"
    if cloth_type == "overall" and hem_ratio > 0.75 and h > w * 1.15:
        style = "sleeveless" if side_mass < 0.48 else "sleeved"
        return style, covers_lower_legs, dbg + f" → {style}(overall+hem)"
    if shoulder_ratio < 1.15 and side_mass < 0.52:
        return "sleeveless", covers_lower_legs, dbg + " → sleeveless"
    if shoulder_ratio > 1.55 or lower_side_mass > 0.35:
        return "sleeved", covers_lower_legs, dbg + " → sleeved(ratio/lower)"
    return "short_sleeve", covers_lower_legs, dbg + " → short_sleeve"


args = parse_args()
device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
if device == "auto":
    device = "cpu"
if device == "cpu" and args.mixed_precision != "no":
    print("CPU detected; switching mixed precision to 'no'.")
    args.mixed_precision = "no"
print(f"Running Lookzi on {device} with mixed_precision={args.mixed_precision}")
repo_path = args.resume_path if os.path.exists(args.resume_path) else snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = LookziPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device=device,
    vae_ckpt=args.vae_model_path,
    skip_safety_check=not args.enable_safety_checker,
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device, 
)

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    mask_source,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    if person_image is None or cloth_image is None:
        raise gr.Error("Iltimos, inson rasmi va kiyim rasmini yuklang.")

    person_image_path = person_image["background"]
    if not person_image_path:
        raise gr.Error("Iltimos, inson rasmini yuklang.")

    layers = person_image.get("layers") or []
    mask = None
    if mask_source == "manual" and layers:
        mask = Image.open(layers[0]).convert("L")
    if mask is not None and len(np.unique(np.array(mask))) > 1:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)
    else:
        mask = None

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)

    person_image = Image.open(person_image_path).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    garment_style, covers_lower_legs, style_debug = infer_garment_style(cloth_image, cloth_type)
    print(f"Detected garment_style={garment_style} covers_lower_legs={covers_lower_legs} | {style_debug}")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
        mask_array = np.array(mask) > 0
        if mask_array.mean() > 0.70:
            gr.Warning("Manual mask is too large. Lookzi used automatic masking instead.")
            mask = automasker(
                person_image,
                cloth_type,
                garment_style=garment_style,
                covers_lower_legs=covers_lower_legs,
            )['mask']
    else:
        mask = automasker(
            person_image,
            cloth_type,
            garment_style=garment_style,
            covers_lower_legs=covers_lower_legs,
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    # try:
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    # except Exception as e:
    #     raise gr.Error(
    #         "An error occurred. Please try again later: {}".format(e)
    #     )
    
    # Post-process
    masked_person = vis_mask(person_image, mask)
    mask_array = np.array(mask) > 127
    coverage = float(mask_array.mean() * 100)
    debug_text = (
        f"Garment style: {garment_style} | covers_legs: {covers_lower_legs}\n"
        f"Style detection: {style_debug}\n"
        f"Clothing type: {cloth_type}\n"
        f"Mask coverage: {coverage:.2f}%\n"
        f"Resolution: {args.width}x{args.height}\n"
        f"Steps: {num_inference_steps}\n"
        f"CFG: {guidance_scale}\n"
        f"Seed: {seed}"
    )
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return result_image, mask, masked_person, debug_text
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
    return new_result_image, mask, masked_person, debug_text


def person_example_fn(image_path):
    return image_path

HEADER = """
<h1 style="text-align: center;">Lookzi Virtual Try-On Studio</h1>
<p style="text-align: center; color: #667085; margin-top: -6px;">
Upload a person image, choose a garment, select the target clothing area, and generate a try-on preview.
</p>
"""
def app_gradio():
    with gr.Blocks(title="Lookzi", css="footer{display:none !important}") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path = gr.Image(
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                    person_image = gr.ImageEditor(
                        interactive=True, label="Person Image", type="filepath", image_mode="RGB", format="png"
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Lookzi automatically detects the target garment area. Choose the clothing type that best matches the garment.</span>'
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )
                        mask_source = gr.State("auto")


                submit = gr.Button("Submit")
                gr.Markdown(
                    '<center><span style="color: #FF0000">Click once and wait for generation to finish.</span></center>'
                )
                
                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">Advanced options control detail, color strength, and repeatable variations.</span>'
                )
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=50
                    )
                    # Guidence Scale
                    guidance_scale = gr.Slider(
                        label="CFG Strength", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                    )
                    # Random Seed
                    seed = gr.Slider(
                        label="Seed", minimum=-1, maximum=10000, step=1, value=42
                    )
                    show_type = gr.Radio(
                        label="Show Type",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="result only",
                    )

            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(interactive=False, label="Result")
                with gr.Accordion("Developer Debug", open=False):
                    debug_info = gr.Textbox(label="Diagnostics", interactive=False, lines=7)
                    with gr.Row():
                        debug_mask = gr.Image(interactive=False, label="Auto Mask")
                        debug_masked_person = gr.Image(interactive=False, label="Masked Person")
                with gr.Row():
                    # Photo Examples
                    root_path = "resource/demo/example"
                    with gr.Column():
                        men_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "men", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "men"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples",
                        )
                        women_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "women", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "women"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="More Person Examples",
                        )
                    with gr.Column():
                        condition_upper_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "upper", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "upper"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Upper Garments",
                        )
                        condition_overall_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "overall", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "overall"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Full Outfit Garments",
                        )
                        condition_person_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "person", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "person"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Reference Garments",
                        )

            image_path.change(
                person_example_fn, inputs=image_path, outputs=person_image
            )

            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    mask_source,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                [result_image, debug_mask, debug_masked_person, debug_info],
            )
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        show_api=False,
        root_path=args.root_path,
    )


if __name__ == "__main__":
    app_gradio()

