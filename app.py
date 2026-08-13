import os
import gc
import traceback

import gradio as gr
import torch

from utils import process_inputs
from pipeline import LayeringVTONPipeline, detect_device, total_vram_gb

# The full bf16 model: 57.7 GB, and what you are paying a GPU host for.
# Under ~40 GB of VRAM set MODEL_PATH=ovedrive/Qwen-Image-Edit-2509-4bit
# instead (~17 GB, visibly lossy). runpod_setup.sh picks this by VRAM.
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509")
LORA_DIR = os.environ.get("LORA_DIR", "./weights")
SHARE = os.environ.get("GRADIO_SHARE", "0") == "1"

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = LayeringVTONPipeline(MODEL_PATH, LORA_DIR)
    return pipeline


def run_vton(person_img, garment_img, custom_pose_img, mode, description,
             steps, cfg, seed, progress=gr.Progress()):
    if person_img is None or garment_img is None:
        raise gr.Error("Please upload both a person image and a garment image.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress(0, desc="Extracting pose and padding inputs...")
    padded_person, padded_garment, padded_pose = process_inputs(person_img, garment_img, custom_pose_img)

    # Show the processed inputs straight away; the model load and sampling that
    # follow take minutes on a T4.
    yield padded_person, padded_garment, padded_pose, None

    try:
        progress(0, desc="Loading model (first run downloads the weights)...")
        pl = get_pipeline()

        def on_step(done, total):
            progress(done / total, desc=f"Sampling step {done}/{total}")

        result_img = pl(
            person_img=padded_person,
            garment_img=padded_garment,
            pose_img=padded_pose,
            description=description,
            mode=mode,
            num_inference_steps=int(steps),
            true_cfg_scale=float(cfg),
            seed=int(seed),
            progress_callback=on_step,
        )
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        raise gr.Error(
            "Out of VRAM. Lower the steps, or set CFG to 1.0 (halves the work by "
            "skipping the negative pass), or switch to a larger GPU runtime."
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"{type(exc).__name__}: {exc}")

    yield padded_person, padded_garment, padded_pose, result_img


with gr.Blocks(title="Layering VTON Demo") as demo:
    gr.Markdown("# Layering VTON")
    gr.Markdown("Upload a person image and a garment image, then select the mode to run the virtual try-on.")

    if torch.cuda.is_available():
        gr.Markdown(
            f"**GPU:** {torch.cuda.get_device_name(0)} ({total_vram_gb():.1f} GB) &nbsp;|&nbsp; "
            f"**Model:** `{MODEL_PATH}`"
        )
    else:
        gr.Markdown("**No GPU detected.** This is a 20B model and will not run on CPU.")

    with gr.Row():
        with gr.Column():
            person_in = gr.Image(type="pil", label="Person Image (Original)")
            garment_in = gr.Image(type="pil", label="Garment Image (Original)")
            pose_in = gr.Image(type="pil", label="Custom Pose Image (Optional)")

            mode_in = gr.Radio(["swap", "add"], label="Mode", value="swap")
            desc_in = gr.Textbox(label="Description", value="swap the beige leggings for dark wash jeans")

            with gr.Accordion("Sampling settings", open=False):
                steps_in = gr.Slider(8, 50, value=40, step=1, label="Inference steps")
                cfg_in = gr.Slider(1.0, 7.0, value=4.0, step=0.5,
                                   label="True CFG scale (1.0 = single pass, ~2x faster, lower fidelity)")
                seed_in = gr.Number(value=42, precision=0, label="Seed")

            run_btn = gr.Button("Run VTON", variant="primary")

            gr.Examples(
                examples=[
                    ["assets/person_1.png", "assets/pants.png",   None, "swap", "swap the deep blue jeans for dark wash jeans"],
                    ["assets/person_2.png", "assets/sweater.png", None, "add",  "add a light gray turtleneck sweater"],
                    ["assets/person_3.png", "assets/coat.png",    None, "add",  "add a black leather jacket"],
                ],
                inputs=[person_in, garment_in, pose_in, mode_in, desc_in]
            )

        with gr.Column():
            gr.Markdown("### Intermediate Processing")
            with gr.Row():
                person_out = gr.Image(type="pil", label="Processed Person (512x896)")
                garment_out = gr.Image(type="pil", label="Processed Garment (512x896)")
                pose_out = gr.Image(type="pil", label="Processed/Extracted Pose (512x896)")

            gr.Markdown("### Output")
            result_out = gr.Image(type="pil", label="Generated Image")

    run_btn.click(
        fn=run_vton,
        inputs=[person_in, garment_in, pose_in, mode_in, desc_in, steps_in, cfg_in, seed_in],
        outputs=[person_out, garment_out, pose_out, result_out]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=SHARE)
