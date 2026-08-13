import os
import gc
import torch
import numpy as np
import copy
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image

try:
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageTransformer2DModel,
    )
except ImportError as exc:
    # `diffusers/` in the repo root is the fork's repository, not its package --
    # the package is at diffusers/src/diffusers. Python searches the working
    # directory first, resolves `diffusers` to that directory, finds no
    # __init__.py, and returns an empty PEP 420 namespace package. The tell is
    # "(unknown location)" in the ImportError.
    import diffusers as _d
    if getattr(_d, "__file__", None) is None:
        raise ImportError(
            "`diffusers` resolved to the local ./diffusers directory instead of the "
            "installed package, so it has no contents.\n\n"
            "Fix:\n"
            "    pip install ./diffusers        # not -e: an editable install is still shadowed\n"
            "    mv diffusers diffusers_src     # stop the directory shadowing the package\n"
            "then restart the Python process."
        ) from exc
    raise
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
from peft import set_peft_model_state_dict, LoraConfig
from safetensors.torch import load_file
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
import inspect


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
# The upstream code hardcodes device="cuda" and torch.bfloat16. That is correct
# for an A100 but breaks on any pre-Ampere card -- Turing (T4, RTX 20xx) has no
# native bfloat16. These helpers pick a working configuration rather than
# assuming one.

def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def total_vram_gb(device="cuda"):
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(torch.device(device).index or 0).total_memory / 1024**3


def detect_dtype(device="cuda"):
    """bfloat16 where it is native (Ampere+), float16 otherwise (T4)."""
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _is_quantized(model):
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "is_quantized", False)
    )


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bnb_override(dtype, library):
    """Force the bnb compute dtype to match the GPU.

    The published 4-bit repo bakes `bnb_4bit_compute_dtype: bfloat16` into its
    config. A T4 is Turing and has no native bf16, so honouring that config
    makes every 4-bit matmul crawl. Passing an explicit quantization_config
    overrides the checkpoint's own. Only needed when we are running fp16.
    """
    if dtype != torch.float16:
        return None
    try:
        if library == "diffusers":
            from diffusers import BitsAndBytesConfig
        else:
            from transformers import BitsAndBytesConfig
    except ImportError:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _load_model(cls, path, subfolder, dtype, device, library="diffusers", **kwargs):
    """Load a model, placing it correctly whether or not it is bnb-quantized.

    A bitsandbytes-quantized module cannot be relocated with `.to(device)` --
    its Params4bit storage is bound to the device it was quantized onto -- so
    quantized repos must be placed via device_map at load time. Non-quantized
    repos ignore device_map on some versions, hence the fallback.
    """
    qc = _bnb_override(dtype, library)
    attempts = []
    if qc is not None:
        attempts.append(dict(device_map={"": device}, quantization_config=qc))
    attempts.append(dict(device_map={"": device}))
    attempts.append(dict())

    last_exc = None
    for i, extra in enumerate(attempts):
        try:
            model = cls.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype, **extra, **kwargs)
            if "device_map" not in extra:
                if _is_quantized(model):
                    raise RuntimeError(
                        "Quantized checkpoint loaded without a device_map; it cannot be moved to the GPU."
                    )
                model = model.to(device)
            return model
        except Exception as exc:  # noqa: BLE001 - fall through to the next strategy
            last_exc = exc
            if i < len(attempts) - 1:
                print(f"  load strategy {i + 1} failed ({type(exc).__name__}: {exc}); retrying")
    raise last_exc


# ---------------------------------------------------------------------------
# Unchanged numerical helpers from the original implementation
# ---------------------------------------------------------------------------

def compute_text_embeddings(
    prompt: list,
    image: list,
    device,
    dtype,
    max_sequence_length: int,
    processor,
    text_encoder,
    num_images_per_prompt: int = 1,
):
    assert isinstance(image, list), "The \"image\" should be a list of images."
    template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 64
    img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    base_img_prompt = ""
    for i, img in enumerate(image):
        base_img_prompt += img_prompt_template.format(i + 1)

    with torch.no_grad():
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        txt = [template.format(base_img_prompt + e) for e in prompt]
        model_inputs = processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
            bool_mask = mask.bool()
            valid_lengths = bool_mask.sum(dim=1)
            selected = hidden_states[bool_mask]
            split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
            return split_result

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = _extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])

        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        ).to(dtype=dtype, device=device)
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        ).to(device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

    return prompt_embeds, prompt_embeds_mask


def compute_image_tokens_by_vae(
    pixel_values,
    vae,
    latents_mean,
    latents_std,
    device,
    weight_dtype,
):
    with torch.no_grad():
        pixel_values = pixel_values.to(device=device, dtype=vae.dtype)
        pixel_values = pixel_values.unsqueeze(dim=2)
        latents = vae.encode(pixel_values).latent_dist.mode()
        latents = (latents - latents_mean) / latents_std
        latents = latents.to(dtype=weight_dtype)
    return latents


def _pack_latents(
    latents,
    batch_size,
    num_channels_latents,
    height,
    width
):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def _unpack_latents(
    latents,
    height,
    width,
    vae_scale_factor
):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
    return latents


def _prepare_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    vae_scale_factor,
    dtype,
    device,
    generator,
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    shape = (batch_size, 1, num_channels_latents, height, width)

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)
    return latents


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps,
    device,
    sigmas,
    **kwargs,
):
    accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    if not accept_sigmas:
        raise ValueError(
            f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
            f" sigmas schedules. Please check whether you are using the correct scheduler."
        )
    scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)
    return timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# Mode handling
# ---------------------------------------------------------------------------
# The Gradio app collects a swap/add radio button but the original code never
# passed it anywhere, so the control silently did nothing -- the mode was only
# ever implied by how the user happened to word the description. This keeps the
# description as the source of truth but makes the radio actually bind to it.

def apply_mode(mode: str, description: str) -> str:
    desc = (description or "").strip()
    if not mode:
        return desc
    mode = mode.strip().lower()
    if not desc:
        return desc
    # If the description already opens with either verb, it wins -- prefixing
    # here would produce prompts like "add swap the jeans for shorts".
    lowered = desc.lower()
    if lowered.startswith("add") or lowered.startswith("swap"):
        return desc
    if mode in ("add", "swap"):
        return f"{mode} {desc}"
    return desc


class LayeringVTONPipeline:
    """Layering VTON sampler.

    Differences from the reference implementation, all driven by not assuming an
    80 GB Ampere card:

    * device / dtype are detected instead of hardcoded to cuda + bfloat16
    * bitsandbytes-quantized checkpoints are supported (a 4-bit repo is ~17 GB
      against 57.7 GB for the bf16 original)
    * `low_vram` loads the text encoder and the transformer one at a time, for
      cards where 40.9 + 16.6 GB does not fit at once
    * the VAE runs in fp32 when the rest of the model is fp16, and the CFG
      renormalisation is done in fp32, to avoid overflow on Turing
    """

    def __init__(
        self,
        pretrained_model_name_or_path,
        lora_weights_dir,
        device=None,
        dtype=None,
        low_vram=None,
        vae_dtype=None,
        lora_rank=32,
    ):
        self.model_path = pretrained_model_name_or_path
        self.lora_weights_dir = lora_weights_dir
        self.lora_rank = lora_rank

        self.device = device or detect_device()
        if self.device == "cpu":
            raise RuntimeError(
                "No CUDA GPU detected. This is a 20B-parameter diffusion model; it cannot run on CPU.\n"
                "Check `nvidia-smi` and that torch was built with CUDA."
            )

        self.weight_dtype = dtype or detect_dtype(self.device)
        # Qwen's VAE overflows in fp16; keep it in fp32 when the rest is fp16.
        self.vae_dtype = vae_dtype or (
            torch.float32 if self.weight_dtype == torch.float16 else self.weight_dtype
        )

        vram = total_vram_gb(self.device)
        if low_vram is None:
            # transformer (~11.6 GB) + text encoder (~5.1 GB) at 4-bit needs
            # ~17 GB resident; anything under ~20 GB has to load sequentially.
            low_vram = vram < 20.0
        self.low_vram = low_vram

        print(f"Device: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
        print(f"Compute dtype: {self.weight_dtype}, VAE dtype: {self.vae_dtype}")
        print(f"Low-VRAM sequential loading: {self.low_vram}")

        print("Loading noise scheduler...")
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            shift=3.0
        )

        print("Loading VAE...")
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
        ).to(self.device)
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device)
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device)
        self.vae.requires_grad_(False)

        print("Loading processor...")
        self.processor = Qwen2VLProcessor.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="processor",
        )

        self.text_encoder = None
        self.transformer = None
        self._embed_cache = {}

        if not self.low_vram:
            self._load_text_encoder()
            self._load_transformer()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # -- component lifecycle ------------------------------------------------

    def _load_text_encoder(self):
        if self.text_encoder is not None:
            return
        print("Loading text encoder...")
        self.text_encoder = _load_model(
            Qwen2_5_VLForConditionalGeneration,
            self.model_path,
            "text_encoder",
            self.weight_dtype,
            self.device,
            library="transformers",
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    def _unload_text_encoder(self):
        if self.text_encoder is None:
            return
        print("Unloading text encoder to free VRAM...")
        del self.text_encoder
        self.text_encoder = None
        _free()

    def _load_transformer(self):
        if self.transformer is not None:
            return
        print("Loading transformer...")
        self.transformer = _load_model(
            QwenImageTransformer2DModel,
            self.model_path,
            "transformer",
            self.weight_dtype,
            self.device,
        )

        print(f"Loading LoRA weights from {self.lora_weights_dir}...")
        lora_state_dict = QwenImageEditPlusPipeline.lora_state_dict(
            load_file(os.path.join(self.lora_weights_dir, "pytorch_lora_weights.safetensors"))
        )
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        # LoRA tensors ship as fp32; match the compute dtype so PEFT does not
        # silently upcast every adapter on a quantized base model.
        transformer_state_dict = {
            k: v.to(self.weight_dtype) for k, v in transformer_state_dict.items()
        }
        transformer_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.transformer.add_adapter(transformer_lora_config)

        incompatible_keys = set_peft_model_state_dict(self.transformer, transformer_state_dict, adapter_name="default")
        if incompatible_keys and incompatible_keys.unexpected_keys:
            print(f"Warning: Loading adapter weights led to unexpected keys: {incompatible_keys.unexpected_keys}.")
        else:
            print("Successfully loaded LoRA weights.")

        self.transformer.requires_grad_(False)
        self.transformer.eval()
        _free()

    def _unload_transformer(self):
        if self.transformer is None:
            return
        print("Unloading transformer to free VRAM...")
        del self.transformer
        self.transformer = None
        _free()

    # -- sampling -----------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        person_img: Image.Image,
        garment_img: Image.Image,
        pose_img: Image.Image,
        description: str,
        mode: str = None,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = None,
        seed: int = 42,
        progress_callback=None,
    ):
        description = apply_mode(mode, description)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        negative_prompt = " "
        has_neg_prompt = True
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        batch_size = 1

        img_height, img_width = person_img.size[1], person_img.size[0]

        src_tensor = self.transform(person_img).unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        garment_tensor = self.transform(garment_img).unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        pose_tensor = self.transform(pose_img).unsqueeze(0).to(self.device, dtype=self.weight_dtype)

        img_shapes = [
            [
                (1, img_height // self.vae_scale_factor // 2, img_width // self.vae_scale_factor // 2),
                (1, img_height // self.vae_scale_factor // 2, img_width // self.vae_scale_factor // 2),
                (1, img_height // self.vae_scale_factor // 2, img_width // self.vae_scale_factor // 2),
                (1, img_height // self.vae_scale_factor // 2, img_width // self.vae_scale_factor // 2),
            ]
        ]

        full_description = f"Edit the person in the first image based on the garment in the second image: {description.lower()} And change the pose of the person in the first image to the pose in the third image"

        # --- stage 1: VAE encoding (VAE is small enough to stay resident) ---
        pose_image_latents = compute_image_tokens_by_vae(
            pixel_values=pose_tensor, vae=self.vae, latents_mean=self.latents_mean, latents_std=self.latents_std,
            device=self.device, weight_dtype=self.weight_dtype
        )
        person_image_latents = compute_image_tokens_by_vae(
            pixel_values=src_tensor, vae=self.vae, latents_mean=self.latents_mean, latents_std=self.latents_std,
            device=self.device, weight_dtype=self.weight_dtype
        )
        garment_image_latents = compute_image_tokens_by_vae(
            pixel_values=garment_tensor, vae=self.vae, latents_mean=self.latents_mean, latents_std=self.latents_std,
            device=self.device, weight_dtype=self.weight_dtype
        )

        # --- stage 2: text encoding ---
        # Re-running with a different seed or step count on unchanged inputs is
        # common, and on a T4 the text-encoder stage costs a full model load, so
        # the embeddings are cached against the inputs that produced them.
        cache_key = (
            full_description,
            hash(person_img.tobytes()),
            hash(garment_img.tobytes()),
            hash(pose_img.tobytes()),
            do_true_cfg,
        )
        if cache_key in self._embed_cache:
            print("Reusing cached text embeddings.")
            prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask = self._embed_cache[cache_key]
        else:
            if self.low_vram:
                self._unload_transformer()
            self._load_text_encoder()

            prompt_embeds, prompt_embeds_mask = compute_text_embeddings(
                [full_description],
                image=[src_tensor, garment_tensor, pose_tensor],
                device=self.device,
                dtype=self.weight_dtype,
                max_sequence_length=128,
                processor=self.processor,
                text_encoder=self.text_encoder,
            )

            negative_prompt_embeds = negative_prompt_embeds_mask = None
            if do_true_cfg:
                negative_prompt_embeds, negative_prompt_embeds_mask = compute_text_embeddings(
                    prompt=[negative_prompt],
                    image=[src_tensor, garment_tensor, pose_tensor],
                    device=self.device,
                    dtype=self.weight_dtype,
                    max_sequence_length=128,
                    processor=self.processor,
                    text_encoder=self.text_encoder,
                )

            self._embed_cache = {
                cache_key: (prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask)
            }

            if self.low_vram:
                self._unload_text_encoder()

        # --- stage 3: denoising ---
        self._load_transformer()

        packed_pose_image_latents = _pack_latents(
            latents=pose_image_latents, batch_size=batch_size, num_channels_latents=pose_image_latents.shape[1],
            height=pose_image_latents.shape[3], width=pose_image_latents.shape[4]
        )
        packed_person_image_latents = _pack_latents(
            latents=person_image_latents, batch_size=batch_size, num_channels_latents=person_image_latents.shape[1],
            height=person_image_latents.shape[3], width=person_image_latents.shape[4]
        )
        packed_garment_image_latents = _pack_latents(
            latents=garment_image_latents, batch_size=batch_size, num_channels_latents=garment_image_latents.shape[1],
            height=garment_image_latents.shape[3], width=garment_image_latents.shape[4]
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents = _prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=img_height,
            width=img_width,
            vae_scale_factor=self.vae_scale_factor,
            dtype=self.weight_dtype,
            device=self.device,
            generator=generator,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len=latents.shape[1],
            base_seq_len=self.noise_scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=self.noise_scheduler.config.get("max_image_seq_len", 4096),
            base_shift=self.noise_scheduler.config.get("base_shift", 0.5),
            max_shift=self.noise_scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=self.noise_scheduler,
            num_inference_steps=num_inference_steps,
            device=self.device,
            sigmas=sigmas,
            mu=mu,
        )

        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=self.device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        self.noise_scheduler.set_begin_index(0)
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            all_packed_latents = torch.cat(
                [
                    latents,
                    packed_person_image_latents,
                    packed_garment_image_latents,
                    packed_pose_image_latents,
                ],
                dim=1
            )

            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=all_packed_latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes * batch_size,
                    txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]
                noise_pred = noise_pred.to(latents.dtype)

            if do_true_cfg:
                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=all_packed_latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes * batch_size,
                        txt_seq_lens=negative_prompt_embeds_mask.sum(dim=1).tolist(),
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    neg_noise_pred = neg_noise_pred.to(latents.dtype)

                # Done in fp32: torch.norm over a 3072-wide fp16 vector can
                # overflow to inf on Turing, which turns the whole latent NaN.
                cond32 = noise_pred.float()
                neg32 = neg_noise_pred.float()
                comb_pred = neg32 + true_cfg_scale * (cond32 - neg32)
                cond_norm = torch.norm(cond32, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = (comb_pred * (cond_norm / noise_norm)).to(latents.dtype)

            latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if progress_callback is not None:
                progress_callback(i + 1, len(timesteps))

        latents = _unpack_latents(
            latents=latents, height=img_height, width=img_width, vae_scale_factor=self.vae_scale_factor
        )

        latents = latents * self.latents_std + self.latents_mean
        latents = latents.to(self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = image.squeeze(dim=2).to(torch.float32)

        image_tensor = (image[0] / 2 + 0.5).clamp(0, 1)
        image_pil = transforms.ToPILImage()(image_tensor.cpu())

        _free()
        return image_pil
