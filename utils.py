import os
import torch
from PIL import Image
import torchvision.transforms as transforms

_DWPOSE = None


def _dwpose_device():
    """Pick a device easy-dwpose can actually use.

    easy-dwpose asks onnxruntime for CUDAExecutionProvider whenever device is
    anything other than "cpu", but environment.yml pins plain `onnxruntime`,
    which is CPU-only. Requesting a provider that is not registered raises, so
    the upstream `"cuda" if torch.cuda.is_available() else "cpu"` crashes on
    exactly the GPU machines this repo targets. Ask onnxruntime what it has.
    """
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
    except Exception:
        return "cpu"

    if torch.cuda.is_available() and "CUDAExecutionProvider" in available:
        return "cuda"
    return "cpu"


def get_dwpose():
    """Build the detector once and reuse it.

    The original code constructed a DWposeDetector on every call, which
    re-created two onnxruntime sessions (and re-checked the HF cache) for each
    generation.
    """
    global _DWPOSE
    if _DWPOSE is None:
        from easy_dwpose import DWposeDetector
        device = _dwpose_device()
        print(f"Initialising DWPose on {device}...")
        _DWPOSE = DWposeDetector(device=device)
    return _DWPOSE


def pad_to_aspect_ratio(image, target_size=(512, 896), pad_color=(255, 255, 255)):
    """
    Resizes and pads an image to the target size while maintaining the aspect ratio.
    Target size is (width, height).
    pad_color is (R, G, B), default is white. For pose, use (0, 0, 0) for black.
    """
    target_w, target_h = target_size
    image = image.convert("RGB")
    img_w, img_h = image.size
    # Calculate ratio
    ratio_w = target_w / img_w
    ratio_h = target_h / img_h
    ratio = min(ratio_w, ratio_h)
    new_w = max(1, int(img_w * ratio))
    new_h = max(1, int(img_h * ratio))
    # Resize image
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Create new image with pad color
    new_image = Image.new("RGB", target_size, pad_color)
    # Paste resized image at the center
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_image.paste(image, (paste_x, paste_y))
    return new_image


def extract_and_process_pose(person_img):
    """
    Extracts pose from the original person image, and returns the padded pose image.
    Uses black padding for the pose image.
    """
    dwpose = get_dwpose()

    # Extract pose from the unpadded, original person image
    img_pose = dwpose(person_img.convert("RGB"), output_type="pil", include_hands=True, include_face=True)

    # Pad the extracted pose with black
    padded_pose = pad_to_aspect_ratio(img_pose, target_size=(512, 896), pad_color=(0, 0, 0))
    return padded_pose


def process_inputs(person_img, garment_img, custom_pose_img=None):
    """
    Processes all inputs:
    - Extracts pose from original person image (if custom pose not provided)
    - Resizes and pads person and garment images with white
    - Resizes and pads pose image with black
    Returns: padded_person, padded_garment, padded_pose
    """
    if custom_pose_img is None:
        padded_pose = extract_and_process_pose(person_img)
    else:
        padded_pose = pad_to_aspect_ratio(custom_pose_img, target_size=(512, 896), pad_color=(0, 0, 0))

    padded_person = pad_to_aspect_ratio(person_img, target_size=(512, 896), pad_color=(255, 255, 255))
    padded_garment = pad_to_aspect_ratio(garment_img, target_size=(512, 896), pad_color=(255, 255, 255))

    return padded_person, padded_garment, padded_pose
