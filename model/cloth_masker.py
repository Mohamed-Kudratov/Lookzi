import os
from PIL import Image
from typing import Union
import numpy as np
import cv2
from diffusers.image_processor import VaeImageProcessor
import torch

from model.SCHP import SCHP  # type: ignore
from model.DensePose import DensePose  # type: ignore

DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}

ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}

LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PROTECT_BODY_PARTS = {
    'upper': ['Left-leg', 'Right-leg'],
    'lower': ['Right-arm', 'Left-arm', 'Face'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg'],
    'outer': ['Left-leg', 'Right-leg'],
}
PROTECT_CLOTH_PARTS = {
    'upper': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'lower': {
        'ATR': ['Upper-clothes'],
        'LIP': ['Upper-clothes', 'Coat']
    },
    'overall': {
        'ATR': [],
        'LIP': []
    },
    'inner': {
        'ATR': ['Dress', 'Coat', 'Skirt', 'Pants'],
        'LIP': ['Dress', 'Coat', 'Skirt', 'Pants', 'Jumpsuits']
    },
    'outer': {
        'ATR': ['Dress', 'Pants', 'Skirt'],
        'LIP': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Jumpsuits']
    }
}
MASK_CLOTH_PARTS = {
    'upper': ['Upper-clothes', 'Coat', 'Dress', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}
MASK_DENSE_PARTS = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}
    
schp_public_protect_parts = ['Hat', 'Hair', 'Sunglasses', 'Left-shoe', 'Right-shoe', 'Bag', 'Glove', 'Scarf']
schp_protect_parts = {
    'upper': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits'],  
    'lower': ['Left-arm', 'Right-arm', 'Upper-clothes', 'Coat'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Coat'],
    'outer': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Upper-clothes']
}
schp_mask_parts = {
    'upper': ['Upper-clothes', 'Dress', 'Coat', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits', 'socks'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits', 'socks'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}

dense_mask_parts = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}

def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))

def part_mask_of(part: Union[str, list],
                 parse: np.ndarray, mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask

def hull_mask(mask_area: np.ndarray):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask

def clean_binary_mask(mask: np.ndarray, kernel: np.ndarray, close_iterations=2, open_iterations=1):
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    return mask > 0


def keep_main_components(mask: np.ndarray, min_area_ratio: float = 0.002):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)

    image_area = mask.shape[0] * mask.shape[1]
    min_area = max(32, int(image_area * min_area_ratio))
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(component_areas.max()) if len(component_areas) else 0
    keep = np.zeros_like(mask, dtype=bool)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area or area >= largest_area * 0.25:
            keep |= labels == label_idx
    return keep


def skirt_silhouette_mask(lower_body_area: np.ndarray, covers_lower_legs: bool):
    ys, xs = np.where(lower_body_area)
    if len(xs) < 20:
        return lower_body_area

    height, width = lower_body_area.shape
    y_top = max(0, int(ys.min() - height * 0.025))
    dense_bottom = int(ys.max())
    if covers_lower_legs:
        y_bottom = dense_bottom
    else:
        y_bottom = min(dense_bottom, int(y_top + (dense_bottom - y_top) * 0.76))

    center_x = int(np.median(xs))
    body_width = int(xs.max() - xs.min() + 1)
    hip_width = max(18, int(body_width * 0.72))
    hem_width = max(int(hip_width * 1.55), int(body_width * 1.15))
    hem_width = min(hem_width, int(width * 0.78))

    rows = max(1, y_bottom - y_top)
    mask = np.zeros_like(lower_body_area, dtype=np.uint8)
    for y in range(y_top, y_bottom + 1):
        t = (y - y_top) / rows
        half_width = int((hip_width + (hem_width - hip_width) * t) / 2)
        x0 = max(0, center_x - half_width)
        x1 = min(width - 1, center_x + half_width)
        mask[y, x0:x1 + 1] = 1

    return mask.astype(bool)
    

class AutoMasker:
    def __init__(
        self, 
        densepose_ckpt='./Models/DensePose', 
        schp_ckpt='./Models/SCHP', 
        device='cuda'):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        self.densepose_processor = DensePose(densepose_ckpt, device)
        self.schp_processor_atr = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908301523-atr.pth'), device=device)
        self.schp_processor_lip = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908261155-lip.pth'), device=device)
        
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def process_densepose(self, image_or_path):
        return self.densepose_processor(image_or_path, resize=1024)

    def process_schp_lip(self, image_or_path):
        return self.schp_processor_lip(image_or_path)

    def process_schp_atr(self, image_or_path):
        return self.schp_processor_atr(image_or_path)
        
    def preprocess_image(self, image_or_path):
        return {
            'densepose': self.densepose_processor(image_or_path, resize=1024),
            'schp_atr': self.schp_processor_atr(image_or_path),
            'schp_lip': self.schp_processor_lip(image_or_path)
        }
    
    @staticmethod
    def cloth_agnostic_mask(
        densepose_mask: Image.Image,
        schp_lip_mask: Image.Image,
        schp_atr_mask: Image.Image,
        part: str='overall',
        garment_style: str='auto',
        covers_lower_legs: bool=True,
        **kwargs
    ):
        assert part in ['upper', 'lower', 'overall', 'inner', 'outer'], f"part should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {part}"
        w, h = densepose_mask.size
        
        dilate_kernel = max(w, h) // 250
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        
        kernal_size = max(w, h) // 25
        kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1
        
        densepose_mask = np.array(densepose_mask)
        schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)
        
        # Strong Protect Area (Hands, Face, Accessory, Feet)
        hands_protect_area = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
        hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
        hands_protect_area = hands_protect_area & \
            (part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING) | \
             part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING))
        face_protect_area = part_mask_of('Face', schp_lip_mask, LIP_MAPPING)

        strong_protect_area = hands_protect_area | face_protect_area 

        # Weak Protect Area (Hair, Irrelevant Clothes, Body Parts)
        body_protect_area = part_mask_of(PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING) | part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING)
        hair_protect_area = part_mask_of(['Hair'], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(['Hair'], schp_atr_mask, ATR_MAPPING)
        cloth_protect_area = part_mask_of(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING)
        accessory_protect_area = part_mask_of((accessory_parts := ['Hat', 'Glove', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Scarf', 'Socks']), schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(accessory_parts, schp_atr_mask, ATR_MAPPING) 
        weak_protect_area = body_protect_area | cloth_protect_area | hair_protect_area | strong_protect_area | accessory_protect_area
        
        # Mask Area
        strong_mask_area = (part_mask_of(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING)).astype(bool)
        background_area = (part_mask_of(['Background'], schp_lip_mask, LIP_MAPPING) & part_mask_of(['Background'], schp_atr_mask, ATR_MAPPING)).astype(bool)
        
        # FIX: If SCHP hallucinates background over light skin/clothes, but DensePose detects a body part, trust DensePose!
        # This prevents bare legs and white pants from being deleted by ~background_area
        densepose_body = (np.array(densepose_mask) > 0)
        background_area = background_area & (~densepose_body)

        mask_dense_area = part_mask_of(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP).astype(bool)
        
        original_shape = mask_dense_area.shape[:2]  # (height, width)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
        mask_dense_area = cv2.resize(mask_dense_area, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        strong_protect_area = strong_protect_area.astype(bool)
        weak_protect_area = weak_protect_area.astype(bool)

        torso_area = part_mask_of('torso', densepose_mask, DENSE_INDEX_MAP).astype(bool)
        upper_arm_area = part_mask_of('big arms', densepose_mask, DENSE_INDEX_MAP).astype(bool)
        forearm_area = part_mask_of('forearms', densepose_mask, DENSE_INDEX_MAP).astype(bool)
        lower_body_area = part_mask_of(['thighs', 'legs'], densepose_mask, DENSE_INDEX_MAP).astype(bool)
        arms_protect_area = forearm_area | part_mask_of(['hands'], densepose_mask, DENSE_INDEX_MAP).astype(bool)

        lower_leg_area = part_mask_of('legs', densepose_mask, DENSE_INDEX_MAP).astype(bool)
        thigh_area = part_mask_of('thighs', densepose_mask, DENSE_INDEX_MAP).astype(bool)
        feet_area = part_mask_of('feet', densepose_mask, DENSE_INDEX_MAP).astype(bool)

        if part in ['upper', 'inner', 'outer']:
            allowed_area = torso_area | strong_mask_area
            if part == 'outer' or garment_style in ['short_sleeve', 'sleeved']:
                allowed_area = allowed_area | upper_arm_area
            # Fix 2: sleeved garments must also mask forearms (cardigan sleeve cut issue)
            if garment_style == 'sleeved':
                allowed_area = allowed_area | forearm_area
                local_arm_protect = part_mask_of(['hands'], densepose_mask, DENSE_INDEX_MAP).astype(bool)
            else:
                local_arm_protect = arms_protect_area
            protect_area = strong_protect_area | background_area | local_arm_protect | part_mask_of(['Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING).astype(bool)
            # Fix 1 result: sleeveless → protect upper arms so model won't add sleeves
            if garment_style == 'sleeveless':
                protect_area = protect_area | upper_arm_area
            mask_area = (allowed_area | (mask_dense_area & torso_area)) & (~protect_area)
            mask_area = clean_binary_mask(mask_area, dilate_kernel, close_iterations=2, open_iterations=1)
            mask_area = cv2.dilate(mask_area.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)
        elif part == 'lower':
            # hands_protect_area includes feet, dilated upward into leg zone → mask shrinks to 6-13%.
            # Fix: remove strong_protect from pixels that belong to the lower body itself.
            # Actual shoe pixels (DensePose 5,6) are NOT in lower_body_area, so shoes stay protected.
            local_strong_protect = strong_protect_area & ~lower_body_area
            # Hull fill: close the gap between legs so pants/skirt cover inner thigh area too
            target_lower_area = lower_body_area if covers_lower_legs else thigh_area
            if garment_style == 'skirt':
                target_lower_area = skirt_silhouette_mask(lower_body_area, covers_lower_legs)

            source_lower_mask = strong_mask_area if garment_style != 'skirt' else np.zeros_like(strong_mask_area)
            lower_base = (target_lower_area | source_lower_mask).astype(np.uint8) * 255
            lower_hull = hull_mask(lower_base).astype(bool)
            allowed_area = lower_hull | target_lower_area | source_lower_mask
            lower_background_protect = background_area & ~target_lower_area
            protect_area = local_strong_protect | lower_background_protect | part_mask_of(['Left-arm', 'Right-arm', 'Face'], schp_lip_mask, LIP_MAPPING).astype(bool)
            mask_area = (allowed_area | mask_dense_area) & (~protect_area)
            mask_area = clean_binary_mask(mask_area, dilate_kernel, close_iterations=3, open_iterations=1)
            mask_area = keep_main_components(mask_area, min_area_ratio=0.003)

            if mask_area.mean() < 0.18:
                fallback_dense_area = mask_dense_area if covers_lower_legs else (mask_dense_area & thigh_area)
                fallback = (lower_hull | target_lower_area | fallback_dense_area) & (~lower_background_protect) & (~feet_area)
                fallback = clean_binary_mask(fallback, dilate_kernel, close_iterations=3, open_iterations=1)
                mask_area = keep_main_components(fallback, min_area_ratio=0.003)
        else:
            # Overall: torso + thighs always masked; lower legs only when garment covers them
            allowed_area = torso_area | thigh_area | strong_mask_area
            if covers_lower_legs:
                allowed_area = allowed_area | lower_leg_area
            if garment_style != 'sleeveless':
                allowed_area = allowed_area | upper_arm_area
            protect_area = strong_protect_area | background_area | hair_protect_area.astype(bool) | arms_protect_area
            if garment_style == 'sleeveless':
                protect_area = protect_area | upper_arm_area
            # Fix 3: short garments → protect lower legs to prevent hallucinated socks
            if not covers_lower_legs:
                protect_area = protect_area | lower_leg_area
            mask_area = (allowed_area | mask_dense_area) & (~protect_area)
            mask_area = clean_binary_mask(mask_area, dilate_kernel, close_iterations=3, open_iterations=1)

        mask_area = cv2.GaussianBlur(mask_area.astype(np.uint8) * 255, (kernal_size, kernal_size), 0)
        mask_area[mask_area < 25] = 0
        mask_area[mask_area >= 25] = 1
        final_protect_area = protect_area | background_area
        if part == 'lower':
            final_protect_area = (protect_area | background_area | feet_area) & ~target_lower_area
        final_mask_seed = mask_area.astype(bool)
        if not (part == 'lower' and garment_style == 'skirt'):
            final_mask_seed = final_mask_seed | strong_mask_area
        mask_area = final_mask_seed & (~final_protect_area)
        if part == 'lower':
            mask_area = keep_main_components(mask_area, min_area_ratio=0.003)
        mask_area = cv2.dilate(mask_area.astype(np.uint8), dilate_kernel, iterations=1)

        return Image.fromarray((mask_area * 255).astype(np.uint8))
        
    def __call__(
        self,
        image: Union[str, Image.Image],
        mask_type: str = "upper",
        garment_style: str = "auto",
        covers_lower_legs: bool = True,
    ):
        assert mask_type in ['upper', 'lower', 'overall', 'inner', 'outer'], f"mask_type should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {mask_type}"
        preprocess_results = self.preprocess_image(image)
        mask = self.cloth_agnostic_mask(
            preprocess_results['densepose'],
            preprocess_results['schp_lip'],
            preprocess_results['schp_atr'],
            part=mask_type,
            garment_style=garment_style,
            covers_lower_legs=covers_lower_legs,
        )
        return {
            'mask': mask,
            'densepose': preprocess_results['densepose'],
            'schp_lip': preprocess_results['schp_lip'],
            'schp_atr': preprocess_results['schp_atr']
        }


if __name__ == '__main__':
    pass
