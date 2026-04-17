# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
MMTok LLaVA patch: utilities and monkey-patch of llava.mm_utils.process_images.

apply_llava_patches() is called at mmtok import time so that any code that does
"import mmtok" before "from llava.mm_utils import process_images" (or
process_images = llava_mm_utils.process_images) gets the patched version.

When the patch is applied, each process_images call stores the image list in
_MMTOK_LATEST_IMAGES. Padding patch indices are only computed when
_MMTOK_USE_PADDING_INDICES is True (set by mmtok(model) for LLaVA-1.5 only).
"""

import math
from loguru import logger as eval_logger

_MMTOK_LATEST_IMAGES = None
_MMTOK_PADDING_PATCH_INDICES = None
_MMTOK_USE_PADDING_INDICES = False


def set_use_padding_indices(use: bool):
    """Enable/disable padding index computation in the process_images wrapper (LLaVA-1.5 only)."""
    global _MMTOK_USE_PADDING_INDICES
    _MMTOK_USE_PADDING_INDICES = use


def calculate_padding_patch_indices(original_size, target_size=336, patch_size=14, include_overlap=False):
    """
    Compute patch indices that fall inside the padding region (after resize to target_size).

    Args:
        original_size: (width, height) of the original image
        target_size: Resize target (default 336)
        patch_size: Patch size (default 14)
        include_overlap: If False (default), only patches fully inside padding (//).
                        If True, any patch touching padding (ceil) is masked.

    Returns:
        List of patch indices in the padding region (no duplicates).
    """
    assert target_size % patch_size == 0, f"target_size ({target_size}) must be divisible by patch_size ({patch_size})"
    orig_width, orig_height = original_size
    if orig_width == orig_height:
        return []
    num_patches = target_size // patch_size  # 24 x 24 = 576

    if orig_width > orig_height:
        scale_factor = target_size / orig_width
        scaled_height = int(round(orig_height * scale_factor))
        padding_top_336 = (target_size - scaled_height) // 2
        padding_bottom_336 = target_size - scaled_height - padding_top_336
        if scaled_height == target_size:
            return []
        if include_overlap:
            pad_top_rows = math.ceil(padding_top_336 / patch_size)
            pad_bot_rows = math.ceil(padding_bottom_336 / patch_size)
        else:
            pad_top_rows = padding_top_336 // patch_size
            pad_bot_rows = padding_bottom_336 // patch_size
        padding_patch_indices = []
        for row in range(pad_top_rows):
            for col in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
        for row in range(num_patches - pad_bot_rows, num_patches):
            for col in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
    else:
        scale_factor = target_size / orig_height
        scaled_width = int(round(orig_width * scale_factor))
        padding_left_336 = (target_size - scaled_width) // 2
        padding_right_336 = target_size - scaled_width - padding_left_336
        if scaled_width == target_size:
            return []
        if include_overlap:
            pad_left_cols = math.ceil(padding_left_336 / patch_size)
            pad_right_cols = math.ceil(padding_right_336 / patch_size)
        else:
            pad_left_cols = padding_left_336 // patch_size
            pad_right_cols = padding_right_336 // patch_size
        padding_patch_indices = []
        for col in range(pad_left_cols):
            for row in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
        for col in range(num_patches - pad_right_cols, num_patches):
            for row in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
    return padding_patch_indices


def get_latest_images(clear=True):
    global _MMTOK_LATEST_IMAGES
    imgs = _MMTOK_LATEST_IMAGES
    if clear:
        _MMTOK_LATEST_IMAGES = None
    return imgs or []  # avoid returning None


def get_padding_patch_indices(clear=True):
    """
    Return padding patch indices (List[List[int]]) set by the last process_images call.
    """
    global _MMTOK_PADDING_PATCH_INDICES
    indices = _MMTOK_PADDING_PATCH_INDICES
    if clear:
        _MMTOK_PADDING_PATCH_INDICES = None
    return indices


def apply_llava_patches():
    """
    Apply monkey-patch to llava.mm_utils.process_images so that latest images and
    padding patch indices are stored for MMTok. Idempotent; safe to call multiple times.
    """
    try:
        import llava.mm_utils as mm_utils
    except ImportError:
        eval_logger.warning("[MMTok] LLaVA not installed. Patch skipped.")
        return

    # 1. Idempotency: avoid double patch (e.g. module reload or other plugins)
    if getattr(mm_utils, "_mmtok_patched", False):
        return

    # 2. Capture original function
    original_process_images = mm_utils.process_images

    # 3. Define wrapper (closure over original_process_images)
    def _new_process_images(flattened_visuals, *args, **kwargs):
        global _MMTOK_LATEST_IMAGES, _MMTOK_PADDING_PATCH_INDICES, _MMTOK_USE_PADDING_INDICES

        if flattened_visuals:
            _MMTOK_LATEST_IMAGES = flattened_visuals
            if _MMTOK_USE_PADDING_INDICES:
                try:
                    padding_patch_indices = []
                    for image in flattened_visuals:
                        patch_indices = calculate_padding_patch_indices(
                            image.size, target_size=336, patch_size=14, include_overlap=False
                        )
                        padding_patch_indices.append(patch_indices)
                    _MMTOK_PADDING_PATCH_INDICES = padding_patch_indices
                except Exception as e:
                    eval_logger.warning(f"[MMTok] Error calculating padding indices: {e}")

        return original_process_images(flattened_visuals, *args, **kwargs)

    # 4. Apply patch and mark
    mm_utils.process_images = _new_process_images
    mm_utils._mmtok_patched = True

    eval_logger.info("[MMTok] ✅ llava.mm_utils.process_images successfully patched.")
