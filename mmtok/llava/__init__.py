# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
MMTok LLaVA backend: injection, patch, CLIP encoder, and arch adapters.
"""

from .llava_inject import mmtok
from .patch_llava import apply_llava_patches, get_latest_images, get_padding_patch_indices

try:
    from .clip_encoder_mmtok import CLIPVisionTower_MMTok
    from .llava_arch_mmtok import (
        encode_images_mmtok,
        encode_images_mmtok_multi,
        prepare_inputs_labels_for_multimodal_mmtok,
        restore_image_features_sorted,
    )
except ImportError:
    CLIPVisionTower_MMTok = None
    encode_images_mmtok = None
    encode_images_mmtok_multi = None
    prepare_inputs_labels_for_multimodal_mmtok = None
    restore_image_features_sorted = None

__all__ = [
    "mmtok",
    "apply_llava_patches",
    "get_latest_images",
    "get_padding_patch_indices",
    "CLIPVisionTower_MMTok",
    "encode_images_mmtok",
    "encode_images_mmtok_multi",
    "prepare_inputs_labels_for_multimodal_mmtok",
    "restore_image_features_sorted",
]
