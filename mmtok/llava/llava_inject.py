# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
MMTok LLaVA injection: coverage-based vision token subset selection for LLaVA models.

Injects MMTok into a LLaVA model so that a subset of vision tokens is selected
under the maximum coverage criterion (cover text tokens + vision token set).
"""

import types

from loguru import logger as eval_logger
from transformers import (CLIPTextModelWithProjection, CLIPTokenizerFast,
                          CLIPVisionModelWithProjection)

from ..core import MMTokCore
from .patch_llava import apply_llava_patches, set_use_padding_indices

try:
    from .clip_encoder_mmtok import CLIPVisionTower_MMTok
    from .llava_arch_mmtok import (
        encode_images_mmtok,
        encode_images_mmtok_multi,
        prepare_inputs_labels_for_multimodal_mmtok,
        restore_image_features_sorted,
    )
except ImportError:
    pass  # LLaVA not installed; mmtok() for LLaVA will raise at call time


def mmtok(model, language_tokenizer=None, target_vision_tokens=64, **mmtok_kwargs):
    """
    Inject MMTok into a LLaVA model: coverage-based subset selection of vision tokens.

    A subset of vision tokens is selected under the maximum coverage criterion
    to cover text tokens (question/keywords) and the vision token set. All
    parameters are passed to MMTokCore.

    Args:
        model: LLaVA model to inject MMTok into
        language_tokenizer: language tokenizer (for text token embedding)
        target_vision_tokens: Target subset size (default: 64)
        **mmtok_kwargs: MMTok config (alpha, temperatures, etc.)
    Returns:
        model: Model with MMTok injection applied

    Example:
        >>> from mmtok import mmtok
        >>> model = mmtok(model, target_vision_tokens=64)
        >>> model = mmtok(model, target_vision_tokens=128, alpha=0.8)
    """
    mmtok_config = {
        "target_vision_tokens": target_vision_tokens,
        "alpha": 0.5,
        "softmax_tv_temperature": 0.02,
        "softmax_vv_temperature": 0.2,
        "remove_padding_indices": None,  # None = auto (resolved below); True only for LLaVA 1.5
        **mmtok_kwargs,
    }

    mmtok_config["device"] = model.device

    # Resolve remove_padding_indices: only LLaVA 1.5 supports True; LLaVA 1.6 must be False
    _model_path = getattr(model.config, "_name_or_path", None) or getattr(model.config, "name_or_path", "") or ""
    _model_path = str(_model_path).lower()
    _is_llava_15 = "llava-v1.5" in _model_path
    if mmtok_config.get("remove_padding_indices") is None:
        mmtok_config["remove_padding_indices"] = _is_llava_15
    elif mmtok_config.get("remove_padding_indices") is True and not _is_llava_15:
        eval_logger.warning(
            "[MMTok] remove_padding_indices=True is only supported for LLaVA 1.5; forcing False for this model."
        )
        mmtok_config["remove_padding_indices"] = False

    # Patch is applied at import mmtok; here we only enable padding-index computation for LLaVA-1.5.
    set_use_padding_indices(_is_llava_15)

    eval_logger.info(
        f"[MMTok] Injecting coverage-based subset selection: "
        f"target_tokens={mmtok_config['target_vision_tokens']}, "
        f"tv_temp={mmtok_config['softmax_tv_temperature']}, "
        f"vv_temp={mmtok_config['softmax_vv_temperature']}, "
        f"alpha={mmtok_config['alpha']}, "
        f"device={mmtok_config['device']}"
    )

    mmtok_core = MMTokCore(**mmtok_config)

    vision_tower = model.get_vision_tower()
    model.encode_images_mmtok = types.MethodType(encode_images_mmtok, model)
    model.restore_image_features_sorted = types.MethodType(restore_image_features_sorted, model)
    model.prepare_inputs_labels_for_multimodal = types.MethodType(prepare_inputs_labels_for_multimodal_mmtok, model)
    model.encode_images_mmtok_multi = types.MethodType(encode_images_mmtok_multi, model)

    CLIPVisionModelWithProjection._no_split_modules = ["CLIPEncoderLayer"]
    vision_tower_with_projection = CLIPVisionModelWithProjection.from_pretrained(
        vision_tower.vision_tower_name, device_map=model.device, use_safetensors=True
    )

    mmtok_core.visual_projection = vision_tower_with_projection.visual_projection
    mmtok_core.text_tokenizer = CLIPTokenizerFast.from_pretrained(vision_tower.vision_tower_name)
    mmtok_core.text_tower = CLIPTextModelWithProjection.from_pretrained(
        vision_tower.vision_tower_name, device_map=model.device, use_safetensors=True
    )
    mmtok_core.text_tower.requires_grad_(False)
    mmtok_core.vision_tower_post_layernorm = vision_tower.vision_tower.vision_model.post_layernorm
    mmtok_core.max_position_embeddings = mmtok_core.text_tower.config.max_position_embeddings
    mmtok_core._main_model_embed_tokens = model.get_model().embed_tokens
    mmtok_core._language_tokenizer = language_tokenizer

    vision_tower._mmtok_core = mmtok_core
    vision_tower.remove_padding_indices = mmtok_core.remove_padding_indices
    vision_tower.forward = types.MethodType(CLIPVisionTower_MMTok.forward, vision_tower)
    vision_tower.mm_projector = model.get_model().mm_projector
    vision_tower._question_for_vision = None
    vision_tower.set_question = types.MethodType(_set_question, vision_tower)
    vision_tower.get_question = types.MethodType(_get_question, vision_tower)

    patch_conv_copy_for_hook("vicuna_v1", vision_tower)

    eval_logger.info("[MMTok] LLaVA components (VisionTower, Projector, Conversation) patched; MMTok injection complete.")

    return model


def _set_question(self, question: str):
    self._question_for_vision = question


def _get_question(self):
    return self._question_for_vision


def patch_conv_copy_for_hook(conv_name, llava_instance):
    """
    Patch the 'copy' method of a conversation template to capture the prompt for MMTok.
    Imports llava.conversation only inside this function so LLaVA remains optional.
    """
    try:
        from llava.conversation import conv_templates
    except ImportError:
        eval_logger.warning("[MMTok] LLaVA conversation not found. Question hooking disabled.")
        return

    if conv_name not in conv_templates:
        return

    conv = conv_templates[conv_name]

    if getattr(conv, "_mmtok_patched", False):
        return

    orig_copy = conv.copy

    def patched_copy(self):
        new_conv = orig_copy()
        original_append = new_conv.append_message

        def patched_append_message(self, role, message):
            if role == self.roles[0] and message:
                llava_instance.set_question(message)
            return original_append(role, message)

        new_conv.append_message = types.MethodType(patched_append_message, new_conv)
        return new_conv

    conv.copy = types.MethodType(patched_copy, conv)
    conv._mmtok_patched = True
