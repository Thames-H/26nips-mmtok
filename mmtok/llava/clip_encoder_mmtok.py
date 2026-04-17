# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
MMTok CLIP encoder: forward runs CLIP -> mm_projector -> coverage-based subset selection.

Subset of vision tokens is selected to cover text tokens (question) and the
vision token set (maximum coverage formulation).
"""

import torch
import torch.nn as nn
from loguru import logger as eval_logger

from .patch_llava import get_padding_patch_indices


class CLIPVisionTower_MMTok(nn.Module):
    """
    MMTok vision tower: CLIP -> mm_projector -> subset selection (coverage criterion).
    """

    @torch.no_grad()
    def forward(self, images):
        """
        Forward with coverage-based subset selection: CLIP -> mm_projector -> select
        vision tokens that cover text (question) and vision set; return selected subset.

        Args:
            images: Input image tensor(s).
        Returns:
            Selected vision features (mm_projector space); second return is selected_indices.
        """
        question = self.get_question()

        if question:
            # Strip leading <image>\n
            if question.startswith("<image>\n"):
                question = question[8:]
            # Strip leading "<image> <image> ...\\n" and keep the rest as question text
            elif question.startswith("<image> "):
                parts = question.split("\n", 1)
                if len(parts) == 2:
                    prefix, remaining = parts[0], parts[1]
                    tokens = prefix.split()
                    if all(t == "<image>" for t in tokens):
                        question = remaining
                    else:
                        raise ValueError(f"Invalid question: {question}")
        else:
            question = "What do you see in this image?"
            eval_logger.error(f"No question found for MMTok, using default: {question}")


        if self.remove_padding_indices:
            padding_patch_indices = get_padding_patch_indices(clear=True)
        else:
            padding_patch_indices = None

        if type(images) is list:
            image_features = []
            for image in images:
                image_feature_before_mm_projection = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True).hidden_states[-2]
                image_feature_after_mm_projection = self.mm_projector(image_feature_before_mm_projection[:, 1:].to(dtype=self.dtype))

                selected_feature, selected_indices = self._mmtok_core.apply_selection(image_feature_after_mm_projection, image_feature_before_mm_projection, images, question, padding_patch_indices)
                image_features.append(selected_feature)
            return image_features, torch.tensor(selected_indices)
        else:
            image_feature_before_mm_projection = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True).hidden_states[-2]
            image_feature_after_mm_projection = self.mm_projector(image_feature_before_mm_projection[:, 1:].to(dtype=self.dtype))

            selected_image_feature_after_mm_projection, selected_indices = self._mmtok_core.apply_selection(
                image_feature_after_mm_projection, image_feature_before_mm_projection, images, question, text_embeds=None, padding_patch_indices=padding_patch_indices
            )
            return selected_image_feature_after_mm_projection, torch.tensor(selected_indices).to(self.device)
