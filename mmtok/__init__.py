# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs.

Vision token subset selection under the coverage criterion: a subset of vision
tokens is chosen to cover (i) text tokens (question/keywords) and (ii) the
original vision token set, formulated as a maximum coverage problem.

Paper: https://arxiv.org/abs/2508.18264
Title: MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
Authors: Sixun Dong, Juhua Hu, Mian Zhang, Ming Yin, Yanjie Fu, Qi Qian
"""

from .llava import mmtok, apply_llava_patches

# Patch llava.mm_utils.process_images at import time so that code which does
# "import mmtok" before "from llava.mm_utils import process_images" (or
# process_images = llava_mm_utils.process_images) gets the patched version.
# Padding indices are only computed when mmtok(model) is called with LLaVA-1.5.
apply_llava_patches()

__version__ = "1.0.0"
__all__ = ["mmtok", "apply_llava_patches"]
