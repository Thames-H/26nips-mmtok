#!/usr/bin/env python3
# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
LLaVA + MMTok example

Flow: load LLaVA → mmtok() injection → conv + tokenizer_image_token → generate.
Uses SDPA for attention 
For flash-attn, please follow instructions in install.md to install flash-attn.

"""

import sys
from pathlib import Path

import torch
from PIL import Image
# Import mmtok first so llava.mm_utils.process_images is patched before we bind the name.
from mmtok import mmtok
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

# ---------- Config (edit as needed) ----------
MODEL_PATH = "liuhaotian/llava-v1.6-vicuna-7b"
IMAGE_PATH = "example/mmtok.jpg"  # Set to your image path, or put an image at example/sample.jpg
QUESTION = "What is in this image?"
TARGET_VISION_TOKENS = 32
DEVICE = "cuda:0"
# --------------------------------------------

if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        print(f"Image not found: {IMAGE_PATH}")
        print("Please set IMAGE_PATH to your image path, or put an image at example/sample.jpg")
        sys.exit(1)

    # 1) Load LLaVA 
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, MODEL_PATH.split("/")[-1], device_map=DEVICE,
        multimodal=True, use_safetensors=True,
        attn_implementation="sdpa", # attn_implementation="flash_attention_2" for flash-attn
    )
    model.eval()
    if not getattr(tokenizer, "is_fast", False):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    # 2) Inject MMTok
    model = mmtok(model, language_tokenizer=tokenizer, target_vision_tokens=TARGET_VISION_TOKENS)

    # 3) Standard LLaVA flow: conv → get_prompt → tokenizer_image_token → generate
    image = Image.open(IMAGE_PATH).convert("RGB")
    model_dtype = next(model.parameters()).dtype
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(dtype=model_dtype, device=DEVICE).contiguous() for t in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=model_dtype, device=DEVICE).contiguous()

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{QUESTION}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)

    attention_mask = input_ids.ne(tokenizer.pad_token_id or tokenizer.eos_token_id)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=[[image.size[0], image.size[1]]],
            do_sample=False,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True).strip())
