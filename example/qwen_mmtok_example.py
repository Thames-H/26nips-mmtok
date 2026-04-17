#!/usr/bin/env python3
# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
Qwen2.5-VL + MMTok example

Flow: load Qwen2.5-VL → mmtok_qwen2_5_vl() injection → apply_chat_template → generate.
Requires: mmtok, transformers>=4.52.4, accelerate, qwen-vl-utils, pillow.
Uses SDPA for attention. For flash-attn, see install.md and set attn_implementation.
"""


import sys
from pathlib import Path

import torch
from PIL import Image
from mmtok.qwen import mmtok_qwen2_5_vl
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------- Config (edit as needed) ----------
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "MMTok/example/mmtok.jpg"
QUESTION = "What is in this image?"
RETAIN_RATIO = 0.2   # keep 20% of vision tokens per image
DEVICE = "cuda:0"
# --------------------------------------------

if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        print(f"Image not found: {IMAGE_PATH}")
        print("Please set IMAGE_PATH to your image path, or put an image at example/mmtok.jpg")
        sys.exit(1)

    # 1) Load Qwen2.5-VL
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="sdpa",  # attn_implementation="flash_attention_2" for flash-attn
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 2) Inject MMTok
    model, processor = mmtok_qwen2_5_vl(
        model,
        language_tokenizer=processor.tokenizer,
        processor=processor,
        retain_ratio=RETAIN_RATIO,
    )

    # 3) Standard Qwen2.5-VL flow: apply_chat_template → process → generate
    image = Image.open(IMAGE_PATH).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=128,
            use_cache=True,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated_ids = out[:, inputs["input_ids"].shape[1]:]
    print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())
