#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from PIL import Image
from mmtok.qwen import mmtok_qwen3_vl
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
QUESTION = "Describe the image in one sentence."
RETAIN_RATIO = 0.2
DEVICE = "cuda:0"
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[1] / "img" / "mmtok.png"


if __name__ == "__main__":
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Pass a local image path as the first argument, or use the bundled sample image.")
        sys.exit(1)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map=DEVICE,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model, processor = mmtok_qwen3_vl(
        model,
        language_tokenizer=processor.tokenizer,
        processor=processor,
        retain_ratio=RETAIN_RATIO,
    )

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        do_resize=False,
        return_tensors="pt",
    ).to(DEVICE)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=128,
            use_cache=True,
        )

    generated_ids = out[:, inputs["input_ids"].shape[1] :]
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(answer.strip())
