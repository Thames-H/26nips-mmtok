#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from mmtok.qwen import mmtok_qwen3_vl
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise SystemExit("Please install qwen-vl-utils to run this example.") from exc


MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
QUESTION = "Summarize the key motion in this video."
RETAIN_RATIO = 0.2
DEVICE = "cuda:0"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example/qwen3_mmtok_video_example.py <video-path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Video not found: {video_path}")
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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=False,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
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
