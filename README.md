# MMTok

MMTok is a training-free vision token selection framework for VLM inference.
This fork adds an independent Qwen3-VL integration that is aware of Qwen3-VL
DeepStack and supports both image and video inputs.

## Highlights

- Supports LLaVA, Qwen2.5-VL, and Qwen3-VL.
- The Qwen3-VL path preserves DeepStack alignment by pruning the final visual
  token sequence and slicing every DeepStack feature tensor with the same keep
  indices.
- The Qwen3-VL path supports both image and video inputs.
- The Qwen3-VL path currently supports `batch_size=1` only.

## Requirements

- Python 3.10+
- PyTorch 2.8.0 was the validated baseline for this repo
- `transformers==4.57.3`

Install the Python dependencies from [requirements.txt](requirements.txt).

## Quick Start

```bash
git clone https://github.com/Thames-H/26nips-mmtok.git
cd 26nips-mmtok
pip install -r requirements.txt
pip install -e .
```

## Examples

```bash
# LLaVA
python example/llava_mmtok_example.py

# Qwen2.5-VL
python example/qwen_mmtok_example.py

# Qwen3-VL image
python example/qwen3_mmtok_image_example.py

# Qwen3-VL video
python example/qwen3_mmtok_video_example.py /path/to/video.mp4
```

## lmms-eval Reference Adapters

- [example/lmms_eval_llava_mmtok.py](example/lmms_eval_llava_mmtok.py)
- [example/lmms_eval_qwen_mmtok.py](example/lmms_eval_qwen_mmtok.py)
- [example/lmms_eval_qwen3_mmtok.py](example/lmms_eval_qwen3_mmtok.py)

## Qwen3-VL Integration Notes

- Entry point: `mmtok.qwen.mmtok_qwen3_vl(...)`
- The wrapper patches the Qwen3-VL vision tower, model forward, and processor
  question hook.
- Image and video inputs share the same `retain_ratio`.
- Video selection is applied over the flattened video visual token set rather
  than per-frame budgets.
- The image example uses the bundled `img/mmtok.png` sample by default.
- The video example expects a local video path as its first CLI argument.

## Package Layout

- `mmtok/qwen/qwen3_vl_mmtok.py`: wrapper injection and question hook
- `mmtok/qwen/qwen3_VLmodel_mmtok.py`: Qwen3-VL model forward patch
- `mmtok/qwen/modeling_qwen3_vl_mmtok.py`: Qwen3-VL vision tower patch

## Status

- Qwen3-VL image support: ready
- Qwen3-VL video support: ready
- Qwen3-VL batch inference: not implemented

See [install.md](install.md) for a fuller setup guide.
