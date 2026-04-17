# MMTok Installation

## 1. Create the environment

```bash
conda create -n mmtok python=3.12 -y
conda activate mmtok
```

## 2. Install PyTorch

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install repo dependencies

This fork pins `transformers==4.57.3` to support the Qwen3-VL path.

```bash
pip install -r requirements.txt
pip install -e .
```

Verify the install:

```bash
python -c "from mmtok.qwen import mmtok_qwen3_vl; print('mmtok installed')"
```

## 4. Optional: install LLaVA-NeXT

Install this only if you want the LLaVA examples.

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]" --no-deps
cd ..
```

## 5. Run the examples

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

## 6. Integrate with lmms-eval

Install `lmms-eval` following its upstream instructions, then use one of the
reference adapters in `example/` as the model entrypoint:

- `example/lmms_eval_llava_mmtok.py`
- `example/lmms_eval_qwen_mmtok.py`
- `example/lmms_eval_qwen3_mmtok.py`

For Qwen3-VL, import and apply:

```python
from mmtok.qwen.qwen3_vl_mmtok import mmtok_qwen3_vl

self._model, self.processor = mmtok_qwen3_vl(
    self._model,
    language_tokenizer=self._tokenizer,
    processor=self.processor,
    retain_ratio=0.1,
)
```

## 7. Qwen3-VL notes

- `mmtok_qwen3_vl(...)` is DeepStack-aware.
- Image and video inputs both run through MMTok selection.
- The image example defaults to the bundled `img/mmtok.png`.
- The video example requires an explicit local video path.
- The same keep indices are applied to:
  `inputs_embeds`
  `position_ids`
  `attention_mask`
  `cache_position`
  `visual_pos_masks`
  every tensor inside `deepstack_visual_embeds`
- `batch_size=1` is currently required for the Qwen3-VL integration.
