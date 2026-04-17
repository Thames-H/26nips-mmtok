# Qwen2.5-VL with MMTok

This folder provides the Qwen2.5-VL integration for **MMTok (ours)**: vision tokens are optionally pruned before being passed to the language model.

We also provide an **optional** reference implementation of **DivPrune** (based on the official DivPrune approach), with a small optimization to reduce repeated selection. This allows you to run **Qwen + DivPrune** using this library for comparison or community use. It is entirely optional; the default is MMTok.

**To use DivPrune:** set the environment variable before running:

```bash
export SELECTION_METHOD=divprune
```

(Default is `SELECTION_METHOD=mmtok`.) The number of vision tokens can be controlled by **`TOKEN_RETAIN_RATIO`** (default `0.1`).

**DivPrune code location:** `qwen2_5_VLmodel_mmtok.py`, lines **L162–L199**.

## Files

- **`qwen2_5_VLmodel_mmtok.py`** — Model forward with token selection (MMTok or DivPrune) applied after the vision encoder.
- **`qwen2_5_vl_mmtok.py`** — High-level wrapper / pipeline for Qwen2.5-VL + MMTok.
- **`modeling_qwen2_5_vl_mmtok.py`** — Modeling utilities for Qwen2.5-VL MMTok.
