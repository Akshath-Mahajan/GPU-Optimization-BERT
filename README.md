# Optimizing Transformers Beyond FlashAttention
## Goal
**Triton-Fused Feed-Forward Kernel for BERT**

BERT-like transformers power most modern NLP services, but their feed-forward
layer (Linear → Bias → GELU) still accounts for the bulk of runtime after
attention is optimised.  

This project replaces that sequence with a **single CUDA/Triton kernel**, delivering
30-40 % faster inference and training on a commodity NVIDIA T4.


## Results


| Variant                                  | Inference | Δ Inference | Training step | Δ Training |
|------------------------------------------|-----------|-------------|---------------|------------|
| Baseline (PyTorch 2.6, FlashAttention ON) | 17.6 ms   | –           | 57.2 ms       | –          |
| **Triton-fused FFN kernel**              | **11.7 ms** | **-34 %**   | **35.4 ms**   | **-38 %**  |


<sub>Batch 8 × Seq 11, fp16, NVIDIA T4; medians of 1 000 forward / 300 train iterations measured with CUDA events.</sub>

## Quick start (Colab)

1. Open **`fused_ffn_triton.ipynb`** in Colab (GPU runtime, T4 or newer).
2. Run **all** cells — first run compiles the Triton kernel.

_No additional packages required beyond the notebook-installed Triton
and Transformers; torch 2.6 ships with an embedded Triton 3.2 runtime._

### Key take-aways

- **Single-kernel fusion** collapses the feed-forward linear, bias, and GELU into one Tensor-Core launch.  
- **No retraining required** – the patch applies to any BERT-style checkpoint with hidden = 768 (BERT-base, RoBERTa-base, GPT-2-small, etc.).  
- **Immediate deploy impact** – 34 % faster inference and 38 % faster training on a commodity NVIDIA T4, cutting cloud GPU cost and iteration time.  
- **Profiler-driven** – `torch.profiler` traces pinpointed the hotspot (≈ 83 % of GPU time), guiding effort to the highest-ROI layer.  
