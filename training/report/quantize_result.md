# Inference quantization (INT8 dynamic) — result

Setup: CPU, partial eval (max_chunks=30), WikiText-103 (general) +
financial_news (finance). `quantize_dynamic({nn.Linear}, qint8)`.

| checkpoint | size | WikiText↓ | finance↓ |
|---|---|---|---|
| phase1_final (general) fp32 | 395MB | 61.50 | 1761.13 |
| phase1_final INT8 | 254MB (-36%) | 65.41 (+6.3%) | 1807.23 (+2.6%) |
| phase5_final (finance) fp32 | 395MB | 338.94 | 43.56 |
| phase5_final INT8 | 254MB (-36%) | 444.29 (+31.1%) | 63.85 (+46.6%) |

Key finding: INT8-dynamic degradation depends strongly on the model.
- On the general (undertrained) checkpoint it looks nearly free (+2.6-6.3%),
  but that is misleading: finance PPL is huge (1761) because the model can't
  do finance, so quantization noise is proportionally tiny.
- On the finance-specialized deployment checkpoint (low finance PPL 43.56),
  INT8 costs +46.6% finance / +31.1% WikiText — a real degradation.
- Lesson: measure quantization loss on the ACTUAL deployment model, not a
  proxy. A small (98.6M), fine-tuned model is sensitive to INT8 dynamic.
- For this scale, consider static quant (per-channel + calibration), keeping
  fp32, or a milder scheme. Only nn.Linear is quantized (embeddings stay fp32),
  hence 36% size.

"size" = serialized state_dict, not GGUF/GPTQ/AWQ file size. CPU-only (dynamic).
