"""vectorphile.quantize — Post-training quantization and quantized inference.

Supported formats
-----------------
  INT8  : symmetric per-tensor and per-channel weight quantization
  INT4  : NF4 (NormalFloat-4) and symmetric INT4 (used in QLoRA / GPTQ)
  FP8   : E4M3 / E5M2 floating-point 8-bit (simulated)
  GGUF  : re-export to GGUF Q4_0 / Q8_0 format for llama.cpp / Ollama

Quantization modes
------------------
  'dynamic'    : activations quantized at runtime, weights pre-quantized
  'static'     : both activations and weights quantized (needs calibration)
  'weight-only': only weights quantized (fastest, minimal accuracy loss)

Usage::
    from vectorphile.quantize import quantize_model, QuantizedLinear

    # Quantize a model in-place (weight-only INT8)
    quantize_model(model, bits=8, mode='weight-only')

    # Manual quantized layer
    q_layer = QuantizedLinear.from_linear(linear_layer, bits=4)
"""
from vectorphile.quantize._quant import (
    quantize_tensor,
    dequantize_tensor,
    QuantizedLinear,
    quantize_model,
    calibrate,
    QConfig,
    fp8_e4m3_quantize,
    fp8_e4m3_dequantize,
    fp8_e5m2_quantize,
    fp8_e5m2_dequantize,
    GPTQQuantizer,
)
from vectorphile.quantize._nf4 import nf4_quantize, nf4_dequantize, nf4_linear
from vectorphile.quantize._gguf_export import export_gguf

__all__ = [
    'quantize_tensor', 'dequantize_tensor',
    'QuantizedLinear', 'quantize_model', 'calibrate', 'QConfig',
    'fp8_e4m3_quantize', 'fp8_e4m3_dequantize',
    'fp8_e5m2_quantize', 'fp8_e5m2_dequantize',
    'GPTQQuantizer',
    'nf4_quantize', 'nf4_dequantize', 'nf4_linear',
    'export_gguf',
]
