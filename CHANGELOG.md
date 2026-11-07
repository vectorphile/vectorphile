# Changelog

All notable changes to vectorphile are documented here.

---

## [0.4.0] — 2026-11-08

### New — Tensor methods
- `Tensor.numel()` — total number of elements
- `Tensor.element_size()` — bytes per element (`itemsize`)
- `Tensor.dim()` — number of dimensions (alias for `ndim`)
- `Tensor.add(other)` — explicit add method (mirrors `+` operator)
- `Tensor.expand_as(other)` — broadcast-expand to match another tensor's shape

### New — Tensor operations (module-level)
- `vp.tril(input, diagonal=0)` — lower triangular
- `vp.triu(input, diagonal=0)` — upper triangular
- `vp.bmm(a, b)` — batched matrix multiply
- `vp.topk(input, k, dim=-1, largest=True, sorted=True)` — top-k values and indices
- `vp.cumsum(input, dim=0)` — cumulative sum
- `vp.multinomial(input, num_samples, replacement=True)` — multinomial sampling
- `vp.isnan(input)` — element-wise NaN check
- `vp.isinf(input)` — element-wise Inf check
- `vp.sort(input, dim=-1, descending=False)` — sort values and indices

### New — nn layers
- `nn.Conv1d` — 1D convolution with `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `bias`, `groups`, `dilation`, `padding_mode`
- `nn.Flatten` — flatten spatial dims (`start_dim`, `end_dim`)
- `nn.Parameter` — alias for `Tensor` (parameter declaration pattern)

### New — nn.Module methods
- `Module.buffers()` / `Module.named_buffers()`
- `Module.children()` / `Module.named_children()`

### New — nn.functional
- `F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)`

### New — hub
- `vp.hub.save_checkpoint(model, path, metadata=None)` — save model to any file path
- `vp.hub.load_checkpoint(path)` — load model from checkpoint file

### New — nn.Linear
- `.bias` property — alias for `.bias_param` for compatibility with standard PyTorch patterns

### New — nn.MultiheadAttention
- `.forward()` now accepts `need_weights` kwarg (ignored, for API compatibility)

### Fixed
- `Tensor.masked_fill()` — was using boolean index assignment which broke on 4D tensors (e.g. `[B, H, T, T]` attention masks); now uses `np.where` with correct broadcasting
- `Tensor.split(sizes)` — now accepts a list of sizes (e.g. `split([dt_rank, d_state, d_state], dim=-1)`) in addition to uniform chunk size
- `vp.cuda.is_available()` — was hardcoded to return `False`; now uses real Numba GPU enumeration via `len(list(numba.cuda.gpus)) > 0`
- `vp.cuda.device_count()` / `vp.cuda.get_device_name()` — same fix

---

## [0.3.0] — initial public release

Core tensor engine, autograd, nn module, optimizers, lr schedulers, dataset utilities, ML algorithms, hub, amp, export, tokenizers, distributed, rl, pretrained, metrics, callbacks, trainer, text, vision.
