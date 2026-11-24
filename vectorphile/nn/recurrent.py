"""Recurrent layers — RNN, LSTM, GRU with bidirectional and multi-layer support.

All forward paths use vectorphile Tensor ops + NDArray/C kernels — no numpy or torch.
"""
from __future__ import annotations
import math
import array as _arr
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor, stack as _t_stack, cat as _t_cat


def _zeros_tensor(N, H, dtype='float32'):
    """Create a (N, H) zero Tensor using NDArray."""
    from vectorphile._core.ndarray import zeros as _nd_zeros
    if dtype == 'float32':
        return Tensor(_nd_zeros((N, H), dtype='float32'))
    # fall back for float64 (gradient checking in tests)
    import array as _a2
    buf = _a2.array('d', bytearray(8 * N * H))
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
    return Tensor(_NDA(buf, (N, H), _cs((N, H)), _DT('float64')))


def _rand_mask(shape, p, dtype_str='float32'):
    """Create a Bernoulli(1-p) mask using stdlib random — no numpy."""
    import random
    n = 1
    for s in shape: n *= s
    buf = _arr.array('f', (1.0 if random.random() > p else 0.0 for _ in range(n)))
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
    nd = _NDA(buf, shape, _cs(shape), _DT('float32'))
    return Tensor(nd)


class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, nonlinearity='tanh', bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        # init weights via NDArray uniform
        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros
        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor(_nd_uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_hh = Tensor(_nd_uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        self._parameters['weight_ih'] = self.weight_ih
        self._parameters['weight_hh'] = self.weight_hh
        self.bias_ih = self.bias_hh = None
        if bias:
            self.bias_ih = Tensor(_nd_zeros(hidden_size, dtype='float32'), requires_grad=True)
            self.bias_hh = Tensor(_nd_zeros(hidden_size, dtype='float32'), requires_grad=True)
            self._parameters['bias_ih'] = self.bias_ih
            self._parameters['bias_hh'] = self.bias_hh

    def forward(self, x: Tensor, hx: Tensor = None) -> Tensor:
        N = x.shape[0]
        if hx is None:
            hx = _zeros_tensor(N, self.hidden_size)
        pre = x @ self.weight_ih.T + hx @ self.weight_hh.T
        if self.bias_ih is not None:
            pre = pre + self.bias_ih + self.bias_hh
        return pre.tanh() if self.nonlinearity == 'tanh' else pre.relu()


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros
        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor(_nd_uniform(-k, k, (4*hidden_size, input_size)), requires_grad=True)
        self.weight_hh = Tensor(_nd_uniform(-k, k, (4*hidden_size, hidden_size)), requires_grad=True)
        self._parameters['weight_ih'] = self.weight_ih
        self._parameters['weight_hh'] = self.weight_hh
        self.bias_ih = self.bias_hh = None
        if bias:
            b_data = _nd_zeros(4*hidden_size, dtype='float32')
            # Init forget gate bias to 1 for better gradient flow
            for i in range(hidden_size, 2*hidden_size):
                b_data._buf[i] = 1.0
            self.bias_ih = Tensor(b_data, requires_grad=True)
            self.bias_hh = Tensor(_nd_zeros(4*hidden_size, dtype='float32'), requires_grad=True)
            self._parameters['bias_ih'] = self.bias_ih
            self._parameters['bias_hh'] = self.bias_hh

    def forward(self, x: Tensor, hx=None):
        N = x.shape[0]
        if hx is None:
            h = _zeros_tensor(N, self.hidden_size)
            c = _zeros_tensor(N, self.hidden_size)
        else:
            h, c = hx[0], hx[1]
        gates = x @ self.weight_ih.T + h @ self.weight_hh.T
        if self.bias_ih is not None:
            gates = gates + self.bias_ih + self.bias_hh
        hs = self.hidden_size
        i_gate = gates[:, :hs].sigmoid()
        f_gate = gates[:, hs:2*hs].sigmoid()
        g_gate = gates[:, 2*hs:3*hs].tanh()
        o_gate = gates[:, 3*hs:].sigmoid()
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * c_new.tanh()
        return h_new, c_new


class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros
        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor(_nd_uniform(-k, k, (3*hidden_size, input_size)), requires_grad=True)
        self.weight_hh = Tensor(_nd_uniform(-k, k, (3*hidden_size, hidden_size)), requires_grad=True)
        self._parameters['weight_ih'] = self.weight_ih
        self._parameters['weight_hh'] = self.weight_hh
        self.bias_ih = self.bias_hh = None
        if bias:
            from vectorphile._core.ndarray import zeros as _nd_zeros
            self.bias_ih = Tensor(_nd_zeros(3*hidden_size, dtype='float32'), requires_grad=True)
            self.bias_hh = Tensor(_nd_zeros(3*hidden_size, dtype='float32'), requires_grad=True)
            self._parameters['bias_ih'] = self.bias_ih
            self._parameters['bias_hh'] = self.bias_hh

    def forward(self, x: Tensor, hx: Tensor = None) -> Tensor:
        N = x.shape[0]
        _xdt = str(getattr(x._data, 'dtype', 'float32'))
        if hx is None:
            hx = _zeros_tensor(N, self.hidden_size, dtype=_xdt)
        hs = self.hidden_size
        ih = x @ self.weight_ih.T
        hh = hx @ self.weight_hh.T
        if self.bias_ih is not None:
            ih = ih + self.bias_ih
            hh = hh + self.bias_hh
        r = (ih[:, :hs] + hh[:, :hs]).sigmoid()
        z = (ih[:, hs:2*hs] + hh[:, hs:2*hs]).sigmoid()
        n = (ih[:, 2*hs:] + r * hh[:, 2*hs:]).tanh()
        # h_new = (1 - z) * n + z * hx; use scalar 1.0 to avoid dtype mismatch
        h_new = (1.0 - z) * n + z * hx
        return h_new


def _rnn_forward(cells, x, hx_list, num_layers, num_dir, hidden_size, dropout_p, training, batch_first):
    """Shared outer loop for RNN/GRU — returns (output_tensor, h_n_list)."""
    if batch_first:
        x = x.transpose(0, 1)  # (T, N, H)
    T, N, _ = x.shape
    # Slice input into time steps preserving gradient
    layer_input = [x[t] for t in range(T)]
    h_n = []
    for layer in range(num_layers):
        outputs_fwd = []
        cell_fwd = cells[layer * num_dir]
        h_fwd = hx_list[layer * num_dir]
        for t in range(T):
            h_fwd = cell_fwd(layer_input[t], h_fwd)
            outputs_fwd.append(h_fwd)
        h_n_fwd = h_fwd
        if num_dir == 2:
            outputs_bwd = [None] * T
            cell_bwd = cells[layer * num_dir + 1]
            h_bwd = hx_list[layer * num_dir + 1]
            for t in reversed(range(T)):
                h_bwd = cell_bwd(layer_input[t], h_bwd)
                outputs_bwd[t] = h_bwd
            h_n.append(h_n_fwd)
            h_n.append(h_bwd)
            layer_input = [_t_cat([f, b], dim=-1) for f, b in zip(outputs_fwd, outputs_bwd)]
        else:
            h_n.append(h_n_fwd)
            layer_input = outputs_fwd
        if dropout_p > 0 and layer < num_layers - 1 and training:
            mask = _rand_mask((N, layer_input[0].shape[-1]), dropout_p)
            scale = 1.0 / (1.0 - dropout_p)
            layer_input = [t_out * mask * scale for t_out in layer_input]
    output = _t_stack(layer_input, dim=0)  # (T, N, H)
    if batch_first:
        output = output.transpose(0, 1)
    return output, h_n


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
                 bias=True, batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_p = dropout
        num_dir = 2 if bidirectional else 1
        self.cells = []
        for layer in range(num_layers):
            for d in range(num_dir):
                in_sz = input_size if layer == 0 else hidden_size * num_dir
                cell = RNNCell(in_sz, hidden_size, nonlinearity, bias)
                name = f'cell_{layer}_{d}'
                setattr(self, name, cell)
                self.cells.append(cell)

    def forward(self, x: Tensor, hx: Tensor = None):
        num_dir = 2 if self.bidirectional else 1
        N = x.shape[0] if self.batch_first else x.shape[1]
        _xdt = str(getattr(x._data, 'dtype', 'float32'))
        if hx is None:
            hx_list = [_zeros_tensor(N, self.hidden_size, dtype=_xdt) for _ in range(self.num_layers * num_dir)]
        else:
            hx_list = [hx[i] for i in range(self.num_layers * num_dir)]
        output, h_n = _rnn_forward(self.cells, x, hx_list, self.num_layers, num_dir,
                                    self.hidden_size, self.dropout_p, self.training, self.batch_first)
        return output, _t_stack(h_n, dim=0)

    def __repr__(self):
        return (f'RNN(hidden_size={self.hidden_size}, num_layers={self.num_layers}, '
                f'bidirectional={self.bidirectional})')


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_p = dropout
        num_dir = 2 if bidirectional else 1
        self.cells = []
        for layer in range(num_layers):
            for d in range(num_dir):
                in_sz = input_size if layer == 0 else hidden_size * num_dir
                cell = LSTMCell(in_sz, hidden_size, bias)
                name = f'cell_{layer}_{d}'
                setattr(self, name, cell)
                self.cells.append(cell)

    def forward(self, x: Tensor, hx=None):
        num_dir = 2 if self.bidirectional else 1
        N = x.shape[0] if self.batch_first else x.shape[1]
        if hx is None:
            h_list = [_zeros_tensor(N, self.hidden_size) for _ in range(self.num_layers * num_dir)]
            c_list = [_zeros_tensor(N, self.hidden_size) for _ in range(self.num_layers * num_dir)]
        else:
            h_list = [hx[0][i] for i in range(self.num_layers * num_dir)]
            c_list = [hx[1][i] for i in range(self.num_layers * num_dir)]

        # LSTM outer loop (cells return (h, c) tuples)
        if self.batch_first:
            x = x.transpose(0, 1)
        T, N2, _ = x.shape
        layer_input = [x[t] for t in range(T)]
        h_n, c_n = [], []
        for layer in range(self.num_layers):
            outputs_fwd = []
            cell_fwd = self.cells[layer * num_dir]
            h_fwd, c_fwd = h_list[layer * num_dir], c_list[layer * num_dir]
            for t in range(T):
                h_fwd, c_fwd = cell_fwd(layer_input[t], (h_fwd, c_fwd))
                outputs_fwd.append(h_fwd)
            if num_dir == 2:
                outputs_bwd = [None] * T
                cell_bwd = self.cells[layer * num_dir + 1]
                h_bwd, c_bwd = h_list[layer * num_dir + 1], c_list[layer * num_dir + 1]
                for t in reversed(range(T)):
                    h_bwd, c_bwd = cell_bwd(layer_input[t], (h_bwd, c_bwd))
                    outputs_bwd[t] = h_bwd
                h_n.append(h_fwd); h_n.append(h_bwd)
                c_n.append(c_fwd); c_n.append(c_bwd)
                layer_input = [_t_cat([f, b], dim=-1) for f, b in zip(outputs_fwd, outputs_bwd)]
            else:
                h_n.append(h_fwd); c_n.append(c_fwd)
                layer_input = outputs_fwd
            if self.dropout_p > 0 and layer < self.num_layers - 1 and self.training:
                mask = _rand_mask((N2, layer_input[0].shape[-1]), self.dropout_p)
                scale = 1.0 / (1.0 - self.dropout_p)
                layer_input = [t_out * mask * scale for t_out in layer_input]
        output = _t_stack(layer_input, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (_t_stack(h_n, dim=0), _t_stack(c_n, dim=0))

    def __repr__(self):
        return (f'LSTM(hidden_size={self.hidden_size}, num_layers={self.num_layers}, '
                f'bidirectional={self.bidirectional})')


class GRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_p = dropout
        num_dir = 2 if bidirectional else 1
        self.cells = []
        for layer in range(num_layers):
            for d in range(num_dir):
                in_sz = input_size if layer == 0 else hidden_size * num_dir
                cell = GRUCell(in_sz, hidden_size, bias)
                name = f'cell_{layer}_{d}'
                setattr(self, name, cell)
                self.cells.append(cell)

    def forward(self, x: Tensor, hx: Tensor = None):
        num_dir = 2 if self.bidirectional else 1
        N = x.shape[0] if self.batch_first else x.shape[1]
        _xdt = str(getattr(x._data, 'dtype', 'float32'))
        if hx is None:
            hx_list = [_zeros_tensor(N, self.hidden_size, dtype=_xdt) for _ in range(self.num_layers * num_dir)]
        else:
            hx_list = [hx[i] for i in range(self.num_layers * num_dir)]
        output, h_n = _rnn_forward(self.cells, x, hx_list, self.num_layers, num_dir,
                                    self.hidden_size, self.dropout_p, self.training, self.batch_first)
        return output, _t_stack(h_n, dim=0)

    def __repr__(self):
        return (f'GRU(hidden_size={self.hidden_size}, num_layers={self.num_layers}, '
                f'bidirectional={self.bidirectional})')
