"""Recurrent layers — RNN, LSTM, GRU with bidirectional and multi-layer support."""
from __future__ import annotations
import numpy as np
import math
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _tanh(x): return np.tanh(x)


class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, nonlinearity='tanh', bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)).astype(np.float32), requires_grad=True)
        self.weight_hh = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)).astype(np.float32), requires_grad=True)
        self._parameters['weight_ih'] = self.weight_ih
        self._parameters['weight_hh'] = self.weight_hh
        self.bias_ih = self.bias_hh = None
        if bias:
            self.bias_ih = Tensor(np.zeros(hidden_size, dtype=np.float32), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(hidden_size, dtype=np.float32), requires_grad=True)
            self._parameters['bias_ih'] = self.bias_ih
            self._parameters['bias_hh'] = self.bias_hh

    def forward(self, x: Tensor, hx: Tensor = None) -> Tensor:
        N = x._data.shape[0]
        if hx is None:
            hx = Tensor(np.zeros((N, self.hidden_size), dtype=np.float32))
        pre = x._data @ self.weight_ih._data.T + hx._data @ self.weight_hh._data.T
        if self.bias_ih is not None:
            pre = pre + self.bias_ih._data + self.bias_hh._data
        h = _tanh(pre) if self.nonlinearity == 'tanh' else np.maximum(0, pre)
        return Tensor(h.astype(np.float32), device=x.device)


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1.0 / math.sqrt(hidden_size)
        # (4*hidden_size) for i, f, g, o gates
        self.weight_ih = Tensor(np.random.uniform(-k, k, (4*hidden_size, input_size)).astype(np.float32), requires_grad=True)
        self.weight_hh = Tensor(np.random.uniform(-k, k, (4*hidden_size, hidden_size)).astype(np.float32), requires_grad=True)
        self._parameters['weight_ih'] = self.weight_ih
        self._parameters['weight_hh'] = self.weight_hh
        self.bias_ih = self.bias_hh = None
        if bias:
            b = np.zeros(4*hidden_size, dtype=np.float32)
            # Init forget gate bias to 1 (helps with gradient flow)
            b[hidden_size:2*hidden_size] = 1.0
            self.bias_ih = Tensor(b.copy(), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(4*hidden_size, dtype=np.float32), requires_grad=True)
            self._parameters['bias_ih'] = self.bias_ih
            self._parameters['bias_hh'] = self.bias_hh

    def forward(self, x: Tensor, hx=None):
        N = x._data.shape[0]
        if hx is None:
            h = np.zeros((N, self.hidden_size), dtype=np.float32)
            c = np.zeros((N, self.hidden_size), dtype=np.float32)
        else:
            h, c = hx[0]._data, hx[1]._data
        gates = x._data @ self.weight_ih._data.T + h @ self.weight_hh._data.T
        if self.bias_ih is not None:
            gates = gates + self.bias_ih._data + self.bias_hh._data
        hs = self.hidden_size
        i_gate = _sigmoid(gates[:, :hs])
        f_gate = _sigmoid(gates[:, hs:2*hs])
        g_gate = _tanh(gates[:, 2*hs:3*hs])
        o_gate = _sigmoid(gates[:, 3*hs:])
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * _tanh(c_new)
        return Tensor(h_new.astype(np.float32)), Tensor(c_new.astype(np.float32))


class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor(np.random.uniform(-k, k, (3*hidden_size, input_size)).astype(np.float32), requires_grad=True)
        self.weight_hh = Tensor(np.random.uniform(-k, k, (3*hidden_size, hidden_size)).astype(np.float32), requires_grad=True)
        self._parameters['weight_ih'] = self.weight_ih
        self._parameters['weight_hh'] = self.weight_hh
        self.bias_ih = self.bias_hh = None
        if bias:
            self.bias_ih = Tensor(np.zeros(3*hidden_size, dtype=np.float32), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(3*hidden_size, dtype=np.float32), requires_grad=True)
            self._parameters['bias_ih'] = self.bias_ih
            self._parameters['bias_hh'] = self.bias_hh

    def forward(self, x: Tensor, hx: Tensor = None) -> Tensor:
        N = x._data.shape[0]
        if hx is None:
            hx = Tensor(np.zeros((N, self.hidden_size), dtype=np.float32))
        h = hx._data
        hs = self.hidden_size
        ih = x._data @ self.weight_ih._data.T
        hh = h @ self.weight_hh._data.T
        if self.bias_ih is not None:
            ih = ih + self.bias_ih._data
            hh = hh + self.bias_hh._data
        r = _sigmoid(ih[:, :hs] + hh[:, :hs])
        z = _sigmoid(ih[:, hs:2*hs] + hh[:, hs:2*hs])
        n = _tanh(ih[:, 2*hs:] + r * hh[:, 2*hs:])
        h_new = (1 - z) * n + z * h
        return Tensor(h_new.astype(np.float32), device=x.device)


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
                self._parameters.update({f'{name}_{k}': v for k, v in cell._parameters.items()})

    def forward(self, x: Tensor, hx: Tensor = None):
        data = x._data
        if self.batch_first:
            data = data.transpose(1, 0, 2)  # (T, N, H)
        T, N, _ = data.shape
        num_dir = 2 if self.bidirectional else 1
        if hx is None:
            h = np.zeros((self.num_layers * num_dir, N, self.hidden_size), dtype=np.float32)
        else:
            h = hx._data
        layer_input = data
        h_n = np.zeros_like(h)
        for layer in range(self.num_layers):
            outputs_fwd = []
            cell_fwd = self.cells[layer * num_dir]
            h_fwd = h[layer * num_dir]
            for t in range(T):
                h_fwd = cell_fwd(Tensor(layer_input[t]), Tensor(h_fwd))._data
                outputs_fwd.append(h_fwd)
            h_n[layer * num_dir] = h_fwd
            if self.bidirectional:
                outputs_bwd = []
                cell_bwd = self.cells[layer * num_dir + 1]
                h_bwd = h[layer * num_dir + 1]
                for t in reversed(range(T)):
                    h_bwd = cell_bwd(Tensor(layer_input[t]), Tensor(h_bwd))._data
                    outputs_bwd.insert(0, h_bwd)
                h_n[layer * num_dir + 1] = h_bwd
                layer_input = np.stack([np.concatenate([f, b], axis=-1) for f, b in zip(outputs_fwd, outputs_bwd)])
            else:
                layer_input = np.stack(outputs_fwd)
            if self.dropout_p > 0 and layer < self.num_layers - 1 and self.training:
                mask = (np.random.rand(*layer_input.shape) > self.dropout_p).astype(np.float32)
                layer_input = layer_input * mask / (1 - self.dropout_p)
        output = layer_input
        if self.batch_first:
            output = output.transpose(1, 0, 2)
        return Tensor(output.astype(np.float32), device=x.device), Tensor(h_n, device=x.device)

    def __repr__(self):
        return (f'RNN(input_size=..., hidden_size={self.hidden_size}, '
                f'num_layers={self.num_layers}, bidirectional={self.bidirectional})')


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
                self._parameters.update({f'{name}_{k}': v for k, v in cell._parameters.items()})

    def forward(self, x: Tensor, hx=None):
        data = x._data
        if self.batch_first:
            data = data.transpose(1, 0, 2)
        T, N, _ = data.shape
        num_dir = 2 if self.bidirectional else 1
        if hx is None:
            h = np.zeros((self.num_layers * num_dir, N, self.hidden_size), dtype=np.float32)
            c = np.zeros((self.num_layers * num_dir, N, self.hidden_size), dtype=np.float32)
        else:
            h, c = hx[0]._data, hx[1]._data
        layer_input = data
        h_n = np.zeros_like(h)
        c_n = np.zeros_like(c)
        for layer in range(self.num_layers):
            outputs_fwd = []
            cell_fwd = self.cells[layer * num_dir]
            h_fwd, c_fwd = h[layer * num_dir], c[layer * num_dir]
            for t in range(T):
                h_t, c_fwd = cell_fwd(Tensor(layer_input[t]), (Tensor(h_fwd), Tensor(c_fwd)))
                h_fwd = h_t._data
                c_fwd = c_fwd._data
                outputs_fwd.append(h_fwd)
            h_n[layer * num_dir] = h_fwd
            c_n[layer * num_dir] = c_fwd
            if self.bidirectional:
                outputs_bwd = []
                cell_bwd = self.cells[layer * num_dir + 1]
                h_bwd, c_bwd = h[layer * num_dir + 1], c[layer * num_dir + 1]
                for t in reversed(range(T)):
                    h_t, c_bwd = cell_bwd(Tensor(layer_input[t]), (Tensor(h_bwd), Tensor(c_bwd)))
                    h_bwd = h_t._data
                    c_bwd = c_bwd._data
                    outputs_bwd.insert(0, h_bwd)
                h_n[layer * num_dir + 1] = h_bwd
                c_n[layer * num_dir + 1] = c_bwd
                layer_input = np.stack([np.concatenate([f, b], axis=-1) for f, b in zip(outputs_fwd, outputs_bwd)])
            else:
                layer_input = np.stack(outputs_fwd)
            if self.dropout_p > 0 and layer < self.num_layers - 1 and self.training:
                mask = (np.random.rand(*layer_input.shape) > self.dropout_p).astype(np.float32)
                layer_input = layer_input * mask / (1 - self.dropout_p)
        output = layer_input
        if self.batch_first:
            output = output.transpose(1, 0, 2)
        return (Tensor(output.astype(np.float32), device=x.device),
                (Tensor(h_n, device=x.device), Tensor(c_n, device=x.device)))

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
                self._parameters.update({f'{name}_{k}': v for k, v in cell._parameters.items()})

    def forward(self, x: Tensor, hx: Tensor = None):
        data = x._data
        if self.batch_first:
            data = data.transpose(1, 0, 2)
        T, N, _ = data.shape
        num_dir = 2 if self.bidirectional else 1
        if hx is None:
            h = np.zeros((self.num_layers * num_dir, N, self.hidden_size), dtype=np.float32)
        else:
            h = hx._data
        layer_input = data
        h_n = np.zeros_like(h)
        for layer in range(self.num_layers):
            outputs_fwd = []
            cell_fwd = self.cells[layer * num_dir]
            h_fwd = h[layer * num_dir]
            for t in range(T):
                h_fwd = cell_fwd(Tensor(layer_input[t]), Tensor(h_fwd))._data
                outputs_fwd.append(h_fwd)
            h_n[layer * num_dir] = h_fwd
            if self.bidirectional:
                outputs_bwd = []
                cell_bwd = self.cells[layer * num_dir + 1]
                h_bwd = h[layer * num_dir + 1]
                for t in reversed(range(T)):
                    h_bwd = cell_bwd(Tensor(layer_input[t]), Tensor(h_bwd))._data
                    outputs_bwd.insert(0, h_bwd)
                h_n[layer * num_dir + 1] = h_bwd
                layer_input = np.stack([np.concatenate([f, b], axis=-1) for f, b in zip(outputs_fwd, outputs_bwd)])
            else:
                layer_input = np.stack(outputs_fwd)
            if self.dropout_p > 0 and layer < self.num_layers - 1 and self.training:
                mask = (np.random.rand(*layer_input.shape) > self.dropout_p).astype(np.float32)
                layer_input = layer_input * mask / (1 - self.dropout_p)
        output = layer_input
        if self.batch_first:
            output = output.transpose(1, 0, 2)
        return Tensor(output.astype(np.float32), device=x.device), Tensor(h_n, device=x.device)
