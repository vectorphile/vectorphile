"""Masked arrays — wraps numpy.ma with vectorphile naming."""
from __future__ import annotations
import numpy as np
import numpy.ma as _ma

# Re-export all numpy.ma classes and functions under our namespace
MaskedArray = _ma.MaskedArray
masked_array = _ma.array
masked = _ma.masked
nomask = _ma.nomask
masked_print_option = _ma.masked_print_option

# Creation
def masked_array(data, mask=_ma.nomask, dtype=None, copy=False, fill_value=None, **kwargs):
    return _ma.array(data, mask=mask, dtype=dtype, copy=copy, fill_value=fill_value, **kwargs)

masked_where = _ma.masked_where
masked_equal = _ma.masked_equal
masked_not_equal = _ma.masked_not_equal
masked_less = _ma.masked_less
masked_less_equal = _ma.masked_less_equal
masked_greater = _ma.masked_greater
masked_greater_equal = _ma.masked_greater_equal
masked_inside = _ma.masked_inside
masked_outside = _ma.masked_outside
masked_values = _ma.masked_values
masked_invalid = _ma.masked_invalid
masked_object = _ma.masked_object

# Operations
getdata = _ma.getdata
getmask = _ma.getmask
getmaskarray = _ma.getmaskarray
filled = _ma.filled
compressed = _ma.compressed
count = _ma.count
make_mask = _ma.make_mask
make_mask_none = _ma.make_mask_none
mask_or = _ma.mask_or

# Arithmetic
add = _ma.add
subtract = _ma.subtract
multiply = _ma.multiply
divide = _ma.divide
power = _ma.power
sqrt = _ma.sqrt
log = _ma.log
exp = _ma.exp
sin = _ma.sin
cos = _ma.cos
arcsin = _ma.arcsin
arccos = _ma.arccos
arctan = _ma.arctan
arctan2 = _ma.arctan2
tan = _ma.tan
hypot = _ma.hypot
conjugate = _ma.conjugate
absolute = _ma.absolute

# Statistics
mean = _ma.mean
median = _ma.median
var = _ma.var
std = _ma.std
sum = _ma.sum
prod = _ma.prod
min = _ma.min
max = _ma.max
ptp = _ma.ptp
cumsum = _ma.cumsum
cumprod = _ma.cumprod
average = _ma.average
anom = _ma.anom
dot = _ma.dot
inner = _ma.inner
outer = _ma.outer
correlate = _ma.correlate
convolve = _ma.convolve

# Manipulation
concatenate = _ma.concatenate
stack = _ma.stack
hstack = _ma.hstack
vstack = _ma.vstack
dstack = _ma.dstack
column_stack = _ma.column_stack
row_stack = _ma.row_stack
reshape = _ma.reshape
resize = _ma.resize
swapaxes = _ma.swapaxes
transpose = _ma.transpose
ravel = _ma.ravel
sort = _ma.sort
argsort = _ma.argsort
argmin = _ma.argmin
argmax = _ma.argmax
nonzero = _ma.nonzero
where = _ma.where
choose = _ma.choose
round_ = _ma.round_
clip = _ma.clip
compress = _ma.compress
take = _ma.take
indices = _ma.indices
ones = _ma.ones
zeros = _ma.zeros
arange = _ma.arange
empty = _ma.empty
ones_like = _ma.ones_like
zeros_like = _ma.zeros_like
empty_like = _ma.empty_like

# I/O helpers
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    return _ma.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def apply_over_axes(func, arr, axes):
    return _ma.apply_over_axes(func, arr, axes)

mask_rowcols = _ma.mask_rowcols
shrink_mask = getattr(_ma, 'shrink_mask', None)
flatnotmasked_contiguous = _ma.flatnotmasked_contiguous
flatnotmasked_edges = _ma.flatnotmasked_edges
notmasked_contiguous = _ma.notmasked_contiguous
notmasked_edges = _ma.notmasked_edges
clump_masked = _ma.clump_masked
clump_unmasked = _ma.clump_unmasked
