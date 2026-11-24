"""
vectorphile._C — C extension package.

vpcore is built by: python setup_ext.py build_ext --inplace
When not compiled, vectorphile falls back to pure Python automatically.
"""
try:
    from vectorphile._C import vpcore
    AVAILABLE = True
except ImportError:
    vpcore = None
    AVAILABLE = False

__all__ = ['vpcore', 'AVAILABLE']
