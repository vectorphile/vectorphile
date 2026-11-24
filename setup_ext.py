"""
Build script for vectorphile's C extension (vpcore).

Usage:
    python setup_ext.py build_ext --inplace

The extension is optional — vectorphile runs in pure-Python mode without it.
When installed, it provides a ~100-1000x speedup for array arithmetic and matmul.
"""
from setuptools import setup, Extension
import sys
import os

extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/W3']
elif sys.platform == 'darwin':
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-Wall']
else:
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-Wall', '-Wno-unused-variable']
    extra_link_args = ['-lm']

vpcore = Extension(
    'vectorphile._C.vpcore',
    sources=['vectorphile/_C/vpcore.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=['m'] if sys.platform != 'win32' else [],
    language='c',
)

setup(
    name='vectorphile-cext',
    ext_modules=[vpcore],
)
