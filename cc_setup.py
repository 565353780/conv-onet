#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension
import numpy

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension('conv_onet.Lib.libkdtree.pykdtree.kdtree',
                     sources=[
                         'conv_onet/Lib/libkdtree/pykdtree/kdtree.c',
                         'conv_onet/Lib/libkdtree/pykdtree/_kdtree_core.c'
                     ],
                     language='c',
                     extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
                     extra_link_args=['-lgomp'],
                     include_dirs=[numpy_include_dir])

ext_modules = [
    pykdtree,
]

setup(ext_modules=cythonize(ext_modules),
      cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
