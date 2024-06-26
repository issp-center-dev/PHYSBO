# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup, Extension

import numpy

compile_flags = [
    "-O3",
]
ext_mods = [
    Extension(
        name="physbo.misc._src.traceAB",
        sources=["physbo/misc/_src/traceAB.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_flags,
    ),
    Extension(
        name="physbo.misc._src.cholupdate",
        sources=["physbo/misc/_src/cholupdate.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_flags,
    ),
    Extension(
        name="physbo.misc._src.diagAB",
        sources=["physbo/misc/_src/diagAB.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_flags,
    ),
    Extension(
        name="physbo.gp.cov._src.enhance_gauss",
        sources=["physbo/gp/cov/_src/enhance_gauss.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_flags,
    ),
    Extension(
        name="physbo.misc._src.logsumexp",
        sources=["physbo/misc/_src/logsumexp.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_flags,
    ),
]
setup(
    package_dir={"physbo": "physbo"},
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_mods,
)
