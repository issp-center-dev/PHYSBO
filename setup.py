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
    name="physbo",
    version="0.1",
    author="PHYSBO developers",
    author_email="physbo-dev@issp.u-tokyo.ac.jp",
    packages=[
        "physbo",
        "physbo.misc",
        "physbo.misc._src",
        "physbo.gp",
        "physbo.gp.cov",
        "physbo.gp.cov._src",
        "physbo.gp.mean",
        "physbo.gp.core",
        "physbo.gp.inf",
        "physbo.gp.lik",
        "physbo.opt",
        "physbo.blm.lik",
        "physbo.blm.prior",
        "physbo.blm.basis",
        "physbo.blm.inf",
        "physbo.blm.core",
        "physbo.blm.lik._src",
        "physbo.blm",
        "physbo.search",
        "physbo.search.discrete",
    ],
    package_dir={"physbo": "physbo"},
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_mods,
)
