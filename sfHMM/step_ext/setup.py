from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

sourcefiles = ["stepc.pyx", "StepFinder.cpp"]

setup(
    cmdclass={"build_ext": build_ext},
    name="stepc",
    ext_modules=cythonize(Extension("stepc", sources=sourcefiles, language='c++')),
    include_dirs=[numpy.get_include()]
)