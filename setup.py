from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as base_build_ext
import os

class build_ext(base_build_ext):
    def finalize_options(self):
        """
        Let pyd file be generated in a right place.
        """
        from Cython.Build import cythonize
        import numpy
        sourcefiles = ["stepc.pyx", "StepFinder.cpp"]
        sourcefiles = [os.path.join("sfHMM", "step_ext", f) for f in sourcefiles]
        ext = Extension("sfHMM.step_ext.stepc", 
                        sources=sourcefiles, 
                        language="c++", 
                        include_dirs = [os.path.join("sfHMM", "step_ext"),
                                        numpy.get_include()],
                        )
        self.distribution.ext_modules[:] = cythonize(ext)
        super().finalize_options()

kwargs = dict(name="sfHMM",
              version="1.2.0",
              description="step finding based HMM",
              author="Hanjin Liu",
              author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
              license="GPLv2",
              packages=find_packages(),
              setup_requires=["Cython", "numpy"],
              install_requires=[
                    "hmmlearn>=0.2.3",
                    "scikit-learn",
                    "matplotlib",
                    ],
              python_requires=">=3.6"
              )

try:
    setup(cmdclass={"build_ext": build_ext},
          ext_modules=[Extension("", [])],
          **kwargs)
except:
    # if C compiler is not installed
    setup(**kwargs)