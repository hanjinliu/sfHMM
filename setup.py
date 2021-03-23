from setuptools import find_packages, setup, Extension
from Cython.Distutils import build_ext
import os

# search for version
with open("sfHMM/__init__.py", encoding="utf-8") as f:
    for line in f:
        if (line.startswith("__version__")):
            VERSION = line.strip().split()[-1][1:-1]
            break


class build_ext(build_ext):
    def finalize_options(self):
        from Cython.Build import cythonize
        import numpy
        sourcefiles = ["_hmmc_motor.pyx"]
        sourcefiles = [os.path.join("sfHMM", f) for f in sourcefiles]
        ext = Extension("sfHMM._hmmc_motor", 
                        sources=sourcefiles, 
                        include_dirs = ["sfHMM",
                                        numpy.get_include()],
                        )
        self.distribution.ext_modules[:] = cythonize(ext)
        super().finalize_options()
        
kwargs = dict(name="sfHMM",
              version=VERSION,
              description="step finding based HMM",
              author="Hanjin Liu",
              author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
              license="GPLv2",
              packages=find_packages(),
              cmdclass={"build_ext": build_ext},
              setup_requires=["Cython", "numpy"],
              ext_modules=[Extension("", [])],
              install_requires=[
                    "hmmlearn>=0.2.3",
                    "scikit-learn",
                    "matplotlib",
                    ],
              python_requires=">=3.6"
              )

setup(**kwargs)