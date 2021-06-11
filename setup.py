from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
import os

# search for version
with open("sfHMM/__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            VERSION = line.strip().split()[-1][1:-1]
            break


class build_ext(build_ext):
    def finalize_options(self):
        from Cython.Build import cythonize
        import numpy
        import numpy.distutils
        sourcefiles = ["_hmmc_motor.pyx"]
        sourcefiles = [os.path.join("sfHMM", "motor", f) for f in sourcefiles]
        ext = Extension("sfHMM.motor._hmmc_motor", 
                        sources=sourcefiles, 
                        include_dirs = ["sfHMM.motor", numpy.get_include()],
                        )
        self.distribution.ext_modules[:] = cythonize(ext)
        
        for ext in self.distribution.ext_modules:
            for k, v in numpy.distutils.misc_util.get_info("npymath").items():
                setattr(ext, k, v)
            ext.include_dirs = [numpy.get_include()]
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
                    "pandas>=1",
                    ],
              python_requires=">=3.7"
              )

setup(**kwargs)