from Cython.Build import cythonize
from distutils.core import setup
from Cython.Distutils import build_ext

class build_ext(build_ext):

    def finalize_options(self):
        from Cython.Build import cythonize
        import numpy

        self.distribution.ext_modules[:] = cythonize("**/*.pyx")
        for ext in self.distribution.ext_modules:
            for k, v in numpy.distutils.misc_util.get_info("npymath").items():
                setattr(ext, k, v)
            ext.include_dirs = [numpy.get_include()]

        super().finalize_options()
        
setup(
    cmdclass={"build_ext": build_ext},
    name="_hmmc_motor",
    setup_requires=["Cython", "numpy"],
    ext_modules=cythonize("_hmmc_motor.pyx"),
)