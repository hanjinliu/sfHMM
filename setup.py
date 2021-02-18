from setuptools import setup, find_packages

setup(name="sfHMM",
      version="1.0.0",
      description="step finding based HMM",
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      license="GPLv2",
      packages=find_packages(),
      install_requires=[
            "hmmlearn>=0.2.5",
            "scikit-learn>=0.23.2",
            "matplotlib",
      ],
      python_requires=">=3.6",
      )