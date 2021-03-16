from setuptools import find_packages, setup

# search for version
with open("sfHMM/__init__.py", encoding="utf-8") as f:
    for line in f:
        if (line.startswith("__version__")):
            VERSION = line.strip().split()[-1][1:-1]
            break

kwargs = dict(name="sfHMM",
              version=VERSION,
              description="step finding based HMM",
              author="Hanjin Liu",
              author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
              license="GPLv2",
              packages=find_packages(),
              install_requires=[
                    "hmmlearn>=0.2.3",
                    "scikit-learn",
                    "matplotlib",
                    ],
              python_requires=">=3.6"
              )

setup(**kwargs)