# For environments without pandas
try:
    from .pd import *
except ImportError:
    def read(*args, **kwargs):
        raise RuntimeError("io depends on pandas.")
    def read_excel(*args, **kwargs):
        raise RuntimeError("io depends on pandas.")
    def save(*args, **kwargs):
        raise RuntimeError("io depends on pandas.")
    