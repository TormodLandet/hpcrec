"""
This is an implementation of the Harmonic Polynomial Cell method by
Shao & Faltinsen

(c) 2016 - Tormod Landet, original Python 2.7 implementation
(c) 2024 - Tormod Landet, updated to Python 3.14
"""

# Version of the package
__version__ = "2026.06.24"


# Exception
class HPCError(Exception):
    pass


# Global configuration ala FEniCS
parameters = {
    # numpy, scipy, petsc or auto (which is numpy for small matrices)
    "linear_algebra_backend": "auto",
    # Iterative KSP solver parameters
    "relative_tolerance": 1e-15,
    "absolute_tolerance": 1e-15,
    "divergence_limit": 10000,
    "max_iterations": 10000,
    "solver": "gmres",
    "preconditioner": "bjacobi",
}


# Timing utilities
import time


class Timer(object):
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        print(f"STARTING {self.task}")
        self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t_start
        print(f"DONE with {self.task} in {t:.4f} seconds")


# Optional Cython module
try:
    import pyximport

    pyximport.install()
    del pyximport
    has_cython = True
except ImportError:
    has_cython = False

if has_cython:
    from . import hpc_cython
else:
    hpc_cython = None
del has_cython


###############################################################################
# Public API
from .mesh import rectangle_domain, HPCDomain, DOF_TYPE_EXTERNAL, DOF_TYPE_INTERNAL
from .linalg import Matrix, Vector, LinearSolver, solve
from .polynomials import eval_phi
from .assembly import assemble, AssemblyMethod
from .boundary_conditions import apply_bcs, BcType
from .plotting import plot, interactive
