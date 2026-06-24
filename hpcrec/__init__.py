"""
This is an implementation of the Harmonic Polynomial Cell method by
Shao & Faltinsen

(c) 2016 - Tormod Landet, original Python 2.7 implementation
(c) 2024 - Tormod Landet, updated to Python 3.14
"""

# Version of the package
__version__ = "2026.06.24"


class HPCError(Exception):
    pass


from .configuration import parameters, get_default_parameters, local_parameters, hpc_cython
from .timer import Timer
from .mesh import rectangle_domain, HPCDomain, DOF_TYPE_EXTERNAL, DOF_TYPE_INTERNAL
from .linalg import Matrix, Vector, LinearSolver, solve
from .polynomials import eval_phi
from .assembly import assemble, AssemblyMethod
from .boundary_conditions import apply_bcs, BcType
from .plotting import plot, interactive
