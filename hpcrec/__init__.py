"""
This is an implementation of the Harmonic Polynomial Cell method by
Shao & Faltinsen

The "hpcrec" Python package solves for the Laplacian velocity potential using the Harmonic
Polynomial Cell method. It is currently being used mostly for wave kinematics reconstruction with a
known free surface location with associated free surface potential, but you can also implement a
general HPC solver for other problems and even create a numerical wave tank with a moving free
surface by implementing a time-stepping scheme and the appropriate free-surface boundary conditions.

The hpcrec package is available on GitHub: https://github.com/TormodLandet/hpcrec

(c) 2016 - Tormod Landet, original Python 2.7 implementation as a general-purpose HPC solver
(c) 2026 - Tormod Landet, updated to Python 3.12+, focused on HPC wave kinematics reconstruction
"""

# Version of the package
__version__ = "1.0.0"


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
