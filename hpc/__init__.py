###############################################################################
# Global configuration ala FEniCS
parameters = {
    # numpy, scipy, petsc or auto (which is numpy for small matrices)
    'linear_algebra_backend': 'auto',
    
    # Iterative KSP solver parameters
    'relative_tolerance': 1e-15,
    'absolute_tolerance': 1e-15,
    'divergence_limit': 10000,
    'max_iterations': 10000,
    'solver': 'gmres',
    'preconditioner': 'hypre'
}


###############################################################################
# Timing utilities
import time
class Timer(object):
    def __init__(self, task):
        self.task = task
    
    def __enter__(self):
        print 'STARTING %s' % self.task
        self.t_start = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t_start
        print 'DONE with %s in %.4f seconds' % (self.task, t) 


###############################################################################
# Public API
from .mesh import rectangle_domain, HPCDomain, DOF_TYPE_EXTERNAL, DOF_TYPE_INTERNAL
from .linalg import Matrix, Vector, LinearSolver, solve
from .polynomials import eval_phi
from .assembly import assemble
from .boundary_conditions import apply_bcs
from .plotting import plot, interactive
