import time

class Timer(object):
    def __init__(self, task):
        self.task = task
    
    def __enter__(self):
        print 'STARTING %s' % self.task
        self.t_start = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t_start
        print 'DONE with %s in %.2f seconds' % (self.task, t) 

from .mesh import rectangle_domain, HPCDomain, DOF_TYPE_EXTERNAL, DOF_TYPE_INTERNAL
from .linalg import Matrix, Vector, LinearSolver, solve
from .polynomials import eval_phi
from .assembly import assemble
from .boundary_conditions import apply_bcs
from .plotting import plot, interactive
