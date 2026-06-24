#cython: language_level=2, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
cimport cython

MYFLOAT = np.float
MYINT = np.int
ctypedef np.float_t MYFLOAT_t
ctypedef np.int_t MYINT_t

def setup_local_matrix(unsigned int dof,
                       np.ndarray[MYINT_t, ndim=2] dof_neighbours,
                       np.ndarray[MYFLOAT_t, ndim=2] dof_coordinates,
                       np.ndarray[MYFLOAT_t, ndim=2] M):
    cdef MYINT_t i, dof_i, N
    cdef MYFLOAT_t x0, y0, x, y
    x0  = dof_coordinates[dof,0]
    y0 = dof_coordinates[dof,1]
    
    for i in range(8):
        dof_i = dof_neighbours[dof,i]
        x = dof_coordinates[dof_i, 0] - x0
        y = dof_coordinates[dof_i, 1] - y0
        
        M[i,0] = 1
        M[i,1] = x
        M[i,2] = y
        M[i,3] = x**2 - y**2
        M[i,4] = 2*x*y
        M[i,5] = x**3 - 3*x*y**2
        M[i,6] = 3*x**2*y - y**3
        M[i,7] = x**4 - 6*x**2*y**2 + y**4
