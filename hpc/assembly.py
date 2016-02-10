from __future__ import division
from .linalg import Matrix, Vector
from .polynomials import eval_phi
import numpy


def assemble(domain, method='csr', additional_connections=None):
    """
    Assemble HPC matrix for the given domain
    """
    N = len(domain.dof_coordinates)
    
    if method == 'csr':
        # Assemble into CSR data structures
        # This is faster for SciPy (compared to go via lil to csr)
        # This is slightly slower (very marginal) for PETSc and Numpy
        data, indices, indptr = [], [], [0]
        for dof in range(N):
            neighbours, coeffs, _, _ = eval_phi(domain, dof)
            tmp = zip(neighbours, coeffs)
            tmp.append((dof, -1))
            
            if additional_connections is not None:
                for d in additional_connections[dof]:
                    assert d not in neighbours
                    tmp.append((d, 0))
            
            tmp.sort()
            data.extend(-c for _d, c in tmp)
            indices.extend(d for d, _c in tmp)
            indptr.append(indptr[-1] + len(tmp))
        
        data = numpy.array(data, numpy.float64)
        indices = numpy.array(indices, numpy.int32)
        indptr = numpy.array(indptr, numpy.int32)
        A = Matrix(N, N, data, indices, indptr)
    
    else:
        # Standard assembly. Will be very slow for SciPy since the code
        # to go via LIL to CSR has been removed in favour of direct
        # assembly to CSR data structures (as above)
        A = Matrix(N, N)
        for dof in range(N):
            neighbours, coeffs, _, _ = eval_phi(domain, dof)
            for i, dof_i in enumerate(neighbours):
                A[dof, dof_i] = -coeffs[i]
            A[dof,dof] = 1
        
        if additional_connections is not None:
            for dof, cons in enumerate(additional_connections):
                for d in cons:
                    assert A[dof, d] == 0
                    A[dof, d] = 0
    
    b = Vector(N)
    return A, b
