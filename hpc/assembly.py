from __future__ import division
from .linalg import Matrix, Vector
from .polynomials import eval_phi


def assemble(domain):
    """
    Assemble HPC matrix for the given domain
    """
    N = len(domain.dof_coordinates)
    A = Matrix(N, N)
    b = Vector(N)
    
    for dof in range(N):
        neighbours, coeffs, _, _ = eval_phi(domain, dof)
        for i, dof_i in enumerate(neighbours):
            A[dof, dof_i] = -coeffs[i]
        A[dof,dof] = 1
    
    return A, b
