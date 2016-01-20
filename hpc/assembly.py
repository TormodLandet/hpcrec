from __future__ import division
import numpy
from .polynomials import eval_phi


def assemble(domain):
    """
    Assemble HPC matrix for the given domain
    """
    N = len(domain.dof_coordinates)
    A = numpy.zeros((N, N), float)
    b = numpy.zeros(N, float)
    
    for dof in range(N):
        neighbours, coeffs, _, _ = eval_phi(domain, dof)
        for i, dof_i in enumerate(neighbours):
            A[dof, dof_i] = -coeffs[i]
        A[dof,dof] = 1
    
    return A, b
