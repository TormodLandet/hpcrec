from typing import TypeAlias, Literal

import numpy

from .mesh import HPCDomain
from .linalg import Matrix, Vector

AssemblyMethod: TypeAlias = Literal["csr", "standard"]


def assemble(domain: HPCDomain, method: AssemblyMethod = "csr"):
    """
    Assemble HPC matrix for the given domain
    """
    N = len(domain.dof_coordinates)
    nb_all, coeffs_all, _, _ = domain.cache.get_all()

    if method == "csr":
        # Assemble into CSR data structures
        # This is faster for SciPy (compared to go via lil to csr)
        # This is slightly slower (very marginal) for PETSc and Numpy
        data, indices, indptr = [], [], [0]
        for dof in range(N):
            neighbours = nb_all[dof]
            coeffs = coeffs_all[dof]
            tmp = list(zip(neighbours, coeffs))
            tmp.append((dof, -1))
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
            neighbours = nb_all[dof]
            coeffs = coeffs_all[dof]
            for i, dof_i in enumerate(neighbours):
                A[dof, dof_i] = -coeffs[i]
            A[dof, dof] = 1

    b = Vector(N)
    return A, b
