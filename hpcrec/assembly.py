from typing import TypeAlias, Literal

import numpy

from .mesh import HPCDomain
from .linalg import Matrix, Vector

AssemblyMethod: TypeAlias = Literal["csr", "standard"]


def assemble(domain: HPCDomain, method: AssemblyMethod = "csr", vectorized: bool = True):
    """
    Assemble HPC matrix for the given domain
    """
    N = len(domain.dof_coordinates)
    nb_all, coeffs_all, _, _ = domain.cache.get_all()

    if method == "csr" and vectorized:
        # Vectorised CSR assembly: no Python loop over DOFs.
        #
        # Each row has exactly 9 nonzeros: 8 off-diagonal neighbour entries
        # (negated interpolation coefficients) + 1 diagonal (-1).  We build
        # the three CSR arrays (data, indices, indptr) in one shot using
        # numpy argsort to satisfy the "sorted column indices" requirement.

        # Augment each row with its diagonal term
        diag_col = numpy.arange(N, dtype=nb_all.dtype).reshape(N, 1)  # (N, 1)
        diag_val = numpy.ones((N, 1))  # (N, 1) = +1 on diagonal
        all_col = numpy.concatenate([nb_all, diag_col], axis=1)  # (N, 9)
        all_val = numpy.concatenate([-coeffs_all, diag_val], axis=1)  # (N, 9)

        # Sort each row by column index (required for valid CSR)
        order = numpy.argsort(all_col, axis=1)  # (N, 9)
        data = numpy.take_along_axis(all_val, order, axis=1).ravel()  # (N*9,)
        indices = numpy.take_along_axis(all_col, order, axis=1).ravel()  # (N*9,)
        indptr = numpy.arange(0, N * 9 + 1, 9, dtype=numpy.int32)  # (N+1,)

        A = Matrix(N, N, data.astype(numpy.float64), indices.astype(numpy.int32), indptr)

    elif method == "csr" and not vectorized:
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
