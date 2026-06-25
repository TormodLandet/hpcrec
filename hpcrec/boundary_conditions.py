from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias, Literal

import numpy as np

if TYPE_CHECKING:
    from .mesh import HPCDomain
    from .linalg import GenericMatrix, GenericVector


BcTypeCode: TypeAlias = Literal["D", "Nx", "Ny"]
BcType: TypeAlias = tuple[BcTypeCode, int, float]


def apply_bcs(domain: HPCDomain, A: GenericMatrix, b: GenericVector, bcs: list[BcType]):
    """
    Apply boundary conditions.

    Uses a fast vectorized path when ``A`` is a :class:`~hpcrec.linalg.ScipyMatrix`
    with the standard 9-nonzeros-per-row CSR structure produced by
    :func:`~hpcrec.assembly.assemble` (the common case).  For all other matrix
    types (PetscMatrix, NumpyMatrix) it falls back to element-by-element updates.
    """
    # Fast path: ScipyMatrix with constant 9-nnz-per-row structure.
    # Directly write into csr.data to avoid ~10 μs Python overhead per
    # element (16 k BCs × 9 elements = ~150 k __setitem__ calls otherwise).
    N = len(domain.dof_coordinates)
    csr = getattr(A, "_csr", None)
    if csr is not None and csr.nnz == N * 9:
        _apply_bcs_scipy_fast(domain, csr, b, bcs)
        return

    # Generic fallback (PetscMatrix, NumpyMatrix, unusual CSR shapes)
    _apply_bcs_loop(domain, A, b, bcs)


def _apply_bcs_scipy_fast(domain, csr, b, bcs: list[BcType]) -> None:
    """
    Vectorized BC application for ScipyMatrix with 9-nnz-per-row CSR.

    Partitions the BC list by type, then updates ``csr.data`` for each group
    in a single NumPy pass (plus an inner loop of 8 for Neumann scatter).
    """
    nb_all, _, cx_all, cy_all = domain.cache.get_all()
    data_arr = csr.data  # (N*9,) – writable view of matrix values
    idx_arr = csr.indices  # (N*9,) – sorted column indices per row
    b_vec = b.array()  # NumpyVector: writable view of RHS

    seen_dofs: set[int] = set()
    d_dofs: list[int] = []
    d_vals: list[float] = []
    nx_dofs: list[int] = []
    nx_vals: list[float] = []
    ny_dofs: list[int] = []
    ny_vals: list[float] = []

    for bc_type, dof, value in bcs:
        assert dof not in seen_dofs, f"Duplicate BC for DOF {dof}"
        seen_dofs.add(dof)
        if bc_type == "D":
            d_dofs.append(dof)
            d_vals.append(value)
        elif bc_type == "Nx":
            nx_dofs.append(dof)
            nx_vals.append(value)
        elif bc_type == "Ny":
            ny_dofs.append(dof)
            ny_vals.append(value)
        else:
            raise NotImplementedError("BC type %r not implemented" % bc_type)

    # ------------------------------------------------------------------
    # Dirichlet: identity row (all off-diagonal → 0, diagonal → 1)
    # ------------------------------------------------------------------
    if d_dofs:
        dd = np.asarray(d_dofs)
        rs = 9 * dd  # row starts (M,)
        flat = (rs[:, None] + np.arange(9)).ravel()  # (M*9,)

        data_arr[flat] = 0.0

        # Locate diagonal within each row: col == dof in sorted columns
        sorted_cols = idx_arr[flat].reshape(len(dd), 9)  # (M, 9)
        diag_off = (sorted_cols == dd[:, None]).argmax(axis=1)  # (M,)
        data_arr[rs + diag_off] = 1.0

        b_vec[dd] = np.asarray(d_vals)

    # ------------------------------------------------------------------
    # Neumann: derivative row  (diagonal → 0, neighbours → coefficients)
    # ------------------------------------------------------------------
    for neu_dofs, neu_vals, coeffs_all in (
        (nx_dofs, nx_vals, cx_all),
        (ny_dofs, ny_vals, cy_all),
    ):
        if not neu_dofs:
            continue
        nd = np.asarray(neu_dofs)
        rs = 9 * nd  # (M,)
        flat = (rs[:, None] + np.arange(9)).ravel()  # (M*9,)

        sorted_cols = idx_arr[flat].reshape(len(nd), 9)  # (M, 9)
        nb_sel = nb_all[nd]  # (M, 8)
        c_sel = coeffs_all[nd]  # (M, 8)

        # Scatter: place each of the 8 per-neighbour coefficients at the
        # correct position in the sorted CSR row.  Diagonal stays 0.
        new_row = np.zeros((len(nd), 9))
        for k in range(8):
            pos = (sorted_cols == nb_sel[:, k : k + 1]).argmax(axis=1)  # (M,)
            new_row[np.arange(len(nd)), pos] = c_sel[:, k]

        data_arr[flat] = new_row.ravel()
        b_vec[nd] = np.asarray(neu_vals)


def _apply_bcs_loop(domain, A, b, bcs: list[BcType]) -> None:
    """
    Generic element-by-element BC application (PetscMatrix, NumpyMatrix, etc.)
    """
    nb_all, _, cx_all, cy_all = domain.cache.get_all()
    seen_dofs: set[int] = set()

    for bc_type, dof, value in bcs:
        # Make sure we only have one BC per dof
        assert dof not in seen_dofs
        seen_dofs.add(dof)

        if bc_type == "D":
            for nb in domain.dof_neighbours[dof]:
                A[dof, nb] = 0
            A[dof, dof] = 1
            b[dof] = value
            continue

        neighbours = nb_all[dof]

        if bc_type == "Nx":
            coeffs_diff = cx_all[dof]
        elif bc_type == "Ny":
            coeffs_diff = cy_all[dof]
        else:
            raise NotImplementedError("BC type %r not implemented" % bc_type)

        A[dof, dof] = 0
        for i, dof_i in enumerate(neighbours):
            A[dof, dof_i] = coeffs_diff[i]
        b[dof] = value
