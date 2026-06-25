from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias, Literal

if TYPE_CHECKING:
    from .mesh import HPCDomain
    from .linalg import GenericMatrix, GenericVector


BcTypeCode: TypeAlias = Literal["D", "Nx", "Ny"]
BcType: TypeAlias = tuple[BcTypeCode, int, float]


def apply_bcs(domain: HPCDomain, A: GenericMatrix, b: GenericVector, bcs: list[BcType]):
    """
    Apply boundary conditions
    """
    # Stencil data is fetched lazily from the cache on the first Neumann BC.
    # If assemble() has already been called the cache is already warm and this
    # is just a dict-lookup; otherwise it triggers a full cache population.
    _cache_data: tuple | None = None

    seen_dofs = set()
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

        # Neumann BC: need derivative coefficients from the cache.
        if _cache_data is None:
            nb_all, _, cx_all, cy_all = domain.cache.get_all()
            _cache_data = (nb_all, cx_all, cy_all)
        else:
            nb_all, cx_all, cy_all = _cache_data

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
