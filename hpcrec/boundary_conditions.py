from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias, Literal

from .polynomials import eval_phi

if TYPE_CHECKING:
    from .mesh import HPCDomain
    from .linalg import GenericMatrix, GenericVector


BcTypeCode: TypeAlias = Literal['D', 'Nx', 'Ny']
BcType: TypeAlias = tuple[BcTypeCode, int, float]


def apply_bcs(domain: HPCDomain, A: GenericMatrix, b: GenericVector, bcs: list[BcType]):
    """
    Apply boundary conditions
    """
    # Apply BCs
    seen_dofs = set()
    for bc_type, dof, value in bcs:
        # Make sure we only have one BC per dof
        assert dof not in seen_dofs
        seen_dofs.add(dof)
        
        if bc_type == 'D':
            for nb in domain.dof_neighbours[dof]:
                A[dof,nb] = 0
            A[dof,dof] = 1
            b[dof] = value
            continue
        
        neighbours, _coeffs, coeffs_diffx, coeffs_diffy = eval_phi(domain, dof)
        
        if bc_type == 'Nx':
            coeffs_diff = coeffs_diffx
        elif bc_type == 'Ny':
            coeffs_diff = coeffs_diffy
        else:
            raise NotImplementedError('BC type %r not implemented' % bc_type)
        
        A[dof,dof] = 0
        for i, dof_i in enumerate(neighbours):
            A[dof, dof_i] = coeffs_diff[i]
        b[dof] = value
