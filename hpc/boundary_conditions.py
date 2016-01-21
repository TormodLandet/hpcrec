# encoding: utf8
from __future__ import division
from .polynomials import eval_phi


def apply_bcs(domain, A, b, bcs):
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
