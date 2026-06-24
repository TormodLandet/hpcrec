"""
High-level helpers for reconstructing wave kinematics from HOSM (or any
periodic-domain) free-surface data using the HPC method.

Typical usage::

    domain = create_wave_domain(depth, xgrid, eta)
    phi    = solve_wave_kinematics(domain, psi)
    u, w   = compute_velocity(domain, phi)
"""
from __future__ import annotations

import numpy as np

import hpcrec
from hpcrec.polynomials import eval_phi


def create_wave_domain(
    depth: float,
    x: np.ndarray,
    eta: np.ndarray,
    Nz: int | None = None,
) -> hpcrec.HPCDomain:
    """
    Build a periodic HPC domain for wave kinematics reconstruction.

    The domain has ``Nx`` periodic columns in x and ``Nz + 1`` rows in z.
    The top boundary follows the wave elevation ``eta(x)``; the flat bottom
    is at ``z = -depth``.  Vertical coordinates are linearly stretched column
    by column so that row ``j=0`` sits at ``z = -depth`` and row ``j=Nz``
    sits at ``z = eta[i]``.

    Parameters
    ----------
    depth:
        Water depth (positive value); bottom is at ``z = -depth``.
    x:
        Periodic x-grid of shape ``(Nx,)`` with endpoint **not** repeated
        (e.g. ``np.linspace(0, L, Nx, endpoint=False)``).
    eta:
        Free-surface elevation at each x-grid point, shape ``(Nx,)``.
    Nz:
        Number of vertical intervals.  If *None* (default) the value is
        chosen so that ``dz ≈ dx`` at the free surface, clamped to ``>= 3``.

    Returns
    -------
    HPCDomain
        Deformed periodic domain ready for assembly and solve.
    """
    x = np.asarray(x, dtype=float)
    eta = np.asarray(eta, dtype=float)
    Nx = len(x)
    assert len(eta) == Nx, "x and eta must have the same length"

    dx = x[1] - x[0]
    L = x[-1] + dx  # full periodic domain length

    if Nz is None:
        Nz = max(3, int(round(depth / dx)))

    # Build flat periodic domain as a topological template
    domain = hpcrec.rectangle_domain((0.0, 0.0), (L, 1.0), Nx, Nz, periodic_in_x=True)

    # Deform z-coordinates: column i spans from -depth (j=0) to eta[i] (j=Nz)
    for i in range(Nx):
        z_top = float(eta[i])
        for j in range(Nz + 1):
            dof = i * (Nz + 1) + j
            domain.dof_coordinates[dof, 0] = x[i]
            domain.dof_coordinates[dof, 1] = -depth + (depth + z_top) * j / Nz

    return domain


def solve_wave_kinematics(
    domain: hpcrec.HPCDomain,
    psi: np.ndarray,
) -> np.ndarray:
    """
    Assemble and solve for the velocity potential in the fluid domain.

    Boundary conditions applied:

    * **Top** (free surface, ``j = Nz``): Dirichlet ``phi = psi``
    * **Bottom** (``j = 0``): Neumann ``dphi/dz = 0``

    The left/right sides are periodic by construction of the domain topology
    (no additional constraints needed).

    Parameters
    ----------
    domain:
        Domain created by :func:`create_wave_domain`.
    psi:
        Velocity potential at the free surface, shape ``(Nx,)``.

    Returns
    -------
    np.ndarray, shape ``(Ndof,)``
        Velocity potential ``phi`` at all DOFs.
    """
    assert domain.grid_shape is not None, "Domain must have grid_shape set"
    
    psi = np.asarray(psi, dtype=float)
    Nx, Nz = domain.grid_shape

    bcs: list[hpcrec.BcType] = []
    for i in range(Nx):
        top_dof = i * (Nz + 1) + Nz
        bot_dof = i * (Nz + 1) + 0
        bcs.append(("D",  top_dof, float(psi[i])))
        bcs.append(("Ny", bot_dof, 0.0))

    A, b = hpcrec.assemble(domain)
    hpcrec.apply_bcs(domain, A, b, bcs)

    phi_vec = hpcrec.Vector(len(b))
    with hpcrec.local_parameters(solver="default_direct"):
        hpcrec.solve(A, phi_vec, b)
    return phi_vec.array()


def compute_velocity(
    domain: hpcrec.HPCDomain,
    phi: np.ndarray,
    dofs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the fluid velocity ``(u, w) = (dphi/dx, dphi/dz)`` at given DOFs.

    Parameters
    ----------
    domain:
        The HPC domain.
    phi:
        Velocity potential at all DOFs, shape ``(Ndof,)``.
    dofs:
        DOF indices at which to evaluate.  If *None* (default), all DOFs are
        used and the output arrays have the same length as ``phi``.

    Returns
    -------
    u : np.ndarray
        Horizontal velocity ``dphi/dx`` at the requested DOFs.
    w : np.ndarray
        Vertical velocity ``dphi/dz`` at the requested DOFs.
    """
    phi = np.asarray(phi, dtype=float)
    if dofs is None:
        dofs = list(range(len(phi)))

    u = np.empty(len(dofs))
    w = np.empty(len(dofs))

    for k, dof in enumerate(dofs):
        neighbours, _c, cx, cy = eval_phi(domain, dof)
        u[k] = float(np.dot(cx, phi[neighbours]))
        w[k] = float(np.dot(cy, phi[neighbours]))

    return u, w
