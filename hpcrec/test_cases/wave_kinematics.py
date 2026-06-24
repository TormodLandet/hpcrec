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


class FreeSurfaceConformingDomain(hpcrec.HPCDomain):
    def __init__(
        self,
        length: float,
        depth: float,
        eta: np.ndarray,
        Nz: int | None = None,
        oversample: int = 1,
    ):
        """
        A periodic HPC domain for wave kinematics reconstruction.

        The domain has ``Nx`` periodic columns in x and ``Nz + 1`` rows in z. The top boundary
        follows the wave elevation ``eta(x)`` and the flat bottom is at ``z = -depth``.

        Vertical coordinates are linearly stretched column by column so that row ``j=0`` sits at
        ``z = -depth`` and row ``j=Nz`` sits at ``z = eta[i]``.

        Parameters
        ----------
        length:
            Domain length in the x-direction. Eta is given at uniform x-grid points spanning from
            x=0 to x=length-dx, where dx = length / Nx and Nx = len(eta). Eta at x=length is assumed
            to be periodic with eta[0].
        depth:
            Water depth (positive value); bottom is at ``z = -depth``. Pass ``-1`` to indicate
            infinite / deep water. In that case the effective depth is set automatically to
            something sufficiently deep to avoid bottom influence on the interior solution for most
            wave components (assuming domain length >> interesting wave lengths).
        eta:
            Free-surface elevation at each x-grid point, shape ``(Nx,)``.
        Nz:
            Number of vertical intervals.  If *None* (default) the value is
            chosen so that ``dz ≈ dx`` (at the **oversampled** resolution) at
            the free surface, clamped to ``>= 3``.
        """

        Nx = len(eta)
        x = np.linspace(0.0, length, len(eta), endpoint=False)
        eta = np.asarray(eta, dtype=float)
        dx = x[1] - x[0]

        if depth < 0:
            # Infinite / deep-water case: pick an effective depth that is probably deep enough
            depth = length / 8.0

        if Nz is None:
            # Base Nz on the original dx so z-resolution is independent of oversample
            Nz = max(3, int(round(depth / dx * oversample)))

        if oversample != 1:
            Nx_calc = oversample * Nx
            eta = resample_using_fft(eta, Nx_calc)
            x = np.linspace(0.0, length, Nx_calc, endpoint=False)
            Nx = Nx_calc

        # Build flat periodic domain as a topological template
        domain = hpcrec.rectangle_domain((0.0, 0.0), (length, 1.0), Nx, Nz, periodic_in_x=True)

        # Deform z-coordinates: column i spans from -depth (j=0) to eta[i] (j=Nz)
        for i in range(Nx):
            z_top = float(eta[i])
            for j in range(Nz + 1):
                dof = i * (Nz + 1) + j
                domain.dof_coordinates[dof, 0] = x[i]
                domain.dof_coordinates[dof, 1] = -depth + (depth + z_top) * j / Nz

        # We just copy the data from the rectangle domain we created above
        super().__init__(
            geometric_dimension=2,
            dof_coordinates=domain.dof_coordinates,
            dof_type=domain.dof_type,
            dof_neighbours=domain.dof_neighbours,
            triangles=domain.triangles,
            periodic_x=domain.periodic_x,
            grid_shape=(Nx, Nz),
        )
        self.grid_shape: tuple[int, int]  # Cannot be None in this class

        # Store some wave-domain specific parameters for convenience
        self.length: float = length
        self.depth: float = depth
        self.dx: float = dx
        self.oversample: int = oversample

        # The input with oversampling applied (if any) is stored for reference
        self.x_fs: np.ndarray = x
        self.z_fs: np.ndarray = eta


def solve_wave_kinematics(
    domain: FreeSurfaceConformingDomain,
    psi: np.ndarray,
    oversample: int = 1,
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
    domain: FreeSurfaceConformingDomain
        Periodic wave domain with free-surface-following z-coordinates.
    psi:
        Velocity potential at the free surface at the **original** HOSM
        resolution, shape ``(Nx_orig,)``.
    oversample:
        Must match the value used in :class:`FreeSurfaceConformingDomain`.  ``psi`` is
        upsampled by this factor (FFT-based, via :func:`scipy.signal.resample`)
        before the Dirichlet boundary condition is applied.

    Returns
    -------
    np.ndarray, shape ``(Ndof,)``
        Velocity potential ``phi`` at all DOFs of the (oversampled) domain.
    """
    assert domain.grid_shape is not None, "Domain must have grid_shape set"

    psi = np.asarray(psi, dtype=float)
    Nx, Nz = domain.grid_shape

    if oversample != 1:
        psi = resample_using_fft(psi, oversample * len(psi))

    bcs: list[hpcrec.BcType] = []
    for i in range(Nx):
        top_dof = i * (Nz + 1) + Nz
        bot_dof = i * (Nz + 1) + 0
        bcs.append(("D", top_dof, float(psi[i])))
        bcs.append(("Ny", bot_dof, 0.0))

    A, b = hpcrec.assemble(domain)
    hpcrec.apply_bcs(domain, A, b, bcs)

    phi_vec = hpcrec.Vector(len(b))
    with hpcrec.local_parameters(solver="default_direct"):
        hpcrec.solve(A, phi_vec, b)
    return phi_vec.array()


def compute_velocity(
    domain: FreeSurfaceConformingDomain,
    phi: np.ndarray,
    dofs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the fluid velocity ``(u, w) = (dphi/dx, dphi/dz)`` at given DOFs.

    Parameters
    ----------
    domain: FreeSurfaceConformingDomain
        The HPC wave domain with free-surface-following z-coordinates.
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


def resample_using_fft(signal: np.ndarray, num: int) -> np.ndarray:
    """
    Resample a real periodic signal to ``num`` points using FFT interpolation.

    Parameters
    ----------
    signal:
        1-D input array of length N.
    num:
        Target number of output samples (must be > 0).
    """
    if num == len(signal):
        return signal

    try:
        from scipy.signal import resample
    except ImportError as e:
        raise ImportError("scipy is required for resampling. Please install scipy.") from e

    return resample(signal, num, axis=0)
