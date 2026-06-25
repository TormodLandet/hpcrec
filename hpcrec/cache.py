"""
Per-domain stencil coefficient cache for the HPC method.

The local stencil coefficients (one 8×8 matrix inversion per DOF) are the
dominant cost in both matrix assembly and gradient evaluation.  By caching
them on the domain we pay that cost once per domain lifetime instead of once
per assembly *and* once per velocity-evaluation call.


Cache invalidation
------------------

The cached data depends **only** on ``domain.dof_coordinates`` and
``domain.dof_neighbours``.  It does **not** depend on the solution vector
``phi``, the boundary conditions, or the right-hand-side ``psi``.

The cache becomes stale – and **must** be invalidated – whenever either of
those two arrays is modified in-place after the domain was constructed.

    domain.dof_coordinates[...] = new_coords  # modifying in-place
    domain.cache.invalidate()                 # <- required before next use

Known situations where invalidation IS required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Modifying ``domain.dof_coordinates`` directly to deform the mesh for a new
  time step without recreating the domain object.

* Modifying ``domain.dof_neighbours`` in-place (unusual, but possible via
  low-level mesh manipulation).

Situations where the cache is safe to reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Solving with different boundary-condition values on the same geometry
  (same ``eta``, different ``psi`` / different time steps when the domain
  object is *reused* without coordinate changes).
* Calling :func:`compute_velocity` multiple times for different DOF subsets
  on the same domain.
* Multiple time-step loops that each create a **fresh** domain object – each
  domain carries its own independent cache; there is no cross-contamination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mesh import HPCDomain


class HpcCache:
    """
    Lazy cache of per-DOF stencil coefficients for an :class:`~hpcrec.mesh.HPCDomain`.

    Stores the four arrays returned by :func:`~hpcrec.polynomials.eval_phi`
    for every DOF so the 8×8 local matrix inversion is only performed once
    per domain lifetime instead of separately during assembly *and* during
    velocity evaluation.

    The cache is populated automatically on first use.  Call
    :meth:`populate` explicitly to pre-warm it before a timed solve, or call
    :meth:`invalidate` to discard stale data after modifying the domain
    geometry in-place.
    """

    def __init__(self, domain: HPCDomain) -> None:
        self.domain: HPCDomain = domain
        self._valid: bool = False
        self.neighbours: np.ndarray | None = None  # shape (N, 8), int – neighbour DOF indices
        self.coeffs: np.ndarray | None = None  # shape (N, 8), float – phi interpolation weights
        self.cx: np.ndarray | None = None  # shape (N, 8), float – ∂phi/∂x weights
        self.cy: np.ndarray | None = None  # shape (N, 8), float – ∂phi/∂y weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """True if cached data is present and has not been invalidated."""
        return self._valid

    def invalidate(self) -> None:
        """
        Discard all cached coefficients.

        Call this whenever ``domain.dof_coordinates`` or
        ``domain.dof_neighbours`` have been modified in-place after the
        domain was constructed.  See the module-level docstring for
        examples.
        """
        self._valid = False
        self.neighbours = None
        self.coeffs = None
        self.cx = None
        self.cy = None

    def populate(self) -> None:
        """
        Compute and cache stencil coefficients for **all** DOFs.

        Called automatically by :meth:`get_all` on first access.  You may
        call it explicitly to pre-warm the cache, e.g.::

            domain.cache.populate()
            phi = solve_wave_kinematics(domain, psi)

        Calling this when the cache is already valid will recompute and
        refresh all cached data (equivalent to invalidate + repopulate).
        """
        if self.domain.dof_neighbours.shape[1] == 8:
            self._populate_vectorized()
        else:
            # Fallback for non-standard stencil sizes (rare)
            self._populate_loop()

    def _populate_vectorized(self) -> None:
        """
        Fast vectorized populate: build all local matrices and batch-invert
        in a single NumPy call.  No Python loop over DOFs.
        """
        domain = self.domain
        coords = domain.dof_coordinates  # (N, 2)
        nb = domain.dof_neighbours  # (N, 8)
        N, n_nb = nb.shape

        # Gather neighbour coordinates; compute relative offsets (N, 8)
        nb_coords = coords[nb]  # (N, 8, 2) – fancy index
        xr = nb_coords[:, :, 0] - coords[:, 0:1]  # (N, 8)
        yr = nb_coords[:, :, 1] - coords[:, 1:2]  # (N, 8)

        # Fold the periodic-x seam: xr → (-L/2, L/2]
        # Matches the per-DOF logic in eval_phi exactly.
        if domain.periodic_x and domain.grid_shape is not None:
            Nx_g, Ny_g = domain.grid_shape
            dx_g = float(coords[Ny_g + 1, 0] - coords[0, 0])
            x_period = Nx_g * dx_g
            half = x_period * 0.5
            xr = np.where(xr > half, xr - x_period, np.where(xr < -half, xr + x_period, xr))

        # Build local polynomial matrix M[i, j, k] = poly_k(xr[i,j], yr[i,j])
        # shape (N, 8, 8):  i = DOF index, j = neighbour index, k = polynomial
        # First 8 harmonic polynomials (matching HARMONIC_POLYNOMIALS_2D[:8]):
        x2, y2 = xr * xr, yr * yr
        M = np.stack(
            [
                np.ones((N, n_nb), dtype=np.float64),  # f1 = 1
                xr,  # f2 = x
                yr,  # f3 = y
                x2 - y2,  # f4 = x² − y²
                2.0 * xr * yr,  # f5 = 2xy
                xr * (x2 - 3.0 * y2),  # f6 = x³ − 3xy²
                yr * (3.0 * x2 - y2),  # f7 = 3x²y − y³
                x2 * x2 - 6.0 * x2 * y2 + y2 * y2,  # f8 = x⁴ − 6x²y² + y⁴
            ],
            axis=2,
        )  # (N, 8, 8)

        # Batch matrix inversion: numpy inverts the last two dimensions.
        # C[i] = inv(M[i]); row k of C[i] gives weights so that
        #   value at origin  = C[i, 0, :] · phi[neighbours]
        #   dphi/dx at origin = C[i, 1, :] · phi[neighbours]
        #   dphi/dy at origin = C[i, 2, :] · phi[neighbours]
        C = np.linalg.inv(M)  # (N, 8, 8)

        self.neighbours = nb.copy()
        self.coeffs = np.ascontiguousarray(C[:, 0, :])  # phi interpolation
        self.cx = np.ascontiguousarray(C[:, 1, :])  # ∂phi/∂x
        self.cy = np.ascontiguousarray(C[:, 2, :])  # ∂phi/∂y
        self._valid = True

    def _populate_loop(self) -> None:
        """
        Fallback populate via per-DOF eval_phi calls (handles non-8-neighbour
        stencils and serves as a reference implementation).
        """
        # Late import to break the potential circular import chain:
        #   mesh.py → cache.py → polynomials.py → hpcrec/__init__ → mesh.py
        from .polynomials import eval_phi

        domain = self.domain
        N = len(domain.dof_coordinates)
        n_nb = domain.dof_neighbours.shape[1]
        neighbours = np.empty((N, n_nb), dtype=domain.dof_neighbours.dtype)
        coeffs = np.empty((N, n_nb), dtype=np.float64)
        cx = np.empty((N, n_nb), dtype=np.float64)
        cy = np.empty((N, n_nb), dtype=np.float64)

        for dof in range(N):
            nb_, c, gx, gy = eval_phi(domain, dof)
            neighbours[dof] = nb_
            coeffs[dof] = c
            cx[dof] = gx
            cy[dof] = gy

        self.neighbours = neighbours
        self.coeffs = coeffs
        self.cx = cx
        self.cy = cy
        self._valid = True

    def get_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(neighbours, coeffs, cx, cy)``, populating the cache on
        the first call.

        Returns
        -------
        neighbours : np.ndarray, shape (N, 8), int
            Neighbour DOF indices for each DOF.
        coeffs : np.ndarray, shape (N, 8), float
            Interpolation weights for the potential value :math:`\\phi`.
        cx : np.ndarray, shape (N, 8), float
            Weights for the x-derivative (horizontal velocity component).
        cy : np.ndarray, shape (N, 8), float
            Weights for the y-derivative (vertical velocity component).
        """
        if not self._valid:
            self.populate()
        return self.neighbours, self.coeffs, self.cx, self.cy
