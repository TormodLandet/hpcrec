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
        # Late import to break the potential circular import chain:
        #   mesh.py → cache.py → polynomials.py → hpcrec/__init__ → mesh.py
        # At the time populate() is first called all modules are fully loaded,
        # so the lazy import resolves correctly.
        from .polynomials import eval_phi

        N = len(self.domain.dof_coordinates)
        neighbours = np.empty((N, 8), dtype=self.domain.dof_neighbours.dtype)
        coeffs = np.empty((N, 8), dtype=np.float64)
        cx = np.empty((N, 8), dtype=np.float64)
        cy = np.empty((N, 8), dtype=np.float64)

        for dof in range(N):
            nb, c, gx, gy = eval_phi(self.domain, dof)
            neighbours[dof] = nb
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
