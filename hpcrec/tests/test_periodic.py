"""
Manufactured-solution tests for the periodic-in-x HPC domain.

The test uses the analytic solution

    phi(x, z) = cosh(k * (z + depth)) * sin(k * x)

which satisfies:

* Laplace equation (by construction -- harmonic polynomial)
* Bottom Neumann:  dphi/dz|_{z=-depth} = k * sinh(0) * sin(kx) = 0
* Periodic in x with period L = 2*pi / k
* Top Dirichlet:   phi(x, 0) = cosh(k * depth) * sin(k * x)
"""

from math import cosh, sin, pi

import numpy as np
import pytest

from hpcrec.test_cases.wave_kinematics import FreeSurfaceConformingDomain, solve_wave_kinematics


@pytest.mark.parametrize("Nx", [16, 32])
@pytest.mark.parametrize("Nz", [8, 16])
def test_periodic_manufactured_solution(Nx: int, Nz: int):
    k = 1.0
    depth = 1.0
    L = 2 * pi / k

    # Periodic x-grid (endpoint not repeated, matching HOSM convention)
    x = np.linspace(0, L, Nx, endpoint=False)

    # Flat free surface (eta = 0) so top is at z = 0
    eta = np.zeros(Nx)

    domain = FreeSurfaceConformingDomain(L, depth, eta, Nz=Nz)

    # Top Dirichlet values: phi(x, 0) = cosh(k * depth) * sin(kx)
    psi = np.array([cosh(k * depth) * sin(k * xi) for xi in x])

    phi = solve_wave_kinematics(domain, psi)

    # Analytical solution at all DOFs
    phi_analytical = np.array(
        [cosh(k * (z + depth)) * sin(k * xd) for xd, z in domain.dof_coordinates]
    )

    error = np.linalg.norm(phi - phi_analytical)
    print(f"Nx={Nx}, Nz={Nz}: absolute error = {error:.3e}")
    # Expected O(dx^4) convergence; threshold is generous for coarse meshes
    dx = L / Nx
    threshold = max(1e-3 * (dx * Nx) ** 4 / Nx, 1e-7)
    assert error < threshold, f"Error {error:.3e} > threshold {threshold:.3e} (Nx={Nx}, Nz={Nz})"
