import pytest
import numpy as np

import hpcrec
from hpcrec.test_cases.case_shoebox import (
    shoebox_analytical_solution,
    shoebox_domain,
    shoebox_solve_using_hpc,
)

# Constants for the shoebox test
N = 10
L = 4.0
h = 1.0
k = 1.0


# @pytest.mark.parametrize("neumann", [False, True], ids=["Dirichlet", "Dirichlet/Neumann"])
@pytest.mark.parametrize("neumann", [False], ids=["Dirichlet"])
@pytest.mark.parametrize("refine", [False, True], ids=["No Refine", "Refine"])
@pytest.mark.parametrize(
    "linear_algebra_backend",
    ["auto", "numpy", "scipy"],
    ids=["Auto", "Numpy", "Scipy"],
)
def test_shoebox_demo(neumann: bool, refine: bool, linear_algebra_backend: str):
    # Create geometry
    with hpcrec.Timer(f"Geometry - shoebox with N={N}, L={L}"):
        domain = shoebox_domain(N=N, L=L, h=h, refine=refine)

    # Assemble and solve using the specified linear algebra backend
    with hpcrec.local_parameters(linear_algebra_backend=linear_algebra_backend):
        # Assemble global system
        with hpcrec.Timer("Assemble"):
            A, b, solve = shoebox_solve_using_hpc(domain, L, k, h, neumann)

        # Solve global system
        Ndof = len(b)
        with hpcrec.Timer(f"Solve with Ndof = {Ndof}"):
            phi_h = solve()

    phi_h = phi_h.array()

    # The analytical solution
    phi_a = shoebox_analytical_solution(domain, k, h)

    # Error limit
    error_lim = 1e-5 if refine else 1e-8

    # Check the error norm
    error = np.linalg.norm(phi_h - phi_a)
    print(f"Error: {error:15.8e}")
    assert error < error_lim
