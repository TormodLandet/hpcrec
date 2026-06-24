from math import cosh, sinh, sin, cos, pi

import numpy as np

import hpcrec


def shoebox_domain(N: int, L: float, h: float = 1.0, refine=False) -> hpcrec.HPCDomain:
    print(f"Calculating shoebox with N={N}, L={L}")

    # Create geometry
    domain = hpcrec.rectangle_domain((0, 0), (L, h), int(N * L / h), N)

    # Apply refinement
    if refine:
        Nd = len(domain.dof_coordinates)
        alpha = 1
        beta_x = 1.0
        beta_y = 0.0
        for i in range(Nd):
            x, y = domain.dof_coordinates[i]
            xc = L / 2 * (1 - cos(pi * x / L) ** alpha)
            yc = h / 2 * (1 - cos(pi * y / h) ** alpha)
            domain.dof_coordinates[i] = (
                beta_x * xc + (1 - beta_x) * x,
                beta_y * yc + (1 - beta_y) * y,
            )

    return domain


def shoebox_analytical_solution(domain: hpcrec.HPCDomain, k: float, h: float) -> np.ndarray:
    """
    Compute the analytical solution
    """
    phi_a = np.zeros(len(domain.dof_coordinates))
    for dof, coord in enumerate(domain.dof_coordinates):
        x, y = coord
        phi_a[dof] = cosh(k * (y + h)) * sin(k * x)
    return phi_a


def shoebox_solve_using_hpc(
    domain: hpcrec.HPCDomain,
    L: float,
    k: float,
    h: float,
    neumann: bool = False,
    show_matrices: bool = False,
):
    """
    Setup and solve the shoebox wave problem using HPC
    """
    # Boundary conditions
    bcs: list[hpcrec.BcType] = []
    dirichlet_dofs: list[int] = []
    for dof, coord in enumerate(domain.dof_coordinates):
        if domain.dof_type[dof] == hpcrec.DOF_TYPE_EXTERNAL:
            x, y = coord
            if not neumann:
                # Use Dirichlet boundary conditions on all boundaries
                bcs.append(("D", dof, cosh(k * (y + h)) * sin(k * x)))
                dirichlet_dofs.append(dof)
            else:
                # Use Neumann boundary conditions on all boundaries except the top boundary
                if y > h - 1e-8:
                    # Top boundary (Dirichlet)
                    bcs.append(("D", dof, cosh(k * (y + h)) * sin(k * x)))
                    dirichlet_dofs.append(dof)
                elif x < 1e-8:
                    # Left boundary (Neumann)
                    bcs.append(("Nx", dof, k * cosh(k * (y + h)) * cos(k * x)))
                elif x > L - 1e-8:
                    # Right boundary (Neumann)
                    bcs.append(("Nx", dof, k * cosh(k * (y + h)) * cos(k * x)))
                else:
                    # Bottom boundary (Neumann)
                    assert y < 1e-8
                    bcs.append(("Ny", dof, k * sinh(k * (y + h)) * sin(k * x)))

    # Setup global equation system
    A, b = hpcrec.assemble(domain)
    hpcrec.apply_bcs(domain, A, b, bcs)
    print(f"Number of unknowns: {len(b)}")

    if show_matrices:
        print("Global system matrix")
        print("   ", " ".join(f"{i:8d}" for i in range(A.shape[0])))
        for i, row in enumerate(A):
            print(f"{i:3d}", end=" ")
            for v in row:
                print(f"{v:8.2g}", end=" ")
            print()

        print("DOF coordinates")
        for i, c in enumerate(domain.dof_coordinates):
            print(f"{i:3d} - {c[0]:8.2g} {c[1]:8.2g}")

    def solve():
        phi_h = hpcrec.Vector(len(b))
        nit = hpcrec.solve(A, phi_h, b)
        print(f"Done in {nit} iterations")
        return phi_h

    return A, b, solve
