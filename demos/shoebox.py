import numpy as np

import hpcrec
from hpcrec.test_cases.case_shoebox import (
    shoebox_domain,
    shoebox_analytical_solution,
    shoebox_solve_using_hpc,
)


def shoebox_demo(
    N, L, h=1, k=1, show_plot=True, neumann=False, refine=False, fem_order: int = 0
) -> float:
    print(f"Calculating shoebox with N={N}, L={L}")

    # Create geometry
    with hpcrec.Timer("Geometry"):
        domain = shoebox_domain(N=N, L=L, h=h, refine=refine)

    # Assemble global system
    with hpcrec.Timer("Assemble"):
        if fem_order > 0:
            A, b, solve = shoebox_solve_using_fem(domain, fem_order, k, h, neumann)
        else:
            A, b, solve = shoebox_solve_using_hpc(domain, L, k, h, neumann)

    # Print system info
    Ndof = len(b)
    if Ndof < 100:
        maxnz = 0
        for row in A:
            nz = sum(1 if v != 0 else 0 for v in row)
            maxnz = max(maxnz, nz)
        print(f"Maximum number of non zeros in a row: {maxnz}")
        # print 'Condition number', np.linalg.cond(A)

    # Solve global system
    with hpcrec.Timer("Solve"):
        try:
            phi_h = solve()
        except Exception as e:
            print("ERROR:")
            print(e)
            print("The global system matrix cannot be inverted!")
            exit()
    phi_h = phi_h.array()

    # The analytical solution
    phi_a = shoebox_analytical_solution(domain, k, h)

    # Print the error
    error = np.linalg.norm(phi_h - phi_a)
    print(f"Error: {error:15.8e}")

    if show_plot:
        from matplotlib import pyplot as plt

        plt.spy(A.array())
        hpcrec.plot(domain)
        hpcrec.plot(domain, phi_h)
        plt.show()

    return error


def shoebox_solve_using_fem(domain, order, k, h, neumann=False):
    """
    Solve the shoebox wave problem using FEM (FEniCS)

    NOTE: this requires an ancient version of FEniCS (this code is from 2016!)
    """
    import dolfin as df
    from dolfin import grad, dot, dx, ds

    mesh = domain.to_fenics()
    V = df.FunctionSpace(mesh, "CG", order)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)

    # Dirichlet and Neumann functions
    gd = df.Expression("cosh(k*(x[1]+h))*sin(k*x[0])", k=k, h=h)
    gn = df.Expression(
        ["k*cosh(k*(x[1]+h))*cos(k*x[0])", "k*sinh(k*(x[1]+h))*sin(k*x[0])"], k=k, h=h
    )

    # Define and assemble the weak form
    a = dot(grad(u), grad(v)) * dx
    L = dot(gn, n) * v * ds
    A = df.assemble(a)
    b = df.assemble(L)
    print(f"Number of unknowns: {len(b)}")

    # Apply Dirichlet boundary condition
    def dirichlet_boundary(x, on_boundary):
        if neumann:
            return on_boundary and x[1] > h - 1e-8
        else:
            return on_boundary

    dbc = df.DirichletBC(V, gd, dirichlet_boundary)
    dbc.apply(A, b)

    def solve():
        q = df.Function(V)
        nit = df.solve(A, q.vector(), b)
        print(f"Done in {nit} iterations")
        phi_h = q.compute_vertex_values()
        return phi_h

    return A.array(), b.array(), solve


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=10, help="number of elements over the height")
    parser.add_argument(
        "-L",
        type=int,
        default=4,
        help="size of the domain in x in (size in y is 1), must be an integer",
    )
    parser.add_argument("--plot", "-p", action="store_true", help="show plots")
    parser.add_argument("--neumann", "-n", action="store_true", help="include Neumann boundaries")
    parser.add_argument(
        "--fem",
        "-f",
        type=int,
        default=0,
        help="use FEM instead of HPC (specify order of method as argument)",
    )

    parser.add_argument("--backend", choices=("auto", "scipy", "petsc", "numpy"), default="auto")
    parser.add_argument("--solver", default="")
    parser.add_argument("--preconditioner", default="")

    args = parser.parse_args()

    hpcrec.parameters["linear_algebra_backend"] = args.backend
    if args.solver:
        hpcrec.parameters["solver"] = args.solver
    if args.preconditioner:
        hpcrec.parameters["preconditioner"] = args.preconditioner

    with hpcrec.Timer("Shoebox demo"):
        try:
            shoebox_demo(
                N=args.N, L=args.L, show_plot=args.plot, neumann=args.neumann, fem_order=args.fem
            )
        except hpcrec.HPCError as e:
            print("ERROR - " * 9 + "ERROR!!!\n")
            print("   ", e)
            print("\n" + "ERROR - " * 9 + "ERROR!!!\n")
