# encoding: utf8
from __future__ import division
from math import cosh, sinh, sin, cos
import numpy
import scipy.sparse.linalg

import hpcrec


def split_shoebox_demo(
    N: int,
    L: float,
    h: float = 1,
    k: float = 1,
    show_plot: bool = True,
    neumann: bool = False,
    refine: bool = False,
):
    print(f"Calculating shoebox with N={N}, L={L}")

    # Create geometry
    domain1 = hpcrec.rectangle_domain((0, 0), (L, h), N * L // h, N)
    domain2 = hpcrec.rectangle_domain((L, 0), (L * 2, h), N * L // h, N)

    # Assemble global systems
    A1, b1 = assemble_side_hpc(domain1, L, k, h)
    A2, b2 = assemble_side_hpc(domain2, L, k, h)

    # Find coupled dofs
    dofs1 = []
    for dof, coord in enumerate(domain1.dof_coordinates):
        if coord[0] > L - 1e-8:  # and coord[1] > 1e-8 and coord[1] < h - 1e-8:
            dofs1.append(dof)
    dofs2 = []
    for dof, coord in enumerate(domain2.dof_coordinates):
        if coord[0] < L + 1e-8:  # and coord[1] > 1e-8 and coord[1] < h - 1e-8:
            dofs2.append(dof)

    dofs1.sort(key=lambda dof: domain1.dof_coordinates[dof][1])
    dofs2.sort(key=lambda dof: domain2.dof_coordinates[dof][1])

    C1 = hpcrec.Matrix(A1.shape[0], A2.shape[1])
    C2 = hpcrec.Matrix(A2.shape[0], A1.shape[1])

    method = "coupled Neuman/Dirichlet"
    if method == "decoupled Neuman/Dirichlet":
        for dof1, dof2 in zip(dofs1, dofs2):
            neighbours1, coeffs_diffx1 = hpcrec.eval_phi(domain1, dof1)[0::2]
            neighbours2 = domain2.dof_neighbours[dof2]

            x1, y1 = domain1.dof_coordinates[dof1]
            x2, y2 = domain2.dof_coordinates[dof2]

            assert x1 == x2 and y1 == y2

            A1[dof1, dof1] = 0
            for i, nb in enumerate(neighbours1):
                A1[dof1, nb] = coeffs_diffx1[i]
                b1[dof1] = k * cosh(k * (y1 + h)) * cos(k * x1)

            for i, nb in enumerate(neighbours2):
                A2[dof2, nb] = 0

            A2[dof2, dof2] = 1
            b2[dof2] = cosh(k * (y2 + h)) * sin(k * x2)

    elif method == "coupled Neuman/Dirichlet":
        for dof1, dof2 in zip(dofs1, dofs2):
            neighbours1, coeffs_diffx1 = hpcrec.eval_phi(domain1, dof1)[0::2]
            neighbours2, coeffs_diffx2 = hpcrec.eval_phi(domain2, dof2)[0::2]

            A1[dof1, dof1] = 0
            b1[dof1] = 0
            for i, nb in enumerate(neighbours1):
                A1[dof1, nb] = coeffs_diffx1[i]

            A2[dof2, dof2] = 1
            b2[dof2] = 0
            for i, nb in enumerate(neighbours2):
                C1[dof1, nb] = -coeffs_diffx2[i]
                A2[dof2, nb] = 0

            C2[dof2, dof1] = -1

    A1 = A1.csc_matrix
    A2 = A2.csc_matrix
    C1 = C1.csc_matrix
    C2 = C2.csc_matrix
    AA = scipy.sparse.bmat([[A1, C1], [C2, A2]], "csc")
    bb = numpy.zeros(len(b1) + len(b2), float)
    bb[: len(b1)] = b1
    bb[len(b1) :] = b2

    if len(bb) < 20:
        print("Block system matrix")
        print("   ", " ".join("%6d" % i for i in range(AA.shape[0])))
        for i, row in enumerate(numpy.array(AA.todense())):
            print("%3d" % i, end=" ")
            for v in row:
                print("%6.2g" % v, end=" ")
            print()
    print(f"AA matrix info: shape {AA.shape} cond {numpy.linalg.cond(AA.todense()):8.2g}")

    # Solve global system
    with hpcrec.Timer("Solve"):
        try:
            lu = scipy.sparse.linalg.splu(AA)
            phi_all = lu.solve(bb)
        except Exception as e:
            print("ERROR:")
            print(e)
            print("The global system matrix cannot be inverted!")
            exit()
    phi1 = phi_all[: len(b1)]
    phi2 = phi_all[len(b1) :]

    # Analytical solution
    for domain_i, phi_i in zip((domain1, domain2), (phi1, phi2)):
        phi_a = numpy.zeros_like(phi_i)
        for dof, coord in enumerate(domain_i.dof_coordinates):
            x, y = coord
            phi_a[dof] = cosh(k * (y + h)) * sin(k * x)

        # Print the error
        print(f"Error: {numpy.linalg.norm(phi_i - phi_a):15.8e}")

    if show_plot:
        from matplotlib import pyplot as plt

        plt.spy(AA.todense())
        domain = add_domains(domain1, domain2)
        hpcrec.plot(domain)
        plt.axvline(L, c="k", ls=":")
        hpcrec.plot(domain, phi_all)
        plt.axvline(L, c="k", ls=":")
        plt.show()


def assemble_side_hpc(domain, L, k, h):
    """
    Solve the shoebox wave problem using HPC
    """
    # Boundary conditions
    bcs = []
    dirichlet_dofs = []
    for dof, coord in enumerate(domain.dof_coordinates):
        if domain.dof_type[dof] == hpcrec.DOF_TYPE_EXTERNAL:
            x, y = coord
            if y > h - 1e-8:
                bcs.append(("D", dof, cosh(k * (y + h)) * sin(k * x)))
                dirichlet_dofs.append(dof)
            elif x < 1e-8:
                bcs.append(("Nx", dof, k * cosh(k * (y + h)) * cos(k * x)))
            elif x > 2 * L - 1e-8:
                bcs.append(("Nx", dof, k * cosh(k * (y + h)) * cos(k * x)))
            elif y < 1e-8:
                bcs.append(("Ny", dof, k * sinh(k * (y + h)) * sin(k * x)))

    # Setup global equation system
    A, b = hpcrec.assemble(domain)
    hpcrec.apply_bcs(domain, A, b, bcs)

    if len(b) < 10:
        print("Global system matrix")
        print("   ", " ".join(f"{i:8d}" for i in range(A.shape[0])))
        for i, row in enumerate(A.array()):
            print(f"{i:3d}", end=" ")
            for v in row:
                print(f"{v:8.2g}", end=" ")
            print()

        print("DOF coordinates")
        for i, c in enumerate(domain.dof_coordinates):
            print(f"{i:3d} - {c[0]:8.2g} {c[1]:8.2g}")

    return A, b


def add_domains(domain1, domain2):
    N1, N2 = len(domain1.dof_coordinates), len(domain2.dof_coordinates)

    domain = hpcrec.HPCDomain()
    domain.geometric_dimension = 2

    for arr_name in ("dof_coordinates", "dof_type", "dof_neighbours"):
        a1 = getattr(domain1, arr_name)
        a2 = getattr(domain2, arr_name)
        shape = list(a1.shape)
        shape[0] += N2
        a = numpy.zeros(shape, a1.dtype)
        a[:N1] = a1
        a[N1:] = a2
        setattr(domain, arr_name, a)

    domain.dof_neighbours[N1:] += N1

    domain.triangles = [t for t in domain1.triangles]
    for d0, d1, d2 in domain2.triangles:
        domain.triangles.append((d0 + N1, d1 + N1, d2 + N1))

    return domain


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

    args = parser.parse_args()

    hpcrec.parameters["linear_algebra_backend"] = "scipy"
    hpcrec.parameters["solver"] = "splu"

    with hpcrec.Timer("Shoebox demo"):
        try:
            split_shoebox_demo(N=args.N, L=args.L, show_plot=args.plot)
        except hpcrec.HPCError as e:
            print("ERROR - " * 9 + "ERROR!!!\n")
            print("   ", e)
            print("\n" + "ERROR - " * 9 + "ERROR!!!\n")
