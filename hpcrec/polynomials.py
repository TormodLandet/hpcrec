import numpy as np

from hpcrec import hpc_cython, HPCDomain, HPCError


# Harmonic polynomials defined as follows: C x^ex y^ey => (C, ex, ey)
HARMONIC_POLYNOMIALS_2D = (
    ((1, 0, 0),),  # 1: 1
    ((1, 1, 0),),  # 2: x
    ((1, 0, 1),),  # 3: y
    ((1, 2, 0), (-1, 0, 2)),  # 4: x² - y²
    ((2, 1, 1),),  # 5: 2xy
    ((1, 3, 0), (-3, 1, 2)),  # 6: x³ - 3xy²
    ((3, 2, 1), (-1, 0, 3)),  # 7: 3x²y - y³
    ((1, 4, 0), (-6, 2, 2), (1, 0, 4)),  # 8: x⁴ - 6x²y² + y⁴
    ((4, 3, 1), (-4, 1, 3)),  # 9: x³y - xy³
    ((1, 5, 0), (-10, 3, 2), (5, 1, 4)),  # 10: x⁵ - 10x³y² + 5xy⁴
    ((1, 0, 5), (-10, 2, 3), (5, 4, 1)),  # 11: 5x⁴y - 10x²y³ + y⁵
    ((1, 6, 0), (-1, 0, 6), (-15, 4, 2), (15, 2, 4)),  # 12: x⁶ - 15x⁴y² + 15x²y⁴ − y⁶
    ((-20, 3, 3), (6, 1, 5), (6, 5, 1)),  # 13: 6x⁵y - 20x³y³ + 6xy⁵
    ((1, 7, 0), (-21, 5, 2), (-7, 1, 6), (35, 3, 4)),  # 14: x⁷ - 21x⁵y² + 35x³y⁴ - 7xy⁶
    ((-1, 0, 7), (-35, 4, 3), (7, 6, 1), (21, 2, 5)),  # 15: 7x⁶y - 35x⁴y³ + 21x²y⁵ - y⁷
    ((1, 8, 0), (1, 0, 8), (-28.0, 2, 6), (-28.0, 6, 2), (70.0, 4, 4)),  # 16
    ((-56.0, 5, 3), (-8.0, 1, 7), (8.0, 7, 1), (56.0, 3, 5)),  # 17
    ((1, 9, 0), (-84.0, 3, 6), (-36.0, 7, 2), (9.0, 1, 8), (126.0, 5, 4)),  # 18
    ((1, 0, 9), (-84.0, 6, 3), (-36.0, 2, 7), (9.0, 8, 1), (126.0, 4, 5)),  # 19
    ((1, 10, 0), (-1.0, 0, 10), (-210.0, 4, 6), (-45.0, 8, 2), (45.0, 2, 8), (210.0, 6, 4)),  # 20
    ((-120.0, 3, 7), (-120.0, 7, 3), (10.0, 1, 9), (10.0, 9, 1), (252.0, 5, 5)),  # 21
    ((1, 11, 0), (-462.0, 5, 6), (-55.0, 9, 2), (-11.0, 1, 10), (165.0, 3, 8), (330.0, 7, 4)),  # 22
    (
        (-1.0, 0, 11),
        (-330.0, 4, 7),
        (-165.0, 8, 3),
        (11.0, 10, 1),
        (55.0, 2, 9),
        (462.0, 6, 5),
    ),  # 23
    (
        (1, 12, 0),
        (1, 0, 12),
        (-924.0, 6, 6),
        (-66.0, 2, 10),
        (-66.0, 10, 2),
        (495.0, 4, 8),
        (495.0, 8, 4),
    ),  # 24
    (
        (-792.0, 5, 7),
        (-220.0, 9, 3),
        (-12.0, 1, 11),
        (12.0, 11, 1),
        (220.0, 3, 9),
        (792.0, 7, 5),
    ),  # 25
    (
        (1, 13, 0),
        (-1716.0, 7, 6),
        (-286.0, 3, 10),
        (-78.0, 11, 2),
        (13.0, 1, 12),
        (715.0, 9, 4),
        (1287.0, 5, 8),
    ),  # 26
    (
        (1, 0, 13),
        (-1716.0, 6, 7),
        (-286.0, 10, 3),
        (-78.0, 2, 11),
        (13.0, 12, 1),
        (715.0, 4, 9),
        (1287.0, 8, 5),
    ),  # 27
    (
        (1, 14, 0),
        (-1.0, 0, 14),
        (-3003.0, 8, 6),
        (-1001.0, 4, 10),
        (-91.0, 12, 2),
        (91.0, 2, 12),
        (1001.0, 10, 4),
        (3003.0, 6, 8),
    ),  # 28
    (
        (-3432.0, 7, 7),
        (-364.0, 3, 11),
        (-364.0, 11, 3),
        (14.0, 1, 13),
        (14.0, 13, 1),
        (2002.0, 5, 9),
        (2002.0, 9, 5),
    ),  # 29
)


def eval_phi(domain: HPCDomain, dof: int, grad_grad: bool = False):
    r"""
    Calculate the interpolation coefficients for ϕ and its gradient ∇ϕ at a
    given dof.
    
    The only polynomials that contribute to the potential at x=y=0 is f_1 = 1.
    For the derivatives, ∇ϕ, the contribution is only from f_2 and f_3:
    
    .. math::

        \nabla f_2 = [1\ 0]  \qquad\qquad
        \nabla f_3 = [0\ 1]
    
    This function returns dofs and weights for ϕ, ∂ϕ/∂x and ∂ϕ/∂y
    
    If ∇∇ϕ = grad(grad(phi)) is requested then these terms are also calculated,
    and the function returns dofs and weights for ϕ, ∂ϕ/∂x, ∂ϕ/∂y, ∂²ϕ/∂x²,
    ∂²ϕ/∂x∂y, ∂²ϕ/∂y∂x and ∂²ϕ/∂y² to enable evaluating the second order tensor
    
    .. math::
    
        \nabla\nabla\phi = \begin{bmatrix}
                             \partial_{xx} \phi & \partial_{xy} \phi \\
                             \partial_{yx} \phi & \partial_{yy} \phi
                           \end{bmatrix}.
    
    The only polynomials that contribute at x=y=0 are f_4 and f_5:
    
    .. math::

        \nabla\nabla f_4 = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix} \qquad\qquad
        \nabla\nabla f_5 = \begin{bmatrix} 0 & 2 \\ 2 &  0 \end{bmatrix}
    
    The number of return values is hence either 4 or 8                    
    """
    dof_neighbours = domain.dof_neighbours
    dof_coordinates = domain.dof_coordinates
    N = dof_neighbours.shape[1]
    M = np.zeros((N, N), float)

    # For periodic-x domains the modular stencil wraps column Nx-1 as the left
    # neighbour of column 0 (and vice versa).  The stored x-coordinate of the
    # wrapped neighbour is on the far side of the domain (x ≈ L-dx instead of
    # -dx).  We fix this by folding the relative x back into (-L/2, L/2] before
    # building the local polynomial matrix.  The Cython path does not support
    # this wrapping, so we fall through to the Python path for periodic domains.
    x_period: float | None = None
    if domain.periodic_x and domain.grid_shape is not None:
        Nx_g, Ny_g = domain.grid_shape
        # dx = x-spacing between the first two columns
        dx_g = float(dof_coordinates[Ny_g + 1, 0] - dof_coordinates[0, 0])
        x_period = Nx_g * dx_g

    if hpc_cython is not None and x_period is None:
        hpc_cython.setup_local_matrix(dof, dof_neighbours, dof_coordinates, M)
    else:
        x0, y0 = dof_coordinates[dof]

        for i, dof_i in enumerate(dof_neighbours[dof]):
            x, y = dof_coordinates[dof_i]
            xr = x - x0
            yr = y - y0
            if x_period is not None:
                # Fold xr into (-L/2, L/2] for the wrapped periodic neighbour
                if xr > x_period / 2:
                    xr -= x_period
                elif xr < -x_period / 2:
                    xr += x_period

            for j, poly in enumerate(HARMONIC_POLYNOMIALS_2D[:N]):
                fij = 0
                for C, ex, ey in poly:
                    fij += C * xr**ex * yr**ey
                M[i, j] = fij

    try:
        C = np.linalg.inv(M)
    except Exception as e:
        debug_local_matrix_errors(domain, dof, M)
        raise HPCError(
            f"Local matrix is not invertible for dof {dof} at {domain.dof_coordinates[dof]}"
        ) from e

    if not grad_grad:
        return dof_neighbours[dof], C[0, :], C[1, :], C[2, :]

    Cd_dxdx = C[3, :] * 2
    Cd_dxdy = C[4, :] * 2
    Cd_dydx = C[4, :] * 2
    Cd_dydy = C[3, :] * -2
    return dof_neighbours[dof], C[0, :], C[1, :], C[2, :], Cd_dxdx, Cd_dxdy, Cd_dydx, Cd_dydy


def debug_local_matrix_errors(domain, dof, M):
    coord = domain.dof_coordinates[dof]
    print(f"ERROR inverting local matrix for dof {dof} at {coord}")

    np.set_printoptions(linewidth=100000)
    cond = np.linalg.cond(M)
    neighbours = domain.dof_neighbours[dof]

    print("Matrix")
    print(M)
    print("Number of neighbours", len(neighbours), M.shape)
    print(neighbours)
    print("Condition number: %15.5e" % cond)

    x0, y0 = domain.dof_coordinates[dof]
    xn, yn = zip(*[domain.dof_coordinates[nb] for nb in neighbours])

    from matplotlib import pyplot as plt

    fig = plt.figure()
    fig.patch.set_facecolor("white")

    plt.plot(xn, yn, "xb", ms=12)
    plt.plot(x0, y0, "or", ms=12)

    for i, nb in enumerate(neighbours):
        x, y = domain.dof_coordinates[nb]
        plt.text(
            x,
            y,
            "%d" % (i + 1,),
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    plt.gca().set_aspect("equal")
    for f in (plt.xlim, plt.ylim):
        l, h = f()
        d = h - l
        l -= d * 0.1
        h += d * 0.1
        f(l, h)

    # plt.title('DOF %d, cells %r' % (main_dof, [c.index for c in cells]))
    plt.title("Condition number %15.5e" % cond)
    plt.show()
