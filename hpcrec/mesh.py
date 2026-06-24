from __future__ import annotations

import numpy as np


DOF_TYPE_INTERNAL = 0
DOF_TYPE_EXTERNAL = 1


class HPCDomain:
    def __init__(
        self,
        geometric_dimension: int,
        dof_coordinates: np.ndarray,
        dof_type: np.ndarray,
        dof_neighbours: np.ndarray,
        triangles: list[tuple[int, int, int]],
        periodic_x: bool = False,
        grid_shape: tuple[int, int] | None = None,
    ):
        """
        A HPC domain with dof coordinates, dof types and neighbour connectivity.

        Additionally a triangulation is kept for plotting and conversion to simplical meshes for
        FEniCS interoperability.

        ``periodic_x`` signals that the x-direction is periodic (no left/right boundary DOFs).
        ``grid_shape`` stores ``(Nx, Ny)`` for structured grids so column/row indices can be
        derived as ``dof = i * (Ny + 1) + j``.
        """
        self.geometric_dimension: int = geometric_dimension
        self.dof_coordinates: np.ndarray = dof_coordinates
        self.dof_neighbours: np.ndarray = dof_neighbours
        self.dof_type: np.ndarray = dof_type

        # When the domain is periodic in x, the grid shape is stored for convenience so that we
        # can map across the seam between dof indices and (i, j) column/row indices in the grid
        # (obviously only works for structured grids).
        self.periodic_x: bool = periodic_x
        self.grid_shape: tuple[int, int] | None = grid_shape

        # For plotting and FEniCS conversion
        self.triangles: list[tuple[int, int, int]] = triangles

    def to_fenics(self):
        """
        Return a FEniCS mesh for this domain
        """
        return to_fenics(self)


def rectangle_domain(
    p0: tuple[float, float], p1: tuple[float, float], Nx: int, Ny: int,
    periodic_in_x: bool = False,
) -> HPCDomain:
    """
    Generate a rectangular domain.

    When ``periodic_in_x=True`` the x-direction is periodic: ``Nx`` unique columns are
    created (no duplicate endpoint), left/right walls are not marked as boundary DOFs,
    and neighbour stencils wrap modularly in x.  The returned domain has
    ``periodic_x=True`` and ``grid_shape=(Nx, Ny)`` set.
    """
    assert isinstance(Nx, int), f"Expected Nx to be an int, got {type(Nx)} = {Nx}"
    assert isinstance(Ny, int), f"Expected Ny to be an int, got {type(Ny)} = {Ny}"
    assert Nx > 1 and Ny > 1
    x0, y0 = p0
    x1, y1 = p1

    if periodic_in_x:
        return _rectangle_domain_periodic_x(x0, y0, x1, y1, Nx, Ny)

    # Make vertices
    xv = np.linspace(x0, x1, Nx + 1)
    yv = np.linspace(y0, y1, Ny + 1)
    Nv = (Nx + 1) * (Ny + 1)

    domain = HPCDomain(
        geometric_dimension=2,
        dof_coordinates=np.zeros((Nv, 2), float),
        dof_type=np.zeros(Nv, int),
        dof_neighbours=np.zeros((Nv, 8), int),
        triangles=[],
    )

    for i, x in enumerate(xv):
        for j, y in enumerate(yv):
            dof = i * (Ny + 1) + j
            domain.dof_coordinates[dof, :] = x, y

            if i == 0 or i == Nx or j == 0 or j == Ny:
                domain.dof_type[dof] = DOF_TYPE_EXTERNAL
            else:
                domain.dof_type[dof] = DOF_TYPE_INTERNAL

    # Find neighbours
    for i in range(Nx + 1):
        # Find the central dof in the x-direction
        if i == 0:
            center_i = i + 1
        elif i == Nx:
            center_i = i - 1
        else:
            center_i = i

        for j in range(Ny + 1):
            # Find the central dof in the x-direction
            if j == 0:
                center_j = j + 1
            elif j == Ny:
                center_j = j - 1
            else:
                center_j = j

            # Identify dofs in this cell
            d1 = (center_i - 1) * (Ny + 1) + center_j + 1
            d2 = (center_i + 0) * (Ny + 1) + center_j + 1
            d3 = (center_i + 1) * (Ny + 1) + center_j + 1
            d4 = (center_i - 1) * (Ny + 1) + center_j + 0
            d5 = (center_i + 1) * (Ny + 1) + center_j + 0
            d6 = (center_i - 1) * (Ny + 1) + center_j - 1
            d7 = (center_i + 0) * (Ny + 1) + center_j - 1
            d8 = (center_i + 1) * (Ny + 1) + center_j - 1
            d9 = (center_i + 0) * (Ny + 1) + center_j + 0

            # Record the neighbours of the center dof
            dof = i * (Ny + 1) + j
            neighbours = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
            neighbours.remove(dof)
            domain.dof_neighbours[dof, :] = neighbours

            # Add triangles used for plotting
            if i != Nx and j != Ny:
                d1t = (i + 0) * (Ny + 1) + j + 0
                d2t = (i + 0) * (Ny + 1) + j + 1
                d3t = (i + 1) * (Ny + 1) + j + 1
                d4t = (i + 1) * (Ny + 1) + j + 0
                domain.triangles.append((d1t, d2t, d3t))
                domain.triangles.append((d1t, d3t, d4t))

    return domain


def _rectangle_domain_periodic_x(
    x0: float, y0: float, x1: float, y1: float, Nx: int, Ny: int
) -> HPCDomain:
    """
    Build a periodic-in-x rectangular domain.

    Nx unique x-columns, no duplicate right-edge column.  DOF layout::

        dof = i * (Ny + 1) + j,  i in [0, Nx), j in [0, Ny]

    Only top (j==Ny) and bottom (j==0) rows are marked EXTERNAL.
    Neighbour stencils wrap modularly in x.
    """
    Nv = Nx * (Ny + 1)
    xv = np.linspace(x0, x1, Nx, endpoint=False)
    yv = np.linspace(y0, y1, Ny + 1)

    domain = HPCDomain(
        geometric_dimension=2,
        dof_coordinates=np.zeros((Nv, 2), float),
        dof_type=np.zeros(Nv, int),
        dof_neighbours=np.zeros((Nv, 8), int),
        triangles=[],
        periodic_x=True,
        grid_shape=(Nx, Ny),
    )

    # Set coordinates and boundary markers
    for i in range(Nx):
        for j in range(Ny + 1):
            dof = i * (Ny + 1) + j
            domain.dof_coordinates[dof, :] = xv[i], yv[j]
            if j == 0 or j == Ny:
                domain.dof_type[dof] = DOF_TYPE_EXTERNAL
            else:
                domain.dof_type[dof] = DOF_TYPE_INTERNAL

    # Build neighbour stencils with modular x-wrapping so every DOF, including edge columns,
    # references neighbours across the periodic seam.
    # Note: the periodic BCs are also handled in eval_phi by a coordinate-wrapping step before
    # building the local polynomial matrix so that dx is not suddenly ~L across the seam.
    for i in range(Nx):
        il = (i - 1) % Nx
        ir = (i + 1) % Nx
        for j in range(Ny + 1):
            # Mirror inward in y for top/bottom boundary rows
            if j == 0:
                cj = 1
            elif j == Ny:
                cj = Ny - 1
            else:
                cj = j

            # 3×3 stencil centred at (i, cj) with modular x-wrapping
            d1 = il * (Ny + 1) + cj + 1
            d2 = i  * (Ny + 1) + cj + 1
            d3 = ir * (Ny + 1) + cj + 1
            d4 = il * (Ny + 1) + cj
            d5 = ir * (Ny + 1) + cj
            d6 = il * (Ny + 1) + cj - 1
            d7 = i  * (Ny + 1) + cj - 1
            d8 = ir * (Ny + 1) + cj - 1
            d9 = i  * (Ny + 1) + cj  # centre of stencil cell

            dof = i * (Ny + 1) + j
            neighbours = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
            neighbours.remove(dof)
            domain.dof_neighbours[dof, :] = neighbours

            # Triangulation (last column wraps back to first for plotting)
            if j < Ny:
                i_next = (i + 1) % Nx
                d1t = i      * (Ny + 1) + j
                d2t = i      * (Ny + 1) + j + 1
                d3t = i_next * (Ny + 1) + j + 1
                d4t = i_next * (Ny + 1) + j
                domain.triangles.append((d1t, d2t, d3t))
                domain.triangles.append((d1t, d3t, d4t))

    return domain


def to_fenics(domain: HPCDomain) -> "dolfin.Mesh":
    """
    Convert a HPCDomain to a FEniCS mesh

    NOTE: this requires an ancient version of FEniCS (this code is from 2016!)
    """
    import dolfin as df

    # Create the mesh and open for editing
    mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(mesh, 2, 2)

    # Add the vertices
    editor.init_vertices(len(domain.dof_coordinates))
    for i, coord in enumerate(domain.dof_coordinates):
        editor.add_vertex(i, coord[0], coord[1])

    # Add the cells (triangular elements)
    editor.init_cells(len(domain.triangles))
    for i, dofs in enumerate(domain.triangles):
        n0, n1, n2 = dofs
        editor.add_cell(i, n0, n1, n2)

    editor.close()
    return mesh
