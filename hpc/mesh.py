# encoding: utf-8
from __future__ import division
import numpy


DOF_TYPE_INTERNAL = 0
DOF_TYPE_EXTERNAL = 1


class HPCDomain(object):
    def __init__(self):
        """
        A HPC domain with dof coordinates, dof types and neighbour
        connectivity. Additionally a triangulation is kept for
        plotting and conversion to simplical meshes for FEniCS
        interoperability. 
        """
        self.geometric_dimension = None
        self.dof_coordinates = None
        self.dof_neighbours = None
        self.dof_type = None
        
        # For plotting and FEniCS conversion
        self.triangles = None
    
    def to_fenics(self):
        """
        Return a FEniCS mesh for this domain
        """
        return to_fenics(self)


def rectangle_domain(p0, p1, Nx, Ny):
    """
    Generate a rectangular domain
    """
    assert isinstance(Nx, (int, long)) and isinstance(Ny, (int, long))
    assert Nx > 1 and Ny > 1
    x0, y0 = p0
    x1, y1 = p1
    
    domain = HPCDomain()
    domain.geometric_dimension = 2
    domain.triangles = []
    
    # Make vertices
    xv = numpy.linspace(x0, x1, Nx+1)
    yv = numpy.linspace(y0, y1, Ny+1)
    Nv = (Nx+1)*(Ny+1)
    domain.dof_coordinates = numpy.zeros((Nv, 2), float)
    domain.dof_type = numpy.zeros(Nv, int)
    for i, x in enumerate(xv):
        for j, y in enumerate(yv):
            dof = i*(Ny+1) + j
            domain.dof_coordinates[dof,:] = x, y
            
            if i == 0 or i == Nx or j == 0 or j == Ny:
                domain.dof_type[dof] = DOF_TYPE_EXTERNAL
            else:
                domain.dof_type[dof] = DOF_TYPE_INTERNAL
    
    # Find neighbours
    domain.dof_neighbours = numpy.zeros((Nv, 8), int)
    for i in range(Nx+1):
        # Find the central dof in the x-direction
        if i == 0:
            center_i = i + 1
        elif i == Nx:
            center_i = i - 1
        else:
            center_i = i
        
        for j in range(Ny+1):
            # Find the central dof in the x-direction
            if j == 0:
                center_j = j + 1
            elif j == Ny:
                center_j = j - 1
            else:
                center_j = j
            
            # Identify dofs in this cell
            d1 = (center_i - 1)*(Ny+1) + center_j + 1
            d2 = (center_i + 0)*(Ny+1) + center_j + 1
            d3 = (center_i + 1)*(Ny+1) + center_j + 1
            d4 = (center_i - 1)*(Ny+1) + center_j + 0
            d5 = (center_i + 1)*(Ny+1) + center_j + 0
            d6 = (center_i - 1)*(Ny+1) + center_j - 1
            d7 = (center_i + 0)*(Ny+1) + center_j - 1
            d8 = (center_i + 1)*(Ny+1) + center_j - 1
            d9 = (center_i + 0)*(Ny+1) + center_j + 0
            
            # Record the neighbours of the center dof
            dof = i*(Ny+1) + j
            neighbours = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
            neighbours.remove(dof)
            domain.dof_neighbours[dof,:] = neighbours
            
            # Add triangles used for plotting
            if i != Nx and j != Ny:
                d1t = (i + 0)*(Ny+1) + j + 0
                d2t = (i + 0)*(Ny+1) + j + 1
                d3t = (i + 1)*(Ny+1) + j + 1
                d4t = (i + 1)*(Ny+1) + j + 0
                domain.triangles.append((d1t, d2t, d3t))
                domain.triangles.append((d1t, d3t, d4t))
    
    return domain


def to_fenics(domain):
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
