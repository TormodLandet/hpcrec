# encoding: utf8
from __future__ import division
import dolfin as df
from dolfin import dx, div, grad, dot
import scipy.sparse
import numpy

class NavierStokesDomain(object):
    def __init__(self, inp):
        self.input = inp
        
        if inp.layout == 'I':
            self.num_dividing_lines = 1
        
        self.form = NavierStokesWeakForm(inp)
        
    def get_dividing_line(self, line_number):
        """
        Get the dofs on the given line between the Navier-Stokes and potential 
        flow domains
        
          +---------------------------------
          |          
          |     +---line 1------------------
          |     |     
          |     |<--line 0
          |     |
          |     +---line 2------------------
          | 
          +---------------------------------
        
        The coordinate system is such that x = 0 on line 0 and y=+/- h2/2 on
        lines 1 and 2
        """
        assert line_number == 0
        coords = self.form.dof_coordinates
        W = self.form.funcspace
        
        dividing_line = []
        for ifs, vel_dir in enumerate([0, 1, -1]):
            dofs_u0 = W.sub(ifs).dofmap().dofs()
            for dof in dofs_u0:
                coord = coords[dof]
                if coord[0] == 0:
                    dividing_line.append((dof, coord, vel_dir))
        
        dividing_line.sort(key=lambda item: item[1][0]+item[1][1])
        return dividing_line
    
    def get_pressure_weights(self, coord, dof0, dof1):
        coord0 = self.form.dof_coordinates[dof0]
        coord1 = self.form.dof_coordinates[dof1]
        
        d0 = ((coord0[0] - coord[0])**2 + (coord0[1] - coord[1])**2)**0.5
        d1 = ((coord1[0] - coord[0])**2 + (coord1[1] - coord[1])**2)**0.5
        
        fac = d1/(d0+d1)
        return (dof0, dof1), (fac, 1-fac)
    
    def assemble_csr(self):
        a, L = self.form.weak_form
        bcs = self.form.dirichlet_bcs
        A, b = df.assemble_system(a, L, bcs)
        b_np = b.array()
        A_sp = mat_to_csr(A)
        return A_sp, b_np
    
    def apply_dirichlet(self, A, dof):
        """
        Set row indicated by dof to zero except A[dof,dof] which should be 1
        """
        # Assume this was handled by dirichlet BCs when assembling system
        pass
    
    def update(self, res):
        w = self.form.func
        w.vector().set_local(res)
        u0, u1, p = self.form.u0, self.form.u1, self.form.p
        self.form.assigner.assign([u0, u1, p], w)
        
    def get_triangulation(self):
        # Build triangulation and get the vertex values of the function
        coords = self.form.mesh.coordinates()
        triangles = []
        for cell in df.cells(self.form.mesh):
            cell_vertices = cell.entities(0)
            triangles.append(cell_vertices)
        return coords, triangles
    
    def get_data(self, func_name):
        func = getattr(self.form, func_name)
        return func.compute_vertex_values()


class NavierStokesWeakForm(object):
    def __init__(self, inp):
        self.input = inp
        self._create_mesh()
        self._create_functions()
        self._create_boundary_conditions()
        self._create_weak_form()
    
    def _create_mesh(self):
        # Geometry
        x0, x1 = 0, self.input.l2
        y0, y1 = -self.input.h2/2, self.input.h2/2
        
        # Create mesh
        p0 = df.Point(x0, y0)
        p1 = df.Point(x1, y1)
        Nx = Ny = self.input.N
        self.mesh = df.RectangleMesh(p0, p1, Nx, Ny)
    
    def _create_functions(self):
        # Elements and function spaces for the individual components
        cell = self.mesh.ufl_cell()
        e_u = df.FiniteElement('CG', cell, 2)
        e_p = df.FiniteElement('CG', cell, 2)
        V = df.FunctionSpace(self.mesh, e_u)
        Q = df.FunctionSpace(self.mesh, e_p)
        
        # Current time step
        self.u0 = df.Function(V)
        self.u1 = df.Function(V)
        # Convective velocity
        self.u_conv0 = df.Function(V)
        self.u_conv1 = df.Function(V)
        self.p = df.Function(Q)
        
        # Elements and function spaces for the mixed space 
        e_mixed = df.MixedElement([e_u, e_u, e_p])
        W = df.FunctionSpace(self.mesh, e_mixed)
        self.funcspace = W
        self.func = df.Function(W)
        self.assigner = df.FunctionAssigner([V, V, Q], W)
        self.dof_coordinates = W.tabulate_dof_coordinates().reshape((-1, 2))
    
    def _create_boundary_conditions(self):
        x0, x1 = 0, self.input.l2
        y0, y1 = -self.input.h2/2, self.input.h2/2
        
        class Inlet(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and df.near(x[0], x0)
        
        class Outlet(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and df.near(x[0], x1)
        
        class TopAndBottom(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (df.near(x[1], y0) or df.near(x[1], y1))
        
        self.region_inlet = Inlet()
        self.region_outlet = Outlet()
        self.region_top_bottom = TopAndBottom()
        
        marker = df.FacetFunction('size_t', self.mesh)
        marker.set_all(0)
        self.region_inlet.mark(marker, 1)
        self.region_outlet.mark(marker, 2)
        self.region_top_bottom.mark(marker, 3)
        
        # Dirichlet
        W = self.funcspace
        zero = df.Constant(0)
        self.dirichlet_bcs = [df.DirichletBC(W.sub(0), zero, marker, 1),
                              df.DirichletBC(W.sub(1), zero, marker, 1),
                              df.DirichletBC(W.sub(1), zero, marker, 2),
                              df.DirichletBC(W.sub(1), zero, marker, 3),
                              df.DirichletBC(W.sub(2), zero, marker, 3)]
        
        self.marker = marker
    
    def _create_weak_form(self):
        # Trial and test functions
        uc = df.TrialFunction(self.funcspace)
        vc = df.TestFunction(self.funcspace)
        u = df.as_vector([uc[0], uc[1]])
        v = df.as_vector([vc[0], vc[1]])
        p = uc[2]
        q = vc[2]
        
        up = df.as_vector([self.u0, self.u1])
        u_conv = df.as_vector([self.u_conv0, self.u_conv1])
        
        inp = self.input
        rho = df.Constant(inp.rho)
        dt = df.Constant(inp.dt)
        mu = df.Constant(inp.d*inp.U0*inp.rho/inp.Re)
        g = df.Constant([0, 0])
        
        eq = 0
        for d in range(2):
            # Divergence free criterion
            # ∇⋅u = 0
            eq += u[d].dx(d)*q*dx
            
            # Time derivative
            # ∂u/∂t
            eq += rho*(u[d] - up[d])/dt*v[d]*dx
            
            # Convection
            # ∇⋅(ρ u ⊗ u_conv)
            eq += div(rho*u[d]*u_conv)*v[d]*dx
            
            # Diffusion
            # -∇⋅μ(∇u)
            eq += mu*dot(grad(u[d]), grad(v[d]))*dx
            
            # Pressure
            # ∇p
            eq -= v[d].dx(d)*p*dx
            
            # Body force (gravity)
            # ρ g
            eq -= rho*g[d]*v[d]*dx
        
        # Store the weak form for assembly
        self.weak_form = df.system(eq)


def mat_to_csr(dolfin_matrix):
    """
    Convert any dolfin.Matrix to csr matrix in scipy.
    Based on code by Miro Kuchta
    """
    assert df.MPI.size(df.mpi_comm_world()) == 1, 'mat_to_csr assumes single process'
    
    rows = [0]
    cols = []
    values = []
    for irow in range(dolfin_matrix.size(0)):
        indices, values_ = dolfin_matrix.getrow(irow)
        rows.append(len(indices)+rows[-1])
        cols.extend(indices)
        values.extend(values_)

    shape = dolfin_matrix.size(0), dolfin_matrix.size(1)
        
    return scipy.sparse.csr_matrix((numpy.array(values, dtype='float'),
                                    numpy.array(cols, dtype='int'),
                                    numpy.array(rows, dtype='int')),
                                    shape)
