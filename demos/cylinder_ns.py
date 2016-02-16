# encoding: utf8
from __future__ import division
import subprocess
import dolfin as df
from dolfin import dx, div, grad, dot
import numpy
from utilities import mat_to_csr


class NavierStokesDomain(object):
    def __init__(self, inp):
        self.input = inp
        
        if inp.layout == 'I':
            self.num_dividing_lines = 1
        
        self.form = NavierStokesWeakForm(inp)
        self.h = self.form.mesh.hmin()
        
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
        
        self.vel_dir = numpy.zeros(W.dim(), int) - 2
        dividing_line = []
        for ifs, vel_dir in enumerate([0, 1, -1]):
            dofs = W.sub(ifs).dofmap().dofs()
            for dof in dofs:
                self.vel_dir[dof] = vel_dir
                coord = coords[dof]
                if coord[0] < 1e-8:
                    dividing_line.append((dof, coord, vel_dir))
        
        dividing_line.sort(key=lambda item: item[1][0]+item[1][1])
        return dividing_line
    
    def get_pressure_weights(self, coord, dof0, dof1):
        """
        Input: a spatial coordinate coord that points to somewhere in between
        the locations of pressure dofs dof0 and dof1
        
        Output: a list of dofs and a list of weight such that the pressure
        at the input coordinate can be evaluated from a pressure solution
        vector
        """
        coord0 = self.form.dof_coordinates[dof0]
        coord1 = self.form.dof_coordinates[dof1]
        
        d0 = ((coord0[0] - coord[0])**2 + (coord0[1] - coord[1])**2)**0.5
        d1 = ((coord1[0] - coord[0])**2 + (coord1[1] - coord[1])**2)**0.5
        fac = d1/(d0+d1)
        
        #print 'p', coord
        return (dof0, dof1), (fac, 1-fac)
    
    def get_system(self, t):
        """
        Return linear system with normal BCs applied (not coupled)
        Matrix format is SciPy CSR
        """
        A, b = self.form.assemble(self.input.inlet_vel(t), t)
        b_np = b.array()
        A_sp = mat_to_csr(A)
        return A_sp, b_np
    
    def update(self, res):
        """
        Update the values of the function after the time step has been solved
        """
        # Update the coupled vector
        w = self.form.func
        w.vector().set_local(res)
        
        # Spread to the component vectors
        self.form.assigner.assign(self.form.functions, w)
        for func in self.form.functions:
            func.vector().apply('insert') # dolfin bug #587
    
    def get_triangulation(self):
        """
        Get the mesh in a format suitable for matplotlib
        """
        coords = self.form.mesh.coordinates()
        triangles = []
        for cell in df.cells(self.form.mesh):
            cell_vertices = cell.entities(0)
            triangles.append(cell_vertices)
        return coords, triangles
    
    def get_data(self, func_name):
        """
        Get the values at each vertex of the given function
        """
        func = getattr(self.form, func_name)
        return func.compute_vertex_values()


class NavierStokesWeakForm(object):
    def __init__(self, inp):
        self.input = inp
        self.use_lagrange_multiplicator = inp.pressure_lagrange_multiplier
        self.use_supg = inp.use_supg
        self._create_mesh()
        self._create_functions()
        self._create_boundary_conditions()
        self._create_weak_form()
        
        # Show mesh
        #df.plot(self.marker)
        #df.interactive()
        self._tensors = None, None
        
    def assemble(self, U0, t):
        # Update coefficients used in the form
        self.U0.assign(df.Constant(U0))
        self.t.assign(df.Constant(t))
        self.u_conv0.assign(self.u0)
        self.u_conv1.assign(self.u1)
        if self.use_supg:
            self.tau_solver.solve_local_rhs(self.tau)
            if False:
                taua = self.tau.vector().array()
                print
                m1, m2, m3 = taua.min(), taua.mean(), taua.max() 
                print m1, m2, m3
                print m3/m1, m3/m2, m2/m1
        
        # Assemble the form and apply BCs
        a, L = self._weak_form
        A, b = self._tensors
        A, b = df.assemble_system(a, L, self.dirichlet_bcs,
                                  A_tensor=A, b_tensor=b)
        return A, b
    
    def _create_mesh(self):
        # Geometry
        x0, x1 = 0, self.input.l2
        y0, y1 = -self.input.h2/2, self.input.h2/2
        
        if self.input.problem != 'Cylinder':
            # Create mesh
            p0 = df.Point(x0, y0)
            p1 = df.Point(x1, y1)
            Ny = self.input.N2
            Nx = int(round(Ny*self.input.l2/self.input.h2))
            
            self.mesh = df.RectangleMesh(p0, p1, Nx, Ny)
            self.regions = None
            return
        
        # Create unstructured mesh with gmsh
        assert df.MPI.size(df.mpi_comm_world()) == 1
        cmd1 = ['gmsh',
                '-setnumber', 'l2', repr(self.input.l2),
                '-setnumber', 'h2', repr(self.input.h2),
                '-setnumber',  'f', repr(self.input.f),
                '-setnumber',  'd', repr(self.input.d),
                '-setnumber',  'h', repr(self.input.h2/self.input.N2),
                '-2', 'cylinder_gmsh.geo', '-o', 'cylinder_gmsh.msh']
        cmd2 = ['dolfin-convert', 'cylinder_gmsh.msh', 'cylinder_gmsh.xml']
        with open('/dev/null', 'w') as devnull:
            for cmd in (cmd1, cmd2):
                print 'Meshgen: ', ' '.join(cmd)
                subprocess.call(cmd, stdout=devnull, stderr=devnull)
                
        self.mesh = df.Mesh('cylinder_gmsh.xml')
        self.regions = df.MeshFunction('size_t', self.mesh, 'cylinder_gmsh_facet_region.xml')
        assert self.mesh.topology().dim() == 2
    
    def _create_functions(self):
        # Elements and function spaces for the individual components
        cell = self.mesh.ufl_cell()
        e_u = df.FiniteElement('CG', cell, self.input.Pu)
        e_p = df.FiniteElement('CG', cell, self.input.Pp)
        V = df.FunctionSpace(self.mesh, e_u)
        Q = df.FunctionSpace(self.mesh, e_p)
        
        # Current time step
        self.u0 = df.Function(V)
        self.u1 = df.Function(V)
        # Convective velocity
        self.u_conv0 = df.Function(V)
        self.u_conv1 = df.Function(V)
        # Pressure
        self.p = df.Function(Q)
        
        if self.use_lagrange_multiplicator:
            e_l = df.FiniteElement('R', cell, 0)
            L = df.FunctionSpace(self.mesh, e_l)
            self.l = df.Function(L)
            
            elements = [e_u, e_u, e_p, e_l]
            func_spaces = [V, V, Q, L]
            self.functions = [self.u0, self.u1, self.p, self.l]
        else:
            elements = [e_u, e_u, e_p]
            func_spaces = [V, V, Q]
            self.functions = [self.u0, self.u1, self.p]
        
        # Elements and function spaces for the mixed space 
        e_mixed = df.MixedElement(elements)
        W = df.FunctionSpace(self.mesh, e_mixed)
        self.funcspace = W
        self.func = df.Function(W)
        self.assigner = df.FunctionAssigner(func_spaces, W)
        self.dof_coordinates = W.tabulate_dof_coordinates().reshape((-1, 2))
    
    def _create_boundary_conditions(self):
        has_cylinder = self.regions is not None
        
        if has_cylinder:
            marker = self.regions
        
        else:
            x0, x1 = 0, self.input.l2
            y0, y1 = -self.input.h2/2, self.input.h2/2
            
            # Helper to mark regions of the mesh
            def mark(marker, number, selector):
                class Region(df.SubDomain):
                    def inside(self, x, on_boundary):
                        return selector(x, on_boundary)
                region = Region()
                region.mark(marker, number) 
            
            # Mark the boundary facets
            marker = df.FacetFunction('size_t', self.mesh)
            marker.set_all(0)
            mark(marker, 4, lambda x, on_boundary: on_boundary and df.near(x[0], x0)) # inlet
            mark(marker, 2, lambda x, on_boundary: on_boundary and df.near(x[0], x1)) # outlet
            mark(marker, 1, lambda x, on_boundary: on_boundary and df.near(x[1], y0)) # bottom
            mark(marker, 3, lambda x, on_boundary: on_boundary and df.near(x[1], y1)) # top
        
        # External facet region marker and integration measure
        self.marker = marker
        self.ds = df.Measure('ds')(subdomain_data=marker)
        
        # "Constants" that are changed every time step before assembly
        self.U0 = df.Constant(0)
        self.t = df.Constant(0)
        
        # Dirichlet boundary conditions
        W = self.funcspace
        zero = df.Constant(0)
        
        if self.input.problem == 'Cylinder':
            self.dirichlet_bcs = [# Inlet BCs (will be overwritten by coupling)
                                  df.DirichletBC(W.sub(0), self.U0, marker, 2),
                                  df.DirichletBC(W.sub(1), zero, marker, 2),
                                  # Outlet BCs
                                  #df.DirichletBC(W.sub(0), uout, marker, 2),
                                  #df.DirichletBC(W.sub(1), zero, marker, 2),
                                  # Bottom BCs
                                  #df.DirichletBC(W.sub(0), self.U0, marker, 1),
                                  df.DirichletBC(W.sub(1), zero, marker, 1),
                                  # Top BCs
                                  #df.DirichletBC(W.sub(0), self.U0, marker, 3),
                                  df.DirichletBC(W.sub(1), zero, marker, 3),
                                  # Cylinder BCs
                                  df.DirichletBC(W.sub(0), zero, marker, 5),
                                  df.DirichletBC(W.sub(1), zero, marker, 5)]
            self.pressure_ds_boundaries = [1, 3, 4, 5]
            self.pressure_outlet_boundaries = [2]
            self.velocity_ds_boundaries = [1, 2, 3, 4, 5]
        
        elif self.input.problem == 'Taylor-Green':
            params = dict(nu=self.input.mu/self.input.rho,
                          t=self.t,
                          element=W.sub(0).ufl_element())
            f1 = df.Expression('-sin(pi*x[1]) * cos(pi*x[0]) * exp(-2*pi*pi*nu*t)', **params)
            f2 = df.Expression(' sin(pi*x[0]) * cos(pi*x[1]) * exp(-2*pi*pi*nu*t)', **params)
            dbcs = [df.DirichletBC(W.sub(0), f1, lambda x, on_boundary: on_boundary),
                    df.DirichletBC(W.sub(1), f2, lambda x, on_boundary: on_boundary)]
            self.dirichlet_bcs = dbcs
            self.pressure_ds_boundaries = [1, 2, 3, 4]
            self.pressure_outlet_boundaries = []
            self.velocity_ds_boundaries = []
            
            # Initial conditions
            df.project(f1, self.u0.function_space(), function=self.u0)
            df.project(f2, self.u1.function_space(), function=self.u1)
            self.u_expressions = (f1, f2)
    
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
        mu = df.Constant(inp.mu)
        g = df.Constant([0, 0])
        n = df.FacetNormal(self.mesh)
        ds = self.ds
        
        # Lagrange multiplier for the pressure
        if self.use_lagrange_multiplicator:
            lm_trial, lm_test = uc[3], vc[3]
            eq = (p*lm_test + q*lm_trial)*dx
        else:
            eq = 0
        
        # The weak form of the momentum equation and the divergence free criterion
        for d in range(2):
            # Divergence free criterion
            # ∇⋅u = 0
            eq += u[d].dx(d)*q*dx
            
            # Time derivative
            # ∂u/∂t
            eq += rho*(u[d] - up[d])/dt*v[d]*dx
                        
            # Convection
            # ∇⋅(ρ u ⊗ u_conv)
            #eq += div(rho*u[d]*u_conv)*v[d]*dx
            eq += rho*dot(u_conv, grad(u[d]))*v[d]*dx
            
            # Diffusion
            # -∇⋅μ(∇u)
            eq += mu*dot(grad(u[d]), grad(v[d]))*dx
            
            # Pressure
            # ∇p
            eq -= v[d].dx(d)*p*dx
            
            # Body force (gravity)
            # ρ g
            eq -= rho*g[d]*v[d]*dx
        
        # Velocity boundary integrals, from integration by parts
        for region in self.velocity_ds_boundaries:
            # Convection
            #eq += rho*dot(u_conv, n)*dot(u, v)*ds(region)
            # Diffusion
            for d in range(2):
                eq -= mu*dot(grad(u[d]), n)*v[d]*ds(region)
        
        # Pressure boundary integral, from integration by parts
        for region in self.pressure_ds_boundaries:
            eq += p*dot(n, v)*ds(region)
        
        # Pressure boundary integral from outlet boundary condition
        # μ ∂u_n/∂n - p = F_n = 0
        for region in self.pressure_outlet_boundaries:
            un = dot(u, n)
            eq += mu*dot(n, grad(un))*dot(n, v)*ds(region)
        
        # The residual of the momentum equation
        rs = rho*(u - up)/dt
        rs += rho*dot(grad(u), u_conv)
        rs -= mu*div(grad(u))
        rs += grad(p)
        rs -= rho*g
        
        # Add SUPG stabilization
        if self.use_supg:
            # Define param used to weight the stabilization
            a = dot(u_conv, u_conv)**0.5 + df.Constant(2e-16)
            h = df.CellSize(self.mesh)
            nu = mu/rho
            tau = ((2*a/h)**2 + 9*(4*nu/h**2)**2 + (rho/dt)**2)**-0.5
            #tau = h/(2*a)*df.Constant(1)
            
            # Tau has a large polynomial degree making quadrature slow. 
            # Let's bring it down to DG0 via a local projection
            Vtau = df.FunctionSpace(self.mesh, 'DG', 0)
            utau, vtau = df.TrialFunction(Vtau), df.TestFunction(Vtau)
            self.tau_solver = df.LocalSolver(utau*vtau*dx, tau*vtau*dx)
            self.tau_solver.factorize()
            self.tau = df.Function(Vtau)
            
            # Multiply with the residual to ensure consistency
            v_supg = dot(grad(v), u_conv)*self.tau
            eq += dot(v_supg, rs)*dx
        
        # Store the weak form for assembly
        self._weak_form = df.system(eq)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=10,
                        help='number of elements over the height')
    parser.add_argument('-s', '--output-step', type=int, default=1e100,
                        help='timesteps between each generated plot')
    parser.add_argument('--no-supg', action='store_true')
    args = parser.parse_args()
    N = args.N
    output_step = args.output_step
    use_supg = not args.no_supg
    
    ###########################################################################
    # Set up Taylor-Green vortex test case to test the Navier-Stokes
    # solver separately from the potentilal flow and domain coupling
    import time
    from cylinder import Input, plot_domains
    from utilities import SimpleLog
    import scipy.sparse.linalg
    
    log = SimpleLog('taylor-green.log')
    
    inp = Input()
    inp.problem = 'Taylor-Green'
    inp.N1 = inp.N2 = N
    inp.tmax = 10.0
    inp.dt = 0.01
    inp.output_step = output_step
    inp.rho = 1
    inp.U0 = 1/2.5 # same magnitudes in the plots as the Cylinder
    nu = 1e-6
    inp.d = 1
    inp.Re = inp.U0*inp.d/nu
    inp.pressure_lagrange_multiplier = True
    inp.use_supg = use_supg
    
    ns_domain = NavierStokesDomain(inp)
    
    # Time loop
    t = 0
    it = 0
    dt = inp.dt
    rho = inp.rho
    timer_loop_start = time.time()
    while t <= inp.tmax + 1e-6 - dt:
        t += dt
        it += 1
        timer_ts_start = time.time()
        log.info('Timestep %5d  t: %8.4f' % (it, t))
        
        # Assemble the system matrix
        with log.timer('  Assemble: '):
            A, b = ns_domain.get_system(t)
        
        # Solve the system matrix
        with log.timer('  Solve: '): 
            uu = scipy.sparse.linalg.spsolve(A, b)
        
        # Update the solution
        ns_domain.update(uu)
        
        with log.timer('  Plot: '):
            if it % inp.output_step == 0:
                fig = plot_domains(inp, [ns_domain])
                fig.savefig('fig/timestep_%05d_t_%08d.png' % (it, t*1e4), dpi=100)
        
        log.info('  Timestep: %4.2fs' % (time.time() - timer_ts_start))
        
        # Errors
        u0a, u1a = ns_domain.form.u_expressions
        u0 = ns_domain.form.u0
        u1 = ns_domain.form.u1
        eu0 = df.errornorm(u0a, u0, degree_rise=0)
        eu1 = df.errornorm(u1a, u1, degree_rise=0)
        log.info('  ERRORS: %8.2e %8.2e\n' % (eu0, eu1))
    log.info('DONE in %.2fs\n' % (time.time() - timer_loop_start))
