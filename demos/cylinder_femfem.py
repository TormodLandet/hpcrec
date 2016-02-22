# encoding: utf-8
from __future__ import division
import time
import numpy
import dolfin as df
from dolfin import div, grad, dot
from cylinder import Input, plot_domains
from cylinder_ns import NavierStokesDomain
from cylinder_hpc import PotentialFlowDomain
from utilities import SimpleLog, StreamFunction


def main(inp):
    log = SimpleLog('cylinder_femfem.log')
    
    # Make domain meshes
    ns_domain = NavierStokesDomain(inp)
    pf_domain = PotentialFlowDomain(inp)
    
    # Get combined triangulation
    comb_coords, comb_triangles = [], []
    domains = [ns_domain, pf_domain]
    for domain in domains:
        c, t = domain.get_triangulation()
        comb_coords.append(c)
        comb_triangles.append(t)
    Nc = sum(len(c) for c in comb_coords)
    
    # Get coordinates as one long numpy array,
    # and triangles as one long list
    coords = numpy.zeros((Nc, 2), float)
    triangles = []
    start = 0
    coord_map = {}
    coord_id_map = []
    for domain, c, t in zip(domains, comb_coords, comb_triangles):
        # Merge duplicated coordinates
        for x, y in c:
            key = (round(x, 6), round(y, 6))
            coord_id = coord_map.setdefault(key, len(coord_map))
            coord_id_map.append(coord_id)
            coords[coord_id] = x, y
        
        # Renumber triangle vertex numbers
        for v0, v1, v2 in t:
            vid0 = coord_id_map[v0+start]
            vid1 = coord_id_map[v1+start]
            vid2 = coord_id_map[v2+start]
            triangles.append((vid0, vid1, vid2))        
        
        start += len(c)
    coords = coords[:len(coord_map)]
    
    domain = FemFemDomain(inp, coords, triangles, len(comb_triangles[0]))
    
    # Time loop
    t = 0
    it = 0
    dt = inp.dt
    timer_loop_start = time.time()
    stream_function = StreamFunction(ns_domain.u)
    while t <= inp.tmax + 1e-6 - dt:
        t += dt
        it += 1
        timer_ts_start = time.time()
        log.info('Timestep %5d  t: %8.4f' % (it, t))
        
        # Assemble the two system matrices
        with log.timer('  Assemble: '):
            A, b = domain.assemble(t)
        
        # Solve the block matrix system
        with log.timer('  Solve: '): 
            domain.solve(A, b)
        
        with log.timer('  Plot: '):
            if it % inp.output_step == 0:
                fig = plot_domains(inp, [ns_domain, pf_domain], t, stream_function)
                fig.savefig('fig/timestep_%05d_t_%08d.png' % (it, round(t*1e4)), dpi=100)
        
        log.info('  Timestep: %4.2fs' % (time.time() - timer_ts_start))
            
        Fv, Fp = ns_domain.get_force()
        log.info('  Fp: % .3e % .3e  Fv: % .3e % .3e\n' % (Fp[0], Fp[1], Fv[0], Fv[1]))
        
    log.info('DONE in %.2fs\n' % (time.time() - timer_loop_start))
    

class FemFemDomain(object):
    def __init__(self, inp, coords, triangles, Nc_ns):
        """
        A domain consiting of a potential theory subdomain and
        a Navier-Stokes subdomain. Both are implemented with FEniCS
        """
        self.input = inp
        self.use_lagrange_multiplicator = inp.pressure_lagrange_multiplier
        self.use_supg = inp.use_supg
        
        self._create_mesh(coords, triangles, Nc_ns)
        self._create_functions()
        self._create_boundary_conditions()
        self._create_weak_form()
        self.solver = df.PETScLUSolver()
        
    def assemble(self, t):
        """
        Assemble matrices
        """
        self.t.assign(df.Constant(t))
        self.U0.assign(df.Constant(self.input.inlet_vel(t)))
        self.u_conv0.assign(self.u0)
        self.u_conv1.assign(self.u1)
        if self.use_supg:
            self.tau_solver.solve_local_rhs(self.tau)
        
        a, L = self._weak_form
        A, b = df.assemble_system(a, L, self.dirichlet_bcs)
        return A, b

    def solve(self, A, b):
        w = self.func
        self.solver.solve(A, self.func.vector(), b)
        
        # Spread to the component vectors
        self.form.assigner.assign(self.functions, w)
        for func in self.form.functions:
            func.vector().apply('insert') # dolfin bug #587
    
    def pressure_force(self, region=5):
        """
        Integrate the pressure force on the given region
        """
        ds = self.ds(region)
        n = df.FacetNormal(self.mesh)
        Fx = df.assemble(self.p*n[0]*ds)
        Fy = df.assemble(self.p*n[1]*ds)
        return [Fx, Fy]
    
    def viscous_force(self, region=5):
        """
        Integrate the viscous force on the given region
        """
        ds = self.ds(region)
        n = df.FacetNormal(self.mesh)
        u = df.as_vector([self.u0, self.u1])
        sigma_n = dot(grad(u), n)
        Fx = df.assemble(sigma_n[0]*ds)
        Fy = df.assemble(sigma_n[1]*ds)
        return [Fx, Fy]
    
    def _create_mesh(self, coords, triangles, Nc_ns):
        """
        Nc_ns is the number of cells in the Navier-Stokes domain
        """
        Nv = len(coords)
        Nc = len(triangles)
        
        # Generate the mesh
        mesh = self.mesh = df.Mesh()
        editor = df.MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(Nv)
        for i, c in enumerate(coords):
            editor.add_vertex(i, c[0], c[1])
        editor.init_cells(Nc)
        for i, t in enumerate(triangles):
            editor.add_cell(i, t[0], t[1], t[2])
        editor.close()
        
        # Mark the two domains, NS=1, PF=2
        cell_marker = df.CellFunction('size_t', mesh)
        cell_marker.set_all(0)
        for i in range(Nc):
            if i < Nc_ns:
                cell_marker[i] = 1
            else:
                cell_marker[i] = 2
        
        # Mark the facets
        facet_marker = df.FacetFunction('size_t', mesh)
        def mark(marker, number, selector):
            class Region(df.SubDomain):
                def inside(self, x, on_boundary):
                    return selector(x, on_boundary)
            region = Region()
            region.mark(marker, number)
        facet_marker.set_all(0)
        eps = 1e-8
        x0, x1, x2 = -self.input.l1, 0, self.input.l2
        y0, y1 = -self.input.h2/2, self.input.h2/2
        mark(facet_marker, 1, lambda x, ob: df.near(x[1], y0) and x[0] >= x1-eps) # NS bottom
        mark(facet_marker, 2, lambda x, ob: df.near(x[0], x2))                    # NS outlet
        mark(facet_marker, 3, lambda x, ob: df.near(x[1], y1) and x[0] >= x1-eps) # NS top
        mark(facet_marker, 4, lambda x, ob: df.near(x[0], x1))                    # Coupling
        mark(facet_marker, 6, lambda x, ob: df.near(x[1], y1) and x[0] <= x1+eps) # PF top
        mark(facet_marker, 7, lambda x, ob: df.near(x[0], x0))                    # PF inlet
        mark(facet_marker, 8, lambda x, ob: df.near(x[1], y0) and x[0] <= x1+eps) # PF bottom
        mark(facet_marker, 5, lambda x, ob: ob and x1+eps < x[0] < x2-eps and y0+eps < x[1] < y1-eps) # cylinder        
        
        self.mesh = mesh
        self.cell_marker = cell_marker
        self.facet_marker = facet_marker
        self.mesh_ns = df.SubMesh(mesh, cell_marker, 1)
        self.mesh_pf = df.SubMesh(mesh, cell_marker, 2)
        self.dx = df.Measure('dx')(subdomain_data=cell_marker)
        self.ds = df.Measure('ds')(subdomain_data=facet_marker)
        self.dx_ns = self.dx(1)
        self.dx_pf = self.dx(2)
        
        if False:
            # Plot the merged mesh
            df.plot(mesh)
            df.plot(cell_marker)
            df.plot(facet_marker)
            df.interactive()
            exit()
    
    def _create_functions(self):
        # Elements and function spaces for the individual components
        cell = self.mesh.ufl_cell()
        e_u = df.FiniteElement('CG', cell, self.input.Pu)
        e_p = df.FiniteElement('CG', cell, self.input.Pp)
        e_r = df.FiniteElement('CG', cell, self.input.Pu)
        V = df.FunctionSpace(self.mesh_ns, e_u)
        Q = df.FunctionSpace(self.mesh_ns, e_p)
        R = df.FunctionSpace(self.mesh_pf, e_r)
        
        # Functions
        self.u0 = df.Function(V)
        self.u1 = df.Function(V)
        self.u_conv0 = df.Function(V)
        self.u_conv1 = df.Function(V)
        self.p = df.Function(Q)
        self.phi = df.Function(R)
        
        if self.use_lagrange_multiplicator:
            e_l = df.FiniteElement('R', cell, 0)
            L = df.FunctionSpace(self.mesh_ns, e_l)
            self.l = df.Function(L)
            
            elements = [e_u, e_u, e_p, e_r, e_l]
            func_spaces = [V, V, Q, R, L]
            self.functions = [self.u0, self.u1, self.p, self.phi, self.l]
        else:
            elements = [e_u, e_u, e_p, e_r]
            func_spaces = [V, V, Q, R]
            self.functions = [self.u0, self.u1, self.p, self.phi]
        
        # Elements and function spaces for the mixed space 
        e_mixed = df.MixedElement(elements)
        W = df.FunctionSpace(self.mesh, e_mixed)
        self.funcspace = W
        self.func = df.Function(W)
        #self.assigner = df.FunctionAssigner(func_spaces, W)
    
    def _create_boundary_conditions(self):        
        # "Constants" that are changed every time step before assembly
        self.U0 = df.Constant(0)
        self.t = df.Constant(0)
        
        # Dirichlet boundary conditions
        W = self.funcspace
        zero = df.Constant(0)
        
        marker = self.facet_marker
        self.dirichlet_bcs = [# Outlet BCs
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
        
        self.coupled_boundaries = [4]
        self.pressure_ds_boundaries = [1, 3, 5]
        self.pressure_outlet_boundaries = [2]
        self.velocity_ds_boundaries = [1, 2, 3, 5]
        self.potential_ds_boundaries = [6, 8]
        self.potential_inflow_boundaries = [7]
        
        if not self.input.coupled_domains:
            # NS inlet BCs
            self.dirichlet_bcs.append(df.DirichletBC(W.sub(0), self.U0, marker, 4))
            self.dirichlet_bcs.append(df.DirichletBC(W.sub(1), zero, marker, 4))
            # PF outlet BC
            self.dirichlet_bcs.append(df.DirichletBC(W.sub(3), zero, marker, 4))
    
    
    def _create_weak_form(self):
        # Trial and test functions
        uc = df.TrialFunction(self.funcspace)
        vc = df.TestFunction(self.funcspace)
        u = df.as_vector([uc[0], uc[1]])
        v = df.as_vector([vc[0], vc[1]])
        p = uc[2]
        q = vc[2]
        phi = uc[3]
        r = vc[3]
        
        up = df.as_vector([self.u0, self.u1])
        u_conv = df.as_vector([self.u_conv0, self.u_conv1])
        
        inp = self.input
        rho = df.Constant(inp.rho)
        dt = df.Constant(inp.dt)
        mu = df.Constant(inp.mu)
        g = df.Constant([0, 0])
        n = df.FacetNormal(self.mesh)
        ds = self.ds
        dx_ns = self.dx_ns
        dx_pf = self.dx_pf
        zero = df.Constant(0)
        
        # Lagrange multiplier for the pressure
        if self.use_lagrange_multiplicator:
            lm_trial, lm_test = uc[4], vc[4]
            eq = (p*lm_test + q*lm_trial)*dx_ns
        else:
            eq = 0
        
        # The weak form of the momentum equation, the divergence free criterion
        # and the laplace equation for the potential
        # ∇⋅u = 0
        eq += div(u)*q*dx_ns
        # ∂u/∂t
        eq += rho*dot(u - up, v)/dt*dx_ns
        # ∇⋅(ρ u ⊗ u_conv)
        #eq += div(rho*u[d]*u_conv)*v[d]*dx
        eq += rho*dot(dot(grad(u), u_conv), v)*dx_ns
        # -∇⋅μ(∇u)
        eq += mu*df.inner(grad(u), grad(v))*dx_ns
        # ∇p
        eq -= div(v)*p*dx_ns
        # ρ g
        eq -= rho*dot(g, v)*dx_ns
        # -∇⋅∇ϕ = 0
        eq += dot(grad(phi), grad(r))*dx_pf
        
        # Velocity boundary integrals, from integration by parts
        for region in self.velocity_ds_boundaries:
            # Convection
            eq += rho*dot(u_conv, n)*dot(u, v)*ds(region)
            # Diffusion
            eq -= mu*dot(dot(grad(u), n), v)*ds(region)
        
        # Pressure boundary integral, from integration by parts
        for region in self.pressure_ds_boundaries:
            eq += p*dot(n, v)*ds(region)
        
        # Coupling
        for region in self.coupled_boundaries:
            # Convection
            eq += rho*dot(u_conv, n)*dot(grad(phi), v)*ds(region)
            # Diffusion
            eq -= mu*dot(dot(grad(grad(phi)), n), v)*ds(region)
            # Pressure
            p_c = - rho/dt(phi - self.phi) - rho/2*dot(u_conv, u_conv)
            eq += p_c*dot(n, v)*ds(region)
            # Potential
            eq -= dot(u, n)*r*ds(region)
        
        # Pressure boundary integral from outlet boundary condition
        # μ ∂u_n/∂n - p = F_n = 0
        for region in self.pressure_outlet_boundaries:
            un = dot(u, n)
            eq += mu*dot(n, grad(un))*dot(n, v)*ds(region)
        
        # Potential wall BCs, from integration by parts
        for region in self.potential_ds_boundaries:
            eq -= zero*r*ds(region)
        
        # Potential inlet BC, from integration by parts
        for region in self.potential_inflow_boundaries:
            eq -= self.U0*r*ds(region)
        
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
            self.tau_solver = df.LocalSolver(utau*vtau*dx_ns, tau*vtau*dx_ns)
            self.tau_solver.factorize()
            self.tau = df.Function(Vtau)
            
            # Multiply with the residual to ensure consistency
            v_supg = dot(grad(v), u_conv)*self.tau
            eq += dot(v_supg, rs)*dx_ns
        
        # Store the weak form for assembly
        self._weak_form = df.system(eq)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=Input.N1,
                        help='number of elements over the height')
    parser.add_argument('--tmax', type=float, default=Input.tmax,
                        help='simulation time')
    parser.add_argument('--dt', type=float, default=Input.dt,
                        help='simulation time step')
    parser.add_argument('-s', '--output-step', type=int, default=Input.output_step,
                        help='timesteps between each generated plot')
    parser.add_argument('--no-supg', action='store_true')
    parser.add_argument('-u', '--uncoupled', action='store_true')
    args = parser.parse_args()
    
    inp = Input()
    inp.N1 = inp.N2 = args.N
    inp.tmax = args.tmax
    inp.dt = args.dt
    inp.coupled_domains = not args.uncoupled
    inp.use_supg = not args.no_supg
    inp.output_step = args.output_step 
    main(inp)
