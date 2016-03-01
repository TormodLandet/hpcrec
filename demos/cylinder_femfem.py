# encoding: utf-8
from __future__ import division
import time
import numpy
from matplotlib import pyplot
import dolfin as df
from dolfin import div, grad, dot
from cylinder import Input
from cylinder_ns import NavierStokesDomain
from cylinder_hpc import PotentialFlowDomain
from utilities import SimpleLog, StreamFunction, SolutionProperties, define_penalty

COUPLED_NO = 'uncoupled'
COUPLED_DIRICHLET = 'dirichlet'
COUPLED_NEUMANN = 'neumann'

def main(inp):
    log = SimpleLog('cylinder_femfem.log')
    log.info('Running cylinder_femfem with input:\n')
    log.dump_object(inp, '    ')
    
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
    uC, _uP = domain.calculate_combined_functions()
    stream_function = StreamFunction(uC)
    solprops = SolutionProperties(domain.u, dt=inp.dt, nu=inp.mu/inp.rho, dx=domain.dx_ns)
    
    # Time loop
    t = 0
    it = 0
    dt = inp.dt
    timer_loop_start = time.time()
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
                fig = plot_domain(inp, domain, t, stream_function)
                fig.savefig('fig/timestep_%05d_t_%08d.png' % (it, round(t*1e4)), dpi=100)
                log.flush()
        
        log.info('  Timestep: %4.2fs' % (time.time() - timer_ts_start))
            
        Fv, Fp = domain.get_force()
        log.info('  Fp: % .3e % .3e  Fv: % .3e % .3e' % (Fp[0], Fp[1], Fv[0], Fv[1]))
        
        # Calculate the Courant and Peclet numbers
        Co_max = solprops.courant_number().vector().max()
        Pe_max = solprops.peclet_number().vector().max()
        log.info('  Co: %6.1e  Pe: %6.1e\n' % (Co_max, Pe_max))
        
        
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
        self.coupling_method = inp.coupling_method
        
        self._create_mesh(coords, triangles, Nc_ns)
        self._create_functions()
        self._create_boundary_conditions()
        self._create_weak_form()
        self._create_combined_functions()
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
        self.disturbance.assign(df.Constant(self.input.disturbance(t)))
        
        a, L = self._weak_form
        A, b = df.assemble_system(a, L, self.dirichlet_bcs)
        A.ident_zeros()
        return A, b

    def solve(self, A, b):
        w = self.func
        self.solver.solve(A, w.vector(), b)
        
        # Spread to the component vectors
        self.phi_p.assign(self.phi)
        self.assigner.assign(self.functions, w)
        for func in self.functions:
            func.vector().apply('insert') # dolfin bug #587
    
    def get_force(self, region=5):
        """
        Integrate the pressure and viscous forces on the given region
        """
        ds = self.ds(region)
        n = df.FacetNormal(self.mesh)
        
        # Viscosity, μ(∇u)
        mu = df.Constant(inp.mu)
        sigma_n = mu*dot(grad(self.u), -n)
        Fvx = df.assemble(sigma_n[0]*ds)
        Fvy = df.assemble(sigma_n[1]*ds)
        
        # Pressure, p
        Fpx = df.assemble(self.p*n[0]*ds)
        Fpy = df.assemble(self.p*n[1]*ds)
        
        return [Fvx, Fvy], [Fpx, Fpy]
    
    def calculate_combined_functions(self):
        """
        Project the two domain solutions into one single solution for the whole domain
        """
        # Combined velocity
        uC, A, L0, L1 = self._uC
        b0 = df.assemble(L0)
        b1 = df.assemble(L1)
        df.solve(A, uC[0].vector(), b0)
        df.solve(A, uC[1].vector(), b1)
        
        # Combined pressure
        uP, A, L = self._pC
        b = df.assemble(L)
        df.solve(A, uP.vector(), b)

        return uC, uP
    
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
        
        self.dx = df.Measure('dx')(subdomain_data=cell_marker)
        self.dx_ns = self.dx(1)
        self.dx_pf = self.dx(2)
        self.ds = df.Measure('ds')(subdomain_data=facet_marker)
        self.dS = df.Measure('dS')(subdomain_data=facet_marker)
                
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
        e_r = df.FiniteElement('CG', cell, self.input.Pu+1)
        V = df.FunctionSpace(self.mesh, e_u)
        Q = df.FunctionSpace(self.mesh, e_p)
        R = df.FunctionSpace(self.mesh, e_r)
        
        # Functions
        self.u0 = df.Function(V)
        self.u1 = df.Function(V)
        self.u = df.as_vector([self.u0, self.u1])
        self.u_conv0 = df.Function(V)
        self.u_conv1 = df.Function(V)
        self.p = df.Function(Q)
        self.phi = df.Function(R)
        self.phi_p = df.Function(R)
        
        if self.use_lagrange_multiplicator:
            e_l = df.FiniteElement('R', cell, 0)
            L = df.FunctionSpace(self.mesh, e_l)
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
        self.assigner = df.FunctionAssigner(func_spaces, W)
    
    def _create_boundary_conditions(self):        
        # "Constants" that are changed every time step before assembly
        self.U0 = df.Constant(-1)
        self.t = df.Constant(-1)
        
        # Dirichlet boundary conditions
        W = self.funcspace
        zero = df.Constant(0)
        self.disturbance = df.Constant(0)
        
        marker = self.facet_marker
        self.dirichlet_bcs = [# Outlet BCs
                              #df.DirichletBC(W.sub(0), uout, marker, 2),
                              #df.DirichletBC(W.sub(1), zero, marker, 2),
                              # Bottom BCs
                              #df.DirichletBC(W.sub(0), self.U0, marker, 1),
                              df.DirichletBC(W.sub(1), self.disturbance, marker, 1),
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
        
        if self.input.coupling_method == COUPLED_NO:
            # NS inlet BCs
            self.dirichlet_bcs.append(df.DirichletBC(W.sub(0), self.U0, marker, 4))
            self.dirichlet_bcs.append(df.DirichletBC(W.sub(1), zero, marker, 4))
            # PF outlet BC
            self.dirichlet_bcs.append(df.DirichletBC(W.sub(3), df.Constant(1), marker, 4))
    
    
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
        
        up = self.u
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
        #eq += rho*dot(dot(grad(u), u_conv), v)*dx_ns
        eq -= rho*dot(u, div(df.outer(v, u_conv)))*dx_ns
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
            eq -= -self.U0*r*ds(region)
        
        # Domain coupling 
        NS, PF = '-', '+' # PF is plus since marker 2 > marker 1
        if self.coupling_method == COUPLED_NEUMANN:
            # Replace all natural boundary conditions (Neumann in this case)
            # with the value from the opposite domain
            for region in self.coupled_boundaries:
                # Convection
                coupling = rho(NS)*dot(u_conv(NS), n(NS))*dot(grad(phi(PF)), v(NS))
                # Diffusion
                coupling -= mu(NS)*dot(dot(grad(grad(phi(PF))), n(NS)), v(NS))
                # Pressure
                p_pf = - rho(NS)/dt(NS)*(phi(PF) - self.phi(PF)) - rho(NS)/2*dot(u_conv(NS), u_conv(NS))
                coupling += p_pf*dot(n(NS), v(NS))
                # Potential
                coupling -= dot(u(NS), n(PF))*r(PF)
                # Add the coupling terms to the equation system
                eq += coupling*self.dS(region)
        
        elif self.coupling_method == COUPLED_DIRICHLET:
            # Set the values at the coupled boundaries to the value in the opposite
            # domain by weak Dirichlet boundary conditions (Nitsche's method)
            
            # Penalties
            Pu = self.u0.function_space().ufl_element().degree()
            Pphi = self.phi.function_space().ufl_element().degree()
            penalty_ns  = define_penalty(self.mesh, Pu, inp.mu, inp.mu)
            penalty_pf  = define_penalty(self.mesh, Pphi, 1, 1)
            penalty_ns = df.Constant(2*penalty_ns) # Penalty on ds = 2*penalty on dS
            penalty_pf = df.Constant(2*penalty_pf) # Penalty on ds = 2*penalty on dS
            
            uconv_uw = (dot(u_conv, n) + abs(dot(u_conv, n)))/2.0
            uconv_dw = (dot(u_conv, n) - abs(dot(u_conv, n)))/2.0
            for region in self.coupled_boundaries:
                coupling = 0
                
                # Convection
                pot_u = grad(phi(PF))
                coupling += rho(NS)*dot(u(NS), v(NS))*uconv_uw(NS)
                coupling += rho(NS)*dot(pot_u, v(NS))*uconv_dw(NS)
                
                # Diffusion
                zero_u = u(NS) - pot_u
                coupling -= mu(NS)*dot(dot(grad(u(NS)), n(NS)), v(NS))
                coupling -= mu(NS)*dot(dot(grad(v(NS)), n(NS)), zero_u)
                coupling += penalty_ns*dot(zero_u, v(NS))
                
                # Weak Dirichlet for the potential / Bernoulli's equation with constant = 0
                zero_phi = (phi(PF) - self.phi(PF))/dt(NS) + \
                           0.5*dot(grad(phi(PF)), grad(self.phi(PF))) + \
                           p(NS)/rho(NS)
                coupling -= dot(grad(phi(PF)), n(PF))*r(PF)
                coupling -= dot(grad(r(PF)), n(PF))*zero_phi
                coupling += penalty_pf*zero_phi*r(PF)
                
                # Pressure IBP term
                coupling += p(NS)*dot(n(NS), v(NS))
                
                # Add the coupling terms to the equation system
                eq += coupling*self.dS(region)
        
        else:
            assert self.coupling_method == COUPLED_NO
            # The boundary terms from integration by parts
            #for region in self.coupled_boundaries:
            #    # Convection
            #    coupling = rho(NS)*dot(u_conv(NS), n(NS))*dot(u(NS), v(NS))
            #    # Diffusion
            #    coupling -= mu(NS)*dot(dot(grad(u(NS)), n(NS)), v(NS))
            #    # Pressure
            #    coupling += p(NS)*dot(n(NS), v(NS))
            #    # Potential
            #    coupling -= dot(grad(phi(PF)), n(PF))*r(PF)
        
        # The residual of the N-S momentum equation
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
        
    def _create_combined_functions(self):
        """
        Define projections into global velocity and pressure functions
        """
        inc_NS = df.Constant(1)
        inc_PF = df.Constant(1)
        
        # Combined velocity, uC
        V = self.u0.function_space()
        u, v = df.TrialFunction(V), df.TestFunction(V)
        a = u*v*inc_NS*self.dx_ns + u*v*inc_PF*self.dx_pf
        L0 = inc_NS*self.u0*v*self.dx_ns + inc_PF*self.phi.dx(0)*v*self.dx_pf
        L1 = inc_NS*self.u1*v*self.dx_ns + inc_PF*self.phi.dx(1)*v*self.dx_pf
        A = df.assemble(a)
        A.ident_zeros()
        uC = df.as_vector([df.Function(V), df.Function(V)])
        self._uC = (uC, A, L0, L1)
        
        # Combined pressure, pC
        rho, dt = self.input.rho, self.input.dt
        V = self.p.function_space()
        u, v = df.TrialFunction(V), df.TestFunction(V)
        a = u*v*inc_NS*self.dx_ns + u*v*inc_PF*self.dx_pf
        L = inc_NS*self.p*v*self.dx_ns
        vel = grad(self.phi)
        p_pf = - rho/dt*(self.phi - self.phi_p) - rho/2*dot(vel, vel)
        L += inc_PF*p_pf*v*self.dx_pf
        A = df.assemble(a)
        A.ident_zeros()
        pC = df.Function(V)
        self._pC = (pC, A, L)


def plot_domain(inp, domain, simulation_time, stream_function=None, quiver=False):
    """
    Plot the combined velocity and pressure fields at the current time step
    """
    # Get figure and axes
    if hasattr(inp, '_plot_domains_save'):
        fig, axes = inp._plot_domains_save
        for ax in axes:
            ax.clear()
    else:
        fig = pyplot.figure(figsize=(12, 9))
        axes = [None]*6
        axes[0] = fig.add_axes([0.04, 0.75, 0.80, 0.25])
        axes[1] = fig.add_axes([0.04, 0.50, 0.80, 0.25])
        axes[2] = fig.add_axes([0.04, 0.25, 0.80, 0.25])
        axes[3] = fig.add_axes([0.04, 0.00, 0.80, 0.25])
        # Colorbar axes
        axes[4] = fig.add_axes([0.88, 0.55, 0.05, 0.35])
        axes[5] = fig.add_axes([0.88, 0.10, 0.05, 0.35])
        
    # Setup color map to be blue via white to red with out of range colors cyan and pink
    cmap = pyplot.cm.get_cmap('RdBu_r')
    cmap.set_over('#ff7ee6')
    cmap.set_under('#25f4ff')
    cmap.set_bad('#acacac')
    
    # Plot functions
    uC, pC = domain.calculate_combined_functions()
    props = dict(backend='matplotlib', shading='gouraud')
    velprops = dict(backend='matplotlib', shading='gouraud',
                    vmin=-2.5*inp.U0, vmax=2.5*inp.U0)
    pyplot.sca(axes[0]); Cu0 = df.plot(uC[0], **velprops)
    pyplot.sca(axes[1]); Cu1 = df.plot(uC[1], **velprops)
    pyplot.sca(axes[3]); Cp = df.plot(pC, **props)
    
    # Stream function
    if stream_function:
        stream_function.compute()
        stream_function.plot(axes[2])
    
    # Quiver plot
    if quiver:
        pyplot.sca(axes[2])
        df.plot(uC, backend='matplotlib')
    
    # Plot triangulation mesh lightly above the functions
    mesh = uC[0].function_space().mesh()
    for ax in axes[2:3]:
        pyplot.sca(ax)
        df.plot(mesh, backend='matplotlib', c='#999999', lw=0.2)
    
    # Colorbars
    fig.colorbar(Cu0, cax=axes[4])
    fig.colorbar(Cp, cax=axes[5])
    
    for ax in axes[:4]:
        ax.axis('off')
    
    # Some informative text
    ax = axes[5]
    tp = dict(transform=fig.transFigure, family='monospace')
    text_propsR = dict(horizontalalignment='right', verticalalignment='bottom', **tp)
    text_propsC = dict(horizontalalignment='center', verticalalignment='center', **tp)
    ax.text(0.99, 0.04, 't=%5.2f' % simulation_time, **text_propsR)
    ax.text(0.99, 0.01, 'Re=%4g' % inp.Re, **text_propsR)
    ax.text(0.02, 0.875, 'u', **text_propsC)
    ax.text(0.02, 0.625, 'v', **text_propsC)
    ax.text(0.02, 0.125, 'p', **text_propsC)
    ax.text(0.905, 0.92, 'u, v', **text_propsC)
    ax.text(0.905, 0.47, 'p', **text_propsC)
    
    inp._plot_domains_save = fig, axes
    return fig


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
    parser.add_argument('-m', '--coupling-method', default=COUPLED_DIRICHLET,
                        choices=[COUPLED_NO, COUPLED_NEUMANN, COUPLED_DIRICHLET])
    args = parser.parse_args()
    
    inp = Input()
    inp.N1 = inp.N2 = args.N
    inp.tmax = args.tmax
    inp.dt = args.dt
    inp.coupling_method = args.coupling_method
    inp.use_supg = not args.no_supg
    inp.output_step = args.output_step 
    main(inp)
