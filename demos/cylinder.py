# encoding: utf8
"""
Flow around a cylinder by domain decomposition where a simple P2P1
coupled Navier-Stokes in FEniCS is coupled to a outer HPC domain.

The computational domain (with the "C" layout):

  +-----------------------------------------------
  |          Potential flow domain
  | -->         +---------------------------------
  | -->         |         Navier-Stokes domain
  | -->         |  /-\
  | -->         |  \-/
  | -->         |
  | -->         +---------------------------------
  | -->
  +------------------------------------------------

As a start we begin with the simple "I" layout

  +-----------------+-----------------------------
  |    Pot.flow     |    Navier-Stokes
  | -->             |
  | -->             |         
  | -->             |  /-\
  | -->             |  \-/ <-- cylinder
  | -->             |
  | -->             |
  | -->             |
  +-----------------+------------------------------

and denote the line separating the pot. and NS domains as dividing line #0.

We find matching dofs such that we can prescribe Dirichlet conditions on
both velocity components on the Navier-Stokes side (by differentiating the
velocity potential to find the velocity). On the potential flow side we
use the Navier-Stokes pressure and the Bernoulli equation to set Dirichlet
boundary conditions on the potential, phi.
"""
from __future__ import division
import time
import numpy
import scipy.sparse.linalg
import hpc
from cylinder_ns import NavierStokesDomain
from cylinder_hpc import PotentialFlowDomain
from utilities import SimpleLog, StreamFunction


class Input(object):
    l1 = 1        # Length before NS domain starts
    l2 = 4        # Length of NS domain
    h1 = 1        # Height of pot domain
    h2 = 1        # Height of NS domain 
    d = 0.1       # Cylinder diameter
    f = 1.5       # Diameters between cylinder center and inlet 
    N1 = N2 = 10  # Geometric discretisation
    
    # Which problem to solve and what type of mesh layout to make
    problem = 'Cylinder'
    layout = 'I'
    coupling_method = 'dirichlet' # or neumann or uncoupled
    
    U0 = 0.1    # Speed at inlet
    rho = 1     # Density
    Re = 100    # Reynolds number (determines the viscosity)
    
    dt = 0.01   # Timestep
    tmax = 150  # Time duration of the simulation
    tramp = 0.3 # Time duration of the initial inlet velocity ramp-up
    output_step = 1e100
    disturbance_time = (10, 12, 14)
    
    # Finite element discretization
    Pu = 2
    Pp = 1
    pressure_lagrange_multiplier = False
    use_supg = True
    
    def inlet_vel(self, t):
        fac = 1
        if t < self.tramp:
            fac = 0.5 - 0.5*numpy.cos(numpy.pi*t/self.tramp)
        return self.U0*fac
    
    @property
    def mu(self):
        return self.d*self.U0*self.rho/self.Re

    def disturbance(self, t):
        "Disturbance to trigger alternating vortex shedding"
        t1, t2, t3 = self.disturbance_time
        if t1 < t < t2:
            tfac = (t - t1)/(t2 - t1)
        elif 12 <= t < 14:
            tfac = 1 - (t - t2)/(t3 - t2)
        else:
            tfac = 0
        return tfac * 0.1 * self.U0 


def main(inp):
    # Detect matrix insertions breaking the non-zero pattern
    import warnings
    warnings.simplefilter('error', scipy.sparse.SparseEfficiencyWarning)
    
    log = SimpleLog('cylinder.log')
    ns_domain = NavierStokesDomain(inp)
    pf_domain = PotentialFlowDomain(inp)
    ns_u_map, pf_p_map = get_domain_coupling(ns_domain, pf_domain)
    log.info('Péclet: %.2f\n' % (inp.U0*ns_domain.h/(2*inp.mu/inp.rho)))
    
    # Time loop
    t = 0
    it = 0
    dt = inp.dt
    rho = inp.rho
    timer_loop_start = time.time()
    stream_function = StreamFunction(ns_domain.u)
    while t <= inp.tmax + 1e-6 - dt:
        t += dt
        it += 1
        timer_ts_start = time.time()
        log.info('Timestep %5d  t: %8.4f' % (it, t))
        
        # Assemble the two system matrices
        with log.timer('  Assemble: '):
            A1, b1 = ns_domain.get_system(t)
            A2, b2 = pf_domain.get_system(t)
        
        # Setup coupling matrices
        with log.timer('  Couple: '):
            if it == 1:
                # Setup coupling blocks between the two system matrices
                C1, C2 = off_diagonal_blocks(A1, A2, ns_u_map, pf_p_map)
            
            phi_prev = pf_domain.phi
            if inp.coupling_method == 'dirichlet':
                # Apply Dirichlet boundary conditions to the Navier-Stokes block matrix
                for ns_dof, _, _ in ns_u_map:
                    apply_dirichlet(A1, ns_dof)
                    b1[ns_dof] = 0
                
                # Apply Dirichlet boundary conditions to the potential flow block matrix
                # and update the right hand side vector with the non-linear term from the 
                # previous time step (see the off_diagonal_blocks() function).
                for pf_dof, _, _ in pf_p_map:
                    # Previous/explicit and next/implicit velocity
                    vel_prev = pf_domain.explicit_velocity_at_dof(pf_dof)
                    nbs, _, cdx, cdy = hpc.eval_phi(pf_domain.domain, pf_dof)
                    
                    apply_dirichlet(A2, pf_dof, rho/dt)
                    for nb, wu, wv in zip(nbs, cdx, cdy):
                        A2[pf_dof,nb] = rho/2*(wu*vel_prev[0] + wv*vel_prev[1])
                    
                    b2[pf_dof] = rho/dt*phi_prev[pf_dof]
            
            else:
                assert inp.coupling_method == 'uncoupled'
                # Remove coupling
                if it == 1:
                    C1 *= 0
                    C2 *= 0
        
        # Assemble the block matrix
        AA = scipy.sparse.bmat([[A1, C1], [C2, A2]], 'csr')
        N1, N2 = len(b1), len(b2)
        bb = numpy.zeros(N1 + N2, float)
        bb[:N1] = b1
        bb[N1:] = b2
        
        # Solve the block matrix system
        with log.timer('  Solve: '): 
            uu = scipy.sparse.linalg.spsolve(AA, bb)
        
        # Update the solutions in the two sub-domains
        ns_domain.update(uu[:N1])
        pf_domain.update(uu[N1:])
        
        with log.timer('  Plot: '):
            if it % inp.output_step == 0:
                fig = plot_domains(inp, [ns_domain, pf_domain], t, stream_function)
                fig.savefig('fig/timestep_%05d_t_%08d.png' % (it, round(t*1e4)), dpi=100)
                
                #from matplotlib import pyplot
                #pyplot.figure()
                #pyplot.spy(AA.todense())
                #pyplot.show()
        
        log.info('  Timestep: %4.2fs' % (time.time() - timer_ts_start))
        if it == 1 and inp.N1 < 7:
            log.info('Cond: %8.2e\n' % numpy.linalg.cond(AA.todense()))
            
        Fv, Fp = ns_domain.get_force()
        log.info('  Fp: % .3e % .3e  Fv: % .3e % .3e\n' % (Fp[0], Fp[1], Fv[0], Fv[1]))
        
        # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        if False:
            for pf_dof, ns_p_dofs, ns_p_weights in pf_p_map:
                # At coupling location
                nbs, c, cdx, cdy = hpc.eval_phi(pf_domain.domain, pf_dof)
                phi = pf_domain.phi[pf_dof]
                phi_p = pf_domain.phi_old[pf_dof]
                phi2 = sum(w*pf_domain.phi[nb] for nb, w in zip(nbs, c))
                dphidx = sum(w*pf_domain.phi_old[nb] for nb, w in zip(nbs, cdx)) 
                dphidy = sum(w*pf_domain.phi_old[nb] for nb, w in zip(nbs, cdy))
                vel = [dphidx, dphidy]
                vel_p = pf_domain.explicit_velocity_at_dof(pf_dof)
                
                p1 = -rho/2*(vel[0]*vel_p[0] + vel[1]*vel_p[1])
                p2 = -rho/dt*(phi - phi_p) 
                p = p1 + p2
                
                p_ns = 0
                for d, w in zip(ns_p_dofs, ns_p_weights):
                    p_ns += uu[d]*w
                
                print '%2d - % 8.2e % 8.2e - % 8.2e % 8.2e - % 8.2e % 8.2e % 8.2e' % (
                        pf_dof, phi, (phi-phi2)/phi, dphidx, dphidy, p1, p2, p),
                print ' - % 8.2e % 8.2e' % (p_ns, (p_ns-p)/p_ns)
        # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
    
    log.info('DONE in %.2fs\n' % (time.time() - timer_loop_start))


def get_domain_coupling(ns_domain, pf_domain, geps=1e-8):
    """
    As a preprocessor we run through all lines separating the potential flow
    and the Navier-Stokes domains and get the mapping of dofs between the
    two so that we can apply Dirichlet BCs both ways in the time loop
    """
    ns_u_map = []
    pf_p_map = []
    
    for iline in range(ns_domain.num_dividing_lines):
        ns_dof_coords = ns_domain.get_dividing_line(iline)
        pf_dof_coords = pf_domain.get_dividing_line(iline)
        
        # Find the gradients in the potential flow domain to use as the 
        # Dirichlet boundary condition in the N-S domain
        ns_p_dof_coords = []
        pf_Ndl = len(pf_dof_coords)
        for ns_dof, ns_coord, ns_dir in ns_dof_coords:
            # Collect and skip pressure dofs
            if ns_dir == -1:
                ns_p_dof_coords.append((ns_dof, ns_coord))
                continue
            
            # Linear search of for matching potential flow
            for i in range(pf_Ndl):
                pf_dof0, pf_coord0 = pf_dof_coords[i]
                pf_dof1, pf_coord1 = pf_dof_coords[i+1]
                
                # Search until we find a match on the potential flow side
                match_x = pf_coord0[0] - geps < ns_coord[0] < pf_coord1[0]  + geps
                match_y = pf_coord0[1] - geps < ns_coord[1] < pf_coord1[1] + geps
                if match_x and match_y:
                    # Get the weights
                    dofs, weights = pf_domain.get_neumann_weights(ns_dir, ns_coord,
                                                                  pf_dof0, pf_dof1)
                    ns_u_map.append((ns_dof, dofs, weights))
                    break
        
        # Find the pressure dofs in the N-S domain to use in the Dirichlet boundary
        # conditions for the potential in the potential flow domain
        ns_Ndl = len(ns_p_dof_coords)
        for pf_dof, pf_coord in pf_dof_coords:
            # Linear search to find matching N-S pressure
            for i in range(ns_Ndl):
                ns_dof0, ns_coord0 = ns_p_dof_coords[i]
                ns_dof1, ns_coord1 = ns_p_dof_coords[i+1]
                
                # Search until we find a match on the potential flow side 
                match_x = ns_coord0[0] <= pf_coord[0] <= ns_coord1[0]
                match_y = ns_coord0[1] <= pf_coord[1] <= ns_coord1[1]  
                if match_x and match_y:
                    # Get the weights
                    dofs, weights = ns_domain.get_pressure_weights(pf_coord, ns_dof0, ns_dof1)
                    pf_p_map.append((pf_dof, dofs, weights))
                    break
    
    return ns_u_map, pf_p_map


def off_diagonal_blocks(A1, A2, ns_u_map, pf_p_map):
    """
    Return block matrices C1 and C2 which will be inserted as
    
        A1  C1
        C2  A2
    
    in the global matrix where A1 is the Navier-Stokes LHS and
    A2 is the potential flow LHS. C1 contains the derivatives
    of the potential used as Dirichlet BC for N-S velocity and
    C2 contains the N-S pressure to be used as Dirichlet BC for
    the potential (through the Bernoulli equation). 
    """
    C1 = scipy.sparse.lil_matrix((A1.shape[0], A2.shape[1]))
    C2 = scipy.sparse.lil_matrix((A2.shape[0], A1.shape[1]))
    
    # Dirichlet boundary conditions for the Navier-Stokes velocity
    for ns_dof, pf_dofs, pf_weights in ns_u_map:
        # u - ∇ϕ = 0
        for d, w in zip(pf_dofs, pf_weights):
            C1[ns_dof, d] += -w
    
    # Dirichlet boundary conditions for the potential using Bernulli's equation
    #    ∂ϕ/∂t + p/ρ + 1/2(∇ϕ)² + gy = C(t)
    # which gives, when pulling C(t) into ϕ and disregarding gravity: 
    #               ϕ^{n+1} + p/ρ Δt = - 1/2(∇ϕ^n)² + ϕ^n 
    # where we have used first order backward time differencing.
    for pf_dof, ns_p_dofs, ns_p_weights in pf_p_map:
        for d, w in zip(ns_p_dofs, ns_p_weights):
            C2[pf_dof, d] = w
    
    return C1.tocsr(), C2.tocsr()


def apply_dirichlet(A, row, diag_value=1):
    """
    Set row to be an identity row
        A[row,:] = 0
        A[row,row] = 1
    """
    j0, j1 = A.indptr[row], A.indptr[row+1]
    A.data[j0:j1] = 0
    A[row,row] = diag_value


def plot_domains(inp, domains, simulation_time, stream_function=None, quiver=False):
    """
    Plot the combined results in terms of velocity and pressure
    """
    from matplotlib import pyplot
    from matplotlib import tri
    
    # Get combined triangulation
    comb_coords, comb_triangles = [], []
    for domain in domains:
        c, t = domain.get_triangulation()
        comb_coords.append(c)
        comb_triangles.append(t)
    Nc = sum(len(c) for c in comb_coords)
    
    # Get coordinates as one long numpy array,
    # triangles as one long list, and function
    # values as one long array from all domains
    coords = numpy.zeros((Nc, 2), float)
    triangles = []
    func_names = ['u0', 'u1', 'p']
    values = numpy.zeros((3, Nc), float)
    start = 0
    for domain, c, t in zip(domains, comb_coords, comb_triangles):
        n = len(c)
        coords[start:start+n] = c
        
        # Renumber triangle vertex numbers
        for v0, v1, v2 in t:
            triangles.append((v0+start, v1+start, v2+start))
        
        # Get function data
        for i, func_name in enumerate(func_names): 
            values[i,start:start+n] = domain.get_data(func_name)
        
        start += n
    
    # Combined triangulation
    X = numpy.array([c[0] for c in coords], float)
    Y = numpy.array([c[1] for c in coords], float)
    mesh = tri.Triangulation(X, Y, triangles)
    
    # Scaling
    scale_u = inp.U0
    scale_p = inp.U0**2*inp.rho
    U = values[0]/scale_u
    V = values[1]/scale_u
    
    # Get color bar limits
    maxabs_u = 2.5
    maxabs_p = abs(values[2]).max()/scale_p
    
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
    params = dict(shading='gouraud', cmap=cmap)
    
    # Plot functions on triangulation
    Cu  = axes[0].tripcolor(mesh, values[0]/scale_u, vmin=-maxabs_u, vmax=maxabs_u, **params)
    _   = axes[1].tripcolor(mesh, values[1]/scale_u, vmin=-maxabs_u, vmax=maxabs_u, **params)
    Cp  = axes[3].tripcolor(mesh, values[2]/scale_p, vmin=-maxabs_p, vmax=maxabs_p, **params)
    
    # Stream function
    if stream_function:
        stream_function.compute()
        stream_function.plot(axes[2])
    
    # Quiver plot
    if quiver:
        params_quiver = dict(scale=inp.N2/2,
                             width=1/(inp.N2*4),
                             scale_units='x',
                             units='x')
        rs = numpy.random.RandomState()
        rs.seed(42)
        I = rs.rand(X.size) < 0.33
        axes[2].quiver(X[I], Y[I], U[I], V[I], **params_quiver)
    
    # Plot triangulation mesh lightly above the functions
    for ax in axes[2:3]:
        ax.triplot(mesh, c='#999999', lw=0.2)
    
    # Colorbars
    fig.colorbar(Cu, cax=axes[4])
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
    parser.add_argument('-m', '--coupling-method', default='dirichlet',
                        choices=['dirichlet', 'neumann', 'uncoupled'])
    args = parser.parse_args()
    
    inp = Input()
    inp.N1 = inp.N2 = args.N
    inp.tmax = args.tmax
    inp.dt = args.dt
    inp.coupling_method = args.coupling_method
    inp.use_supg = not args.no_supg
    inp.output_step = args.output_step 
    main(inp)
