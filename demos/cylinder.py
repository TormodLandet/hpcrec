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
import numpy
import scipy.sparse.linalg
import hpc
from cylinder_ns import NavierStokesDomain
from cylinder_hpc import PotentialFlowDomain


class Input(object):
    l1 = 2    # Length before NS domain starts
    l2 = 4    # Length of NS domain
    h1 = 1    # Height of pot domain
    h2 = 1    # Height of NS domain 
    d = 0.3   # Cylinder diameter
    f = 2/0.3 # Diameters between cylinder center and inlet
    N1 = N2 = 10
    layout = 'I'
    coupled = False
    
    U0 = 0.1  # Speed at inlet
    rho = 1   # Density
    Re = 1000 # Reynolds number (determines the viscosity)
    
    dt = 0.01   # Timestep
    tmax = 0.5  # Time duration of the simulation
    tramp = 0.3 # Time duration of the initial inlet velocity ramp-up
    output_step = 1
    
    def inlet_vel(self, t):
        fac = 1
        if t < self.tramp:
            fac = 0.5 - 0.5*numpy.cos(numpy.pi*t/self.tramp)
        return self.U0*fac


def main(inp, uncoupled=False):
    ns_domain = NavierStokesDomain(inp)
    pf_domain = PotentialFlowDomain(inp)
    ns_u_map, pf_p_map = get_domain_coupling(ns_domain, pf_domain)
    C1, C2  = None, None
    
    import warnings
    warnings.simplefilter('error', scipy.sparse.SparseEfficiencyWarning)
    
    t = 0
    it = 0
    dt = inp.dt
    rho = inp.rho
    while t <= inp.tmax + 1e-6 - dt:
        t += dt
        it += 1
        print 'Timestep %8.4f' % t
        
        # Assemble the two system matrices
        A1, b1 = ns_domain.get_system(t)
        A2, b2 = pf_domain.get_system(t)
        
        if C1 is None:
            # Setup coupling blocks between the two system matrices
            C1, C2 = off_diagonal_blocks(A1, A2, ns_u_map, pf_p_map)
        
        phi_prev = pf_domain.phi
        if inp.coupled:
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
            # Remove coupling
            C1 *= 0; C2 *= 0
            for ns_dof, _, _ in ns_u_map:
                # Set coupled velocity to inlet velocity 
                apply_dirichlet(A1, ns_dof)
                vel_dir = ns_domain.vel_dir[ns_dof] 
                b1[ns_dof] = inp.inlet_vel(t) if vel_dir == 0 else 0
            
            for pf_dof, _, _ in pf_p_map:
                # Set coupled potential neumann to inlet velocity
                nbs, _, cdx, _ = hpc.eval_phi(pf_domain.domain, pf_dof)
                A2[pf_dof, pf_dof] = 0
                for nb, c in zip(nbs, cdx):
                    A2[pf_dof, nb] = c
                b2[pf_dof] = inp.inlet_vel(t)
        
        # Assemble the block matrix
        AA = scipy.sparse.bmat([[A1, C1], [C2, A2]], 'csr')
        N1, N2 = len(b1), len(b2)
        bb = numpy.zeros(N1 + N2, float)
        bb[:N1] = b1
        bb[N1:] = b2
        
        # Solve the block matrix system 
        if it == 1 and inp.N1 < 7:
            print 'Cond: %8.2e' % numpy.linalg.cond(AA.todense())
        uu = scipy.sparse.linalg.spsolve(AA, bb)
        
        # Update the solutions in the two sub-domains
        ns_domain.update(uu[:N1])
        pf_domain.update(uu[N1:])
        
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
        
        if it % inp.output_step == 0:
            fig = plot_domains(inp, ns_domain, pf_domain)
            fig.savefig('fig/timestep_%05d_t_%08d.png' % (it, t*1e4), dpi=100)
            
            #from matplotlib import pyplot
            #pyplot.figure()
            #pyplot.spy(AA.todense())
            #pyplot.show()
        #exit()


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
            C1[ns_dof, d] = -w
    
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
    cols = A.indices[j0:j1]
    for col in cols:
        A[row,col] = 0
    A[row,row] = diag_value


def plot_domains(inp, ns_domain, pf_domain):
    """
    Plot the combined results in terms of velocity and pressure
    """
    from matplotlib import pyplot
    from matplotlib import tri
    
    # Get combined triangulation
    ns_coords, ns_triangles = ns_domain.get_triangulation()
    pf_coords, pf_triangles = pf_domain.get_triangulation()
    Nv_ns, Nv_pf = len(ns_coords), len(pf_coords)
    
    coords = numpy.zeros((Nv_ns + Nv_pf, 2), float)
    coords[:Nv_ns] = ns_coords
    coords[Nv_ns:] = pf_coords
    
    triangles = list(ns_triangles)
    for v0, v1, v2 in pf_triangles:
        triangles.append((v0+Nv_ns, v1+Nv_ns, v2+Nv_ns))
    mesh = tri.Triangulation([c[0] for c in coords], [c[1] for c in coords], triangles)
    
    # Get combined data
    func_names = ['u0', 'u1', 'p']
    values = numpy.zeros((3, len(coords)), float)
    for i, func_name in enumerate(func_names):
        values[i,:Nv_ns] = ns_domain.get_data(func_name)
        values[i,Nv_ns:] = pf_domain.get_data(func_name) 
    
    # Scaling
    scale_u = inp.U0
    scale_p = inp.U0**2*inp.rho
    
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
        axes = [None]*5
        axes[0] = fig.add_axes([0.02, 0.66, 0.80, 0.33])
        axes[1] = fig.add_axes([0.02, 0.33, 0.80, 0.33])
        axes[2] = fig.add_axes([0.02, 0.00, 0.80, 0.33])
        # Colorbar axes
        axes[3] = fig.add_axes([0.85, 0.55, 0.05, 0.35])
        axes[4] = fig.add_axes([0.85, 0.10, 0.05, 0.35])
        
    # Setup color map to be blue via white to red with out of range colors cyan and pink
    cmap = pyplot.cm.get_cmap('RdBu_r')
    cmap.set_over('#ff7ee6')
    cmap.set_under('#25f4ff')
    cmap.set_bad('#acacac')
    params = dict(shading='gouraud', cmap=cmap)
    
    # Plot functions on triangulation
    Cu  = axes[0].tripcolor(mesh, values[0]/scale_u, vmin=-maxabs_u, vmax=maxabs_u, **params)
    _   = axes[1].tripcolor(mesh, values[1]/scale_u, vmin=-maxabs_u, vmax=maxabs_u, **params)
    Cp  = axes[2].tripcolor(mesh, values[2]/scale_p, vmin=-maxabs_p, vmax=maxabs_p, **params)
    
    # Plot triangulation mesh lightly above the functions
    for ax in axes[:3]:
        ax.triplot(mesh, c='#cccccc', lw=0.2)
    
    # Colorbars
    fig.colorbar(Cu, cax=axes[3])
    fig.colorbar(Cp, cax=axes[4])
    
    for ax in axes[:3]:
        ax.axis('off')
        ax.plot([0, 0], [-inp.h2/2, inp.h2/2], ':k')
    
    inp._plot_domains_save = fig, axes
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=10,
                        help='number of elements over the height')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-u', '--uncoupled', action='store_true')
    args = parser.parse_args()
    
    inp = Input()
    inp.N1 = inp.N2 = args.N
    inp.coupled = not args.uncoupled
    if not args.plot:
        inp.output_step = 1e100 
    main(inp)
