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
from cylinder_ns import NavierStokesDomain
from cylinder_hpc import PotentialFlowDomain
import numpy
import scipy.sparse.linalg


class Input(object):
    l1 = 2    # Length before NS domain starts
    l2 = 4    # Length of NS domain
    h1 = 1    # Height of pot domain
    h2 = 1    # Height of NS domain 
    d = 0.3   # Cylinder diameter
    N1 = N2 = 10
    layout = 'I'
    coupled = True
    
    U0 = 0.01    # Speed at inlet
    rho = 1   # Density
    Re = 100  # Reynolds number (determines the viscosity)
    
    dt = 0.01    # Timestep
    tmax = 0.15  # Time duration of the simulation
    tramp = 0.1
    
    def inlet_vel(self, t):
        fac = min(1, t/self.tramp)
        return self.U0*fac


def main(inp, plot=True, uncoupled=False):
    ns_domain = NavierStokesDomain(inp)
    pf_domain = PotentialFlowDomain(inp)
    ns_u_map, pf_p_map = get_domain_coupling(ns_domain, pf_domain)
    C1, C2  = None, None
    
    import warnings
    warnings.simplefilter('error', scipy.sparse.SparseEfficiencyWarning)
    
    t = 0
    it = 0
    dt = inp.dt
    while t <= inp.tmax + 1e-6 - dt:
        t += dt
        it += 1
        print 'Timestep %8.4f' % t
        
        # Assemble the two system matrices
        A1, b1 = ns_domain.get_system(t)
        A2, b2 = pf_domain.get_system(t)
        
        if C1 is None:
            # Setup coupling blocks between the two system matrices
            C1, C2 = off_diagonal_blocks(A1, A2, ns_u_map, pf_p_map, dt, inp.rho)
        
        if inp.coupled:
            # Apply Dirichlet boundary conditions to the Navier-Stokes block matrix
            for ns_dof, _, _ in ns_u_map:
                apply_dirichlet(A1, ns_dof)
                b1[ns_dof] = 0
            
            # Apply Dirichlet boundary conditions to the potential flow block matrix
            # and update the right hand side vector with the non-linear term from the 
            # previous time step (see the off_diagonal_blocks() function).
            phi_old = pf_domain.phi_old
            for pf_dof, _, _ in pf_p_map:
                apply_dirichlet(A2, pf_dof)
                vel = pf_domain.explicit_velocity_at_dof(pf_dof)            
                b2[pf_dof] = -(vel[0]**2 + vel[1]**2)/2*dt + phi_old[pf_dof]
        
        else:
            # Remove coupling
            C1 *= 0; C2 *= 0
            for ns_dof, _, _ in ns_u_map:
                # Set coupled velocity to inlet velocity 
                apply_dirichlet(A1, ns_dof)
                vel_dir = ns_domain.vel_dir[ns_dof] 
                b1[ns_dof] = inp.inlet_vel(t) if vel_dir == 0 else 0
            for pf_dof, _, _ in pf_p_map:
                # Set coupled potential to a constant
                apply_dirichlet(A2, pf_dof)
                b2[pf_dof] = 42
        
        # Assemble the block matrix
        AA = scipy.sparse.bmat([[A1, C1], [C2, A2]], 'csr')
        N1, N2 = len(b1), len(b2)
        bb = numpy.zeros(N1 + N2, float)
        bb[:N1] = b1
        bb[N1:] = b2
        
        # Solve the block matrix system 
        if inp.N1 < 7:
            print 'Cond: %8.2e' % numpy.linalg.cond(AA.todense())
        uu = scipy.sparse.linalg.spsolve(AA, bb)
        
        # Update the solutions in the two sub-domains
        ns_domain.update(uu[:N1])
        pf_domain.update(uu[N1:])
        
        if plot:
            fig = plot_domains(inp, ns_domain, pf_domain)
            fig.savefig('fig/timestep_%05d_t_%08d.png' % (it, t*1e4), dpi=100)
            
            #from matplotlib import pyplot
            #pyplot.figure()
            #pyplot.spy(AA.todense())
            #pyplot.show()
        #exit()


def get_domain_coupling(ns_domain, pf_domain):
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
                match_x = pf_coord0[0] <= ns_coord[0] <= pf_coord1[0]
                match_y = pf_coord0[1] <= ns_coord[1] <= pf_coord1[1]  
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


def off_diagonal_blocks(A1, A2, ns_u_map, pf_p_map, dt, rho):
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
            C2[pf_dof, d] = w*dt/rho
    
    return C1.tocsr(), C2.tocsr()


def apply_dirichlet(A, row):
    """
    Set row to be an identity row
        A[row,:] = 0
        A[row,row] = 1
    """
    j0, j1 = A.indptr[row], A.indptr[row+1]
    cols = A.indices[j0:j1]
    for col in cols:
        A[row,col] = 0
    A[row,row] = 1
    
    # Debug
    r = A.getrow(row)
    assert abs(r).sum() == 1 and A[row,row] == 1


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
    
    # Get color bar limits
    #maxabs_u = max(abs(values[0]).max(), abs(values[1]).max())
    maxabs_u = abs(inp.U0)*1.5
    maxabs_p = abs(values[2]).max()
    
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
    Cu  = axes[0].tripcolor(mesh, values[0], vmin=-maxabs_u, vmax=maxabs_u, **params)
    _   = axes[1].tripcolor(mesh, values[1], vmin=-maxabs_u, vmax=maxabs_u, **params)
    Cp  = axes[2].tripcolor(mesh, values[2], vmin=-maxabs_p, vmax=maxabs_p, **params)
    
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
    main(inp, plot=args.plot)
