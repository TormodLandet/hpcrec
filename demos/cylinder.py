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
    N = 10
    layout = 'I'
    
    U0 = 1    # Speed at inlet
    rho = 1   # Density
    Re = 100  # Reynolds number (determines the viscosity)
    
    dt = 0.01
    tmax = 1.0


def main(inp):
    ns_domain = NavierStokesDomain(inp)
    pf_domain = PotentialFlowDomain(inp)
    ns_u_map, pf_p_map = get_domain_coupling(ns_domain, pf_domain)
    C1, C2  = None, None
    
    t = 0
    it = 0
    dt = inp.dt
    while t <= inp.tmax + 1e-6:
        t += inp.dt
        it += 1
        print 'Timestep %8.4f' % t
        
        # Assemble the two system matrices
        A1, b1 = ns_domain.assemble_csr()
        A2, b2 = pf_domain.assemble_csr()
        
        if C1 is None:
            # Setup coupling blocks between the two system matrices
            C1, C2 = off_diagonal_blocks(A1, A2, ns_u_map, pf_p_map, dt, inp.rho)
            
        # Apply Dirichlet boundary conditions to the Navier-Stokes block matrix
        for ns_dof, _, _ in ns_u_map:
            ns_domain.apply_dirichlet(A1, ns_dof)
        
        # Apply Dirichlet boundary conditions to the potential flow block matrix
        # and update the right hand side vector with the non-linear term from the 
        # previous time step (see the off_diagonal_blocks() function).
        for pf_dof, _, _ in pf_p_map:
            pf_domain.apply_dirichlet(A2, pf_dof)
            vel = pf_domain.explicit_velocity_at_dof(pf_dof)
            b2[pf_dof] = (vel[0]**2 + vel[1]**2)/2
        
        # Assemble the block matrix
        AA = scipy.sparse.bmat([[A1, C1], [C2, A2]], 'csr')
        N1, N2 = len(b1), len(b2)
        bb = numpy.zeros(N1 + N2, float)
        bb[:N1] = b1
        bb[N1:] = b2
        
        # Solve the block matrix system 
        uu = scipy.sparse.linalg.spsolve(AA, bb)
        
        # Update the solutions in the two sub-domains
        ns_domain.update(uu[:N1])
        pf_domain.update(uu[N1:])


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
    #               ϕ^{n+1} + p/ρ Δt = - 1/2(∇ϕ^n)²
    # where we have used first order backward time differencing.
    for pf_dof, ns_p_dofs, ns_p_weights in pf_p_map:
        for d, w in zip(ns_p_dofs, ns_p_weights):
            C2[pf_dof, d] = w*dt/rho
    
    return C1.tocsr(), C2.tocsr()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=10,
                        help='number of elements over the height')
    args = parser.parse_args()
    
    inp = Input()
    inp.N = args.N
    main(inp)
