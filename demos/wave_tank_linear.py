
from typing import Literal
from pathlib import Path
from math import sin, cos, pi
import pickle

import numpy as np

import hpcrec


class WaveTankInput:
    # Geometry
    N: int
    L: float
    h: float = 1.0
    refine_y: float = 1.0
    
    # Timestepping
    tmax: float = 1.0
    tramp: float = 0.1
    Nt: int = 100
    
    # Wave maker input
    wm_ampls: list[float] = [0.01]
    wm_freqs: list[float] = [5.0]
    wm_phases: list[float] = [0.0]
    
    # Physical constants
    g: float = 9.81


def wavetank_demo(wave_tank_input: WaveTankInput, show_plot: bool=True):
    N = wave_tank_input.N
    L = wave_tank_input.L
    h = wave_tank_input.h
    print(f"Calculating wavetank with N={N}, h={h}, L={L}")
    
    # Setup geometry
    domain = hpcrec.rectangle_domain((0, 0), (L, h), int(N*L/h), N)
    
    # Refine the mesh towards the free surface
    Nd = len(domain.dof_coordinates)
    beta_y = wave_tank_input.refine_y
    for i in range(Nd):
        _, y = domain.dof_coordinates[i]
        yc = h*sin(pi*y/h/2)
        domain.dof_coordinates[i][1] = beta_y*yc + (1-beta_y)*y
    
    # Get boundary info
    fs_info = [] # Free surface
    wm_info = [] # Wave maker
    nm_info: list[tuple[Literal["Nx", "Ny"], int]] = [] # Neumann dofs at the rest of the boundary
    for dof, coord in enumerate(domain.dof_coordinates):
        if domain.dof_type[dof] == hpcrec.DOF_TYPE_EXTERNAL:
            x, y = coord
            if x < 1e-8:
                wm_info.append((coord, dof))
            elif x > L*h - 1e-8:
                nm_info.append(('Nx', dof))
            elif y > h - 1e-8:
                fs_info.append((tuple(coord), dof))
            else:
                assert y < 1e-8
                nm_info.append(('Ny', dof))
    
    # Sort and store free surface dofs
    fs_info.sort()
    fs_dofs = [dof for _coord, dof in fs_info]
    fs_xpos = [coord[0] for coord, _dof in fs_info]
    Nf = len(fs_info)
    assert Nf > 5
    
    # Assemble the system matrix and rhs vector without boundary conditions
    A, b = hpcrec.assemble(domain)
    Nd = len(b)
    print(f"Number of unknowns: {Nd}")
    solver = hpcrec.LinearSolver()
    solver.reuse_preconditioner = True
    
    # Run time loop
    Nt = wave_tank_input.Nt
    tvec = np.linspace(0, wave_tank_input.tmax, Nt)
    dt = tvec[1]
    eta = np.zeros((Nt, Nf), float)
    phi_fs = np.zeros((Nt, Nf), float)
    phi = hpcrec.Vector(Nd)
    for it in range(1, Nt):
        t = tvec[it]
        
        # Calculate wave maker amplitude at y = h (still water height)
        wm_speed = 0
        max_speed = 0
        ramp = min(t/wti.tramp, 1)
        for ia, wm_ampl in enumerate(wave_tank_input.wm_ampls):  
            wm_freq = wave_tank_input.wm_freqs[ia]
            wm_phase = wave_tank_input.wm_phases[ia]
            wm_speed += wm_freq*wm_ampl*cos(wm_freq*t + wm_phase)*ramp
            max_speed += wm_freq*wm_ampl
        
        # Wave maker visualization
        iampl = int(round(39*(wm_speed + max_speed)/(2*max_speed)))
        wm_vis = ' '*iampl + '.' + ' '*(39 - iampl)
        
        # Print some info about the time step
        print(f"Timestep {it:4d} at t = {tvec[it]:6.3f}"),
        print(wm_vis),
        print(f"min/max(eta) = {eta[it-1].min():6.3f} {eta[it-1].max():6.3f}")
        
        #######################################################################
        # Update boundary conditions
        bcs: list[hpcrec.BcType] = []
        
        # Boundary conditions on the free surface
        for idof, dof in enumerate(fs_dofs):
            bcs.append(('D', dof, phi_fs[it-1,idof]))
        
        # Boundary conditions on the wave maker
        for coord, dof in wm_info:
            x, y = coord
            speed = y/h * wm_speed
            bcs.append(('Nx', dof, speed))
        
        # Boundary conditions on the rest of the Neumann boundary
        bcs += [(bc_type, dof, 0.0) for bc_type, dof in nm_info]
        
        #######################################################################
        # Apply boundary conditions
        if it == 1:
            hpcrec.apply_bcs(domain, A, b, bcs)
        else:
            for _bc_type, dof, value in bcs:
                b[dof] = value
        
        # Solve linear HPCrec equation system
        phi = hpcrec.Vector(Nd)
        solver.solve(A, phi, b)
        phi = phi.array()
        
        #######################################################################
        # Smoothing to avoid saw tooth patterns
        if it % 30:
            eta_smoothed = np.zeros(Nf, float)
            for ie in range(Nf):
                if ie == 0:
                    eta_smoothed[ie] = (eta[it,ie] + eta[it,ie+1] + eta[it,ie+2])/3
                elif ie == 1:
                    eta_smoothed[ie] = (eta[it,ie-1] + eta[it,ie] + eta[it,ie+1] + eta[it,ie+2])/4
                elif ie == Nf - 2:
                    eta_smoothed[ie] = (eta[it,ie-2] + eta[it,ie-1] + eta[it,ie] + eta[it,ie+1])/4
                elif ie == Nf - 1:
                    eta_smoothed[ie] = (eta[it,ie-2] + eta[it,ie-1] + eta[it,ie])/3
                else:
                    eta_smoothed[ie] = (eta[it,ie-2] + eta[it,ie-1] + eta[it,ie] + eta[it,ie+1] + eta[it,ie+2])/5
            #eta[it,:] = eta_smoothed#
            
        #######################################################################
        # Update free surface position (kinematic free surface condition)
        for ie in range(Nf):
            coord, dof = fs_info[ie]
            neighbours, _coeffs, _coeffs_diffx, coeffs_diffy = hpcrec.eval_phi(domain, dof)
            dphi_dy = 0
            for nb, coeff in zip(neighbours, coeffs_diffy):
                dphi_dy += coeff*phi[nb]
            eta[it, ie] = eta[it-1,ie] + dt*dphi_dy
        
        # Update phi at free surface (dynamic free surface condition)
        g = wave_tank_input.g
        for ie in range(Nf):
            phi_fs[it, ie] = phi_fs[it-1,ie] - dt*g*eta[it, ie]
    
    save_results('result_wave_tank_demo.out', wave_tank_input, fs_xpos, tvec, eta)
    
    if show_plot:
        from matplotlib import pyplot as plt
        #plt.spy(A)
        #uhpcrec.plot(domain)
        #uhpcrec.plot(domain, phi)
        from plot_wave_tank_results import plot_free_surface
        plot_free_surface(wti, fs_xpos, eta, tvec, 'Free surface elevation')
        #fs_plot(fs_info, phi_fs, tvec, 'Free surface potential')
        plt.show()


def save_results(file_name: str|Path, wave_tank_input: WaveTankInput, xpos: np.ndarray, tvec: np.ndarray, eta: np.ndarray):
    data = {'wave_tank_input': wave_tank_input,
            'xpos': xpos,
            'tvec': tvec,
            'eta': eta}
    with open(file_name, 'wb') as out:
        pickle.dump(data, out, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=10,
                        help='number of elements over the height')
    parser.add_argument('-L', type=int, default=4,
                        help='size of the domain in x in (size in y is 1), must be an integer')
    parser.add_argument('--tmax', type=float, default=1)
    parser.add_argument('-Nt', type=int, default=101,
                        help='Number of time steps')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='show plots')
    
    parser.add_argument('--backend', choices=('auto', 'scipy', 'petsc', 'numpy'), default='auto')
    parser.add_argument('--solver', default='default_direct')
    parser.add_argument('--preconditioner', default='')
    
    args = parser.parse_args()
    
    wti = WaveTankInput()
    wti.N = args.N
    wti.L = args.L
    wti.tmax = args.tmax
    wti.Nt = args.Nt
    wti.tramp = args.tmax/10
    
    hpcrec.parameters['linear_algebra_backend'] = args.backend
    hpcrec.parameters['solver'] = args.solver
    if args.preconditioner: hpcrec.parameters['preconditioner'] = args.preconditioner
    
    with hpcrec.Timer('Wave tank demo'):
        wavetank_demo(wti, show_plot=args.plot)
