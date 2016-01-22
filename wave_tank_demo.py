# encoding: utf8
from __future__ import division
from math import sin
import numpy
import hpc


class WaveTankInput(object):
    N = None
    L = None
    
    h = 1
    k = 1
    g = 9.81
    
    tmax = 1
    Nt = 100
    
    wm_ampl = 0.1
    wm_freq = 5


def wavetank_demo(wave_tank_input, show_plot=True):
    N = wave_tank_input.N
    L = wave_tank_input.L
    h = wave_tank_input.h
    print 'Calculating wavetank with N=%d, h=%d, L=%d' % (N, h, L)
    
    # Setup geometry
    domain = hpc.rectangle_domain((0, 0), (L, h), N*L//h, N)
    
    # Get boundary info
    fs_info = [] # Free surface
    wm_info = [] # Wave maker
    nm_info = [] # Neumann dofs at the rest of the boundary
    for dof, coord in enumerate(domain.dof_coordinates):
        if domain.dof_type[dof] == hpc.DOF_TYPE_EXTERNAL:
            x, y = coord
            if y + 1e-8 > h:
                fs_info.append((tuple(coord), dof))
            elif y < 1e-8:
                nm_info.append(('Ny', dof))
            elif x < 1e-8:
                wm_info.append((coord, dof))
            elif x > L*h - 1e-8:
                nm_info.append(('Nx', dof))
    
    # Sort free surface dofs and report locations
    fs_info.sort()
    fs_dofs = [dof for _coord, dof in fs_info]
    fs_xpos = [coord[0] for coord, _dof in fs_info]
    Nf = len(fs_info)
    assert Nf > 5
    
    # Setup matrix and vector without boundary conditions
    A, b = hpc.assemble(domain)
    Nd = len(b)
    print 'Number of unknowns: %d' % len(b)
    
    # Run time loop
    Nt = wave_tank_input.Nt
    tvec = numpy.linspace(0, wave_tank_input.tmax, Nt)
    dt = tvec[1]
    
    eta = numpy.zeros((Nt, Nf), float)
    phi_fs = numpy.zeros((Nt, Nf), float)
    phi = hpc.Vector(Nd)
    for it in range(1, Nt):
        t = tvec[it]
        wm_ampl = wave_tank_input.wm_ampl
        wm_freq = wave_tank_input.wm_freq
        ampl = wm_ampl*sin(wm_freq*t)
        iampl = int(round(39*(ampl + wm_ampl)/(2*wm_ampl)))
        
        # Print some info about the time step
        print 'Timestep %4d at t = %6.3f' % (it, tvec[it]),
        print ' '*iampl + '*' + ' '*(39 - iampl),
        print 'min/max(eta) = % 6.3f %6.3f' % (eta[it-1].min(), eta[it-1].max())
        
        # Update boundary conditions
        bcs = []
        
        # Boundary conditions on the free surface
        for idof, dof in enumerate(fs_dofs):
            bcs.append(('D', dof, phi_fs[it-1,idof]))
        
        # Boundary conditions on the wave maker
        for coord, dof in wm_info:
            speed = coord[1]/h * ampl
            bcs.append(('Nx', dof, speed))
        
        # Boundary conditions on the rest of the Neumann boundary
        bcs += [(bc_type, dof, 0) for bc_type, dof in nm_info]
        
        # Apply boundary conditions
        hpc.apply_bcs(domain, A, b, bcs)
        
        # Solve
        hpc.solve(A, phi, b)
        
        #######################################################################
        
        # Smoothing to avoid saw tooth patterns
        if it % 30:
            eta_smoothed = numpy.zeros(Nf, float)
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
        
        # Update free surface position (kinematic free surface condition)
        for ie in range(Nf):
            coord, dof = fs_info[ie]
            neighbours, _coeffs, _coeffs_diffx, coeffs_diffy = hpc.eval_phi(domain, dof)
            dphi_dy = 0
            for nb, coeff in zip(neighbours, coeffs_diffy):
                dphi_dy += coeff*phi[nb]
            eta[it, ie] = eta[it-1,ie] + dt*dphi_dy
        
        # Update phi at free surface (dynamic free surface condition)
        g = wave_tank_input.g
        for ie in range(Nf):
            phi_fs[it, ie] = phi_fs[it-1,ie] - dt*g*eta[it, ie]
    
    save_results('result_wave_tank_demo.out', fs_xpos, tvec, eta)
    
    if show_plot:
        from matplotlib import pyplot
        #pyplot.spy(A)
        #uhpc.plot(domain)
        #uhpc.plot(domain, phi)
        from plot_wave_tank_results import plot_free_surface
        plot_free_surface(fs_xpos, eta, tvec, 'Free surface elevation')
        #fs_plot(fs_info, phi_fs, tvec, 'Free surface potential')
        pyplot.show()


def save_results(file_name, xpos, tvec, eta):
    with open(file_name, 'wt') as out:
        out.write('xpos %d\n' % len(xpos))
        out.write(' '.join('% 14.8e' % v for v in xpos))
        out.write('\n')
        
        out.write('tvec %d\n' % len(tvec))
        out.write(' '.join('% 14.8e' % v for v in tvec))
        out.write('\n')
        
        out.write('eta %d %d\n' % eta.shape)
        for row in eta:
            out.write(' '.join('% 14.8e' % v for v in row))
            out.write('\n')


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
    parser.add_argument('--solver', default='')
    parser.add_argument('--preconditioner', default='')
    
    args = parser.parse_args()
    
    wti = WaveTankInput()
    wti.N = args.N
    wti.L = args.L
    wti.tmax = args.tmax
    wti.Nt = args.Nt
    
    hpc.parameters['linear_algebra_backend'] = args.backend
    if args.solver: hpc.parameters['solver'] = args.solver
    if args.preconditioner: hpc.parameters['preconditioner'] = args.preconditioner
    
    with hpc.Timer('Wave tank demo'):
        wavetank_demo(wti, show_plot=args.plot)
