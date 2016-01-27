# encoding: utf8
from __future__ import division
from math import cosh, sinh, sin, cos, pi
import numpy
import scipy.sparse, scipy.sparse.linalg
import hpc


def split_shoebox_demo(N, L, h=1, k=1, show_plot=True, neumann=False, refine=False):
    print 'Calculating shoebox with N=%d, L=%d' % (N, L)
    
    # Create geometry
    domain1 = hpc.rectangle_domain((0, 0), (L, h), N*L//h, N)
    domain2 = hpc.rectangle_domain((L, 0), (L*2, h), N*L//h, N)
    
    # Assemble global systems
    A1, b1 = assemble_side_hpc(domain1, L, k, h)
    A2, b2 = assemble_side_hpc(domain2, L, k, h)

    # Find coupled dofs
    dofs1 = []
    for dof, coord in enumerate(domain1.dof_coordinates): 
        if coord[0] > L - 1e-8 and coord[1] > 1e-8 and coord[1] < h - 1e-8:
            dofs1.append(dof)
    dofs2 = []
    for dof, coord in enumerate(domain2.dof_coordinates): 
        if coord[0] < L + 1e-8 and coord[1] > 1e-8 and coord[1] < h - 1e-8:
            dofs2.append(dof)
    
    dofs1.sort(key=lambda dof: domain1.dof_coordinates[dof][1])
    dofs2.sort(key=lambda dof: domain2.dof_coordinates[dof][1])
    
    Q = len(dofs1)
    assert len(dofs2) == Q
    L1 = hpc.Matrix(Q, A1.shape[1])
    L2 = hpc.Matrix(Q, A2.shape[1])
    
    for i, (dof1, dof2) in enumerate(zip(dofs1, dofs2)):
        L1[i, dof1] = 1
        L2[i, dof2] = -1
    
    A1 = A1.csr_matrix
    A2 = A2.csr_matrix
    L1 = L1.csr_matrix
    L2 = L2.csr_matrix
    AA = scipy.sparse.bmat([[A1,   None, L1.T],
                            [None,   A2, L2.T],
                            [L1,     L2, None]], 'csc')
    bb = numpy.zeros(len(b1) + len(b2) + Q, float)
    bb[:len(b1)] = b1
    bb[len(b1):-Q] = b2
    
    if len(bb) < 20:
        print 'Block system matrix'
        print '   ', ' '.join('%6d' % i for i in range(AA.shape[0]))
        for i, row in enumerate(numpy.array(AA.todense())):
            print '%3d' % i,
            for v in row:
                print '%6.2g' % v,
            print
    
    # Solve global system
    with hpc.Timer('Solve'):
        try:
            lu = scipy.sparse.linalg.splu(AA)
            q = lu.solve(bb)
        except Exception as e:
            print 'ERROR:'
            print e
            print 'The global system matrix cannot be inverted!'
            exit()
    phi1 = q[:len(b1)]
    phi2 = q[len(b1):-Q]
    
    # Analytical solution
    for domain, phi in zip((domain1, domain2), (phi1, phi2)):
        phi_a = numpy.zeros_like(phi)
        for dof, coord in enumerate(domain.dof_coordinates):
            x, y = coord
            phi_a[dof] = cosh(k*(y+h))*sin(k*x)
    
        # Print the error
        print 'Error: %15.8e' % numpy.linalg.norm(phi - phi_a)
    
    if show_plot:
        from matplotlib import pyplot
        pyplot.spy(AA.todense())
        for domain, phi in zip((domain1, domain2), (phi1, phi2)):
            hpc.plot(domain)
            hpc.plot(domain, phi)
        pyplot.show()


def assemble_side_hpc(domain, L, k, h):
    """
    Solve the shoebox wave problem using HPC
    """
    # Boundary conditions
    bcs = []
    dirichlet_dofs = []
    for dof, coord in enumerate(domain.dof_coordinates): 
        if domain.dof_type[dof] == hpc.DOF_TYPE_EXTERNAL:
            x, y = coord
            if y > h - 1e-8:
                bcs.append(('D',  dof,   cosh(k*(y+h))*sin(k*x)))
                dirichlet_dofs.append(dof)
            elif x < 1e-8:
                bcs.append(('Nx', dof, k*cosh(k*(y+h))*cos(k*x)))
            elif x > 2*L - 1e-8:
                bcs.append(('Nx', dof, k*cosh(k*(y+h))*cos(k*x)))
            elif y < 1e-8:
                bcs.append(('Ny', dof, k*sinh(k*(y+h))*sin(k*x)))
    
    # Setup global equation system
    A, b = hpc.assemble(domain)
    hpc.apply_bcs(domain, A, b, bcs)
    
    if len(b) < 10:
        print 'Global system matrix'
        print '   ', ' '.join('%8d' % i for i in range(A.shape[0]))
        for i, row in enumerate(A.array()):
            print '%3d' % i,
            for v in row:
                print '%8.2g' % v,
            print
        
        print 'DOF coordinates'
        for i, c in enumerate(domain.dof_coordinates):
            print '%3d - %8.2g %8.2g' % (i, c[0], c[1])
    
    return A, b 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=10,
                        help='number of elements over the height')
    parser.add_argument('-L', type=int, default=4,
                        help='size of the domain in x in (size in y is 1), must be an integer')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='show plots')
    
    args = parser.parse_args()
    
    hpc.parameters['linear_algebra_backend'] = 'scipy'
    hpc.parameters['solver'] = 'splu'
    
    with hpc.Timer('Shoebox demo'):
        try:
            split_shoebox_demo(N=args.N,
                               L=args.L,
                               show_plot=args.plot)
        except hpc.HPCError as e:
            print 'ERROR - '*9 + 'ERROR!!!\n'
            print '   ', e 
            print '\n' + 'ERROR - '*9 + 'ERROR!!!'

