# encoding: utf8
from __future__ import division
import numpy
import hpc
from math import cosh, sinh, sin, cos, pi


def shoebox_demo(N, L, h=1, k=1, show_plot=True, neumann=False, refine_y=0.0, fem=0):
    print 'Calculating shoebox with N=%d, L=%d' % (N, L)
    
    # Create geometry
    with hpc.Timer('Geometry'):
        domain = hpc.rectangle_domain((0, 0), (L, h), N*L//h, N)
    
    # Get layer dofs
    dofs_layer1 = get_dofs(domain.dof_coordinates, h)
    dofs_layer2 = get_dofs(domain.dof_coordinates, h * (N-1)/N)
    dofs_layer3 = get_dofs(domain.dof_coordinates, h * (N-2)/N)
    dofs_layer4 = get_dofs(domain.dof_coordinates, h * (N-3)/N)
    
    # Apply refinement towwards the top boundary
    Nd = len(domain.dof_coordinates)
    alpha = 1
    beta_x = 0.0
    beta_y = refine_y
    for i in range(Nd):
        x, y = domain.dof_coordinates[i]
        xc = L*sin(pi*x/L/2)**alpha
        yc = h*sin(pi*y/h/2)**alpha
        domain.dof_coordinates[i] = (beta_x*xc + (1-beta_x)*x,
                                     beta_y*yc + (1-beta_y)*y)
    
    # Get distances between layers
    dy1 = domain.dof_coordinates[dofs_layer1[0]][1] - domain.dof_coordinates[dofs_layer2[0]][1]
    dy2 = domain.dof_coordinates[dofs_layer2[0]][1] - domain.dof_coordinates[dofs_layer3[0]][1]
    dy3 = domain.dof_coordinates[dofs_layer3[0]][1] - domain.dof_coordinates[dofs_layer4[0]][1]
    
    # Assemble global system
    with hpc.Timer('Assemble'):
        if fem > 0:
            A, b, solve = shoebox_demo_fem(domain, fem, k, h, dofs_layer1, neumann)
        else:
            A, b, solve = shoebox_demo_hpc(domain, L, k, h, dofs_layer1, neumann)
    
    # Print system info
    Ndof = len(b)
    if Ndof < 100:
        maxnz = 0
        for row in A:
            nz = sum(1 if v != 0 else 0 for v in row)
            maxnz = max(maxnz, nz)
        print 'Maximum number of non zeros in a row:', maxnz
        #print 'Condition number', numpy.linalg.cond(A)
    
    # Solve global system
    with hpc.Timer('Solve'):
        try:
            phi_h, d_phi_dy_h = solve()
        except Exception as e:
            print 'ERROR:'
            print e
            print 'The global system matrix cannot be inverted!'
            exit()
    assert d_phi_dy_h.shape == dofs_layer1.shape
    
    # Analytical solution
    phi_a = numpy.zeros_like(phi_h)
    for dof, coord in enumerate(domain.dof_coordinates):
        x, y = coord
        phi_a[dof] = cosh(k*(y+h))*sin(k*x)
    
    # Analytical solution of d_phi_dz on the free surface
    d_phi_dy_a = numpy.zeros_like(d_phi_dy_h)
    for i, dof in enumerate(dofs_layer1):
        x, y = domain.dof_coordinates[dof]
        d_phi_dy_a[i] = k*sinh(k*(y+h))*sin(k*x)
        
    # Dinite difference solutions
    phi_layer_1 = phi_h[dofs_layer1]
    phi_layer_2 = phi_h[dofs_layer2]
    phi_layer_3 = phi_h[dofs_layer3]
    phi_layer_4 = phi_h[dofs_layer4]
    if dy1 == dy2:
        dy = dy1
        d_phi_dy_fdm1 = (phi_layer_1 - phi_layer_2)/dy
        d_phi_dy_fdm2 = (3*phi_layer_1 - 4*phi_layer_2 + phi_layer_3)/(2*dy)
        d_phi_dy_fdm3 = (11*phi_layer_1 - 18*phi_layer_2 + 9*phi_layer_3 - 2*phi_layer_4)/(6*dy)
    else:
        # Sympy derived expresions for dy.. != dy..
        d_phi_dy_fdm1 = (phi_layer_1 - phi_layer_2)/dy1
        d_phi_dy_fdm2 = (2*phi_layer_1*dy1*dy2 + phi_layer_1*dy2**2 - phi_layer_2*dy1**2 - 2*phi_layer_2*dy1*dy2 - 
                         phi_layer_2*dy2**2 + phi_layer_3*dy1**2)/(dy1*dy2*(dy1 + dy2))
        d_phi_dy_fdm3 = (3*phi_layer_1*dy1**2*dy2**2*dy3 + 3*phi_layer_1*dy1**2*dy2*dy3**2 + 4*phi_layer_1*dy1*dy2**3*dy3 + 
                         6*phi_layer_1*dy1*dy2**2*dy3**2 + 2*phi_layer_1*dy1*dy2*dy3**3 + phi_layer_1*dy2**4*dy3 + 
                         2*phi_layer_1*dy2**3*dy3**2 + phi_layer_1*dy2**2*dy3**3 - phi_layer_2*dy1**4*dy3 - 
                         4*phi_layer_2*dy1**3*dy2*dy3 - 2*phi_layer_2*dy1**3*dy3**2 - 6*phi_layer_2*dy1**2*dy2**2*dy3 - 
                         6*phi_layer_2*dy1**2*dy2*dy3**2 - phi_layer_2*dy1**2*dy3**3 - 4*phi_layer_2*dy1*dy2**3*dy3 - 
                         6*phi_layer_2*dy1*dy2**2*dy3**2 - 2*phi_layer_2*dy1*dy2*dy3**3 - phi_layer_2*dy2**4*dy3 - 
                         2*phi_layer_2*dy2**3*dy3**2 - phi_layer_2*dy2**2*dy3**3 + phi_layer_3*dy1**4*dy2 + 
                         phi_layer_3*dy1**4*dy3 + 2*phi_layer_3*dy1**3*dy2**2 + 4*phi_layer_3*dy1**3*dy2*dy3 + 
                         2*phi_layer_3*dy1**3*dy3**2 + phi_layer_3*dy1**2*dy2**3 + 3*phi_layer_3*dy1**2*dy2**2*dy3 + 
                         3*phi_layer_3*dy1**2*dy2*dy3**2 + phi_layer_3*dy1**2*dy3**3 - phi_layer_4*dy1**4*dy2 - 
                         2*phi_layer_4*dy1**3*dy2**2 - phi_layer_4*dy1**2*dy2**3)/(dy1*dy2*dy3*(dy1**2*dy2 + dy1**2*dy3 + 
                         2*dy1*dy2**2 + 3*dy1*dy2*dy3 + dy1*dy3**2 + dy2**3 + 2*dy2**2*dy3 + dy2*dy3**2))
    
    # Print the errors
    print 'Error phi          : %15.8e' % numpy.linalg.norm(phi_h - phi_a)
    print 'Error d_phi_dy     : %15.8e' % numpy.linalg.norm(d_phi_dy_h - d_phi_dy_a)
    print 'Error d_phi_dy FDM1: %15.8e' % numpy.linalg.norm(d_phi_dy_fdm1 - d_phi_dy_a)
    print 'Error d_phi_dy FDM2: %15.8e' % numpy.linalg.norm(d_phi_dy_fdm2 - d_phi_dy_a)
    print 'Error d_phi_dy FDM3: %15.8e' % numpy.linalg.norm(d_phi_dy_fdm3 - d_phi_dy_a)
    
    if False:
        from matplotlib import pyplot
        pyplot.plot(d_phi_dy_a, c='k', label='Analytical')
        pyplot.plot(d_phi_dy_h, label='Numerical')
        pyplot.plot(d_phi_dy_fdm1, label='FDM 1')
        pyplot.plot(d_phi_dy_fdm2, label='FDM 2')
        pyplot.plot(d_phi_dy_fdm3, label='FDM 3')
        pyplot.legend(loc='best')
        pyplot.show()
    
    if show_plot:
        from matplotlib import pyplot
        pyplot.spy(A)
        hpc.plot(domain)
        hpc.plot(domain, phi_h)
        pyplot.show()


def shoebox_demo_hpc(domain, L, k, h, dofs_layer1, neumann=False):
    """
    Solve the shoebox wave problem using HPC
    """
    # Boundary conditions
    bcs = []
    dirichlet_dofs = []
    for dof, coord in enumerate(domain.dof_coordinates): 
        if domain.dof_type[dof] == hpc.DOF_TYPE_EXTERNAL:
            x, y = coord
            if not neumann:
                bcs.append(('D',  dof,   cosh(k*(y+h))*sin(k*x)))
                dirichlet_dofs.append(dof)
            else:
                if y > h - 1e-8:
                    bcs.append(('D',  dof,   cosh(k*(y+h))*sin(k*x)))
                    dirichlet_dofs.append(dof)
                elif x < 1e-8:
                    bcs.append(('Nx', dof, k*cosh(k*(y+h))*cos(k*x)))
                elif x > L*h - 1e-8:
                    bcs.append(('Nx', dof, k*cosh(k*(y+h))*cos(k*x)))
                else:
                    assert y < 1e-8
                    bcs.append(('Ny', dof, k*sinh(k*(y+h))*sin(k*x)))
    
    # Setup global equation system
    A, b = hpc.assemble(domain)
    hpc.apply_bcs(domain, A, b, bcs)
    print 'Number of unknowns: %d' % len(b)
    
    if False:
        print 'Global system matrix'
        print '   ', ' '.join('%8d' % i for i in range(A.shape[0]))
        for i, row in enumerate(A):
            print '%3d' % i,
            for v in row:
                print '%8.2g' % v,
            print
        
        print 'DOF coordinates'
        for i, c in enumerate(domain.dof_coordinates):
            print '%3d - %8.2g %8.2g' % (i, c[0], c[1])
    
    def solve():
        phi_h = hpc.Vector(len(b))
        nit = hpc.solve(A, phi_h, b)
        print 'Done in %d iterations' % nit
        
        # Find the derivative at the free surface
        d_phi_dy_h = numpy.zeros(len(dofs_layer1), float)
        for i, dof in enumerate(dofs_layer1):
            neighbours, _coeffs, _coeffs_diffx, coeffs_diffy = hpc.eval_phi(domain, dof)
            for j, dof_j in enumerate(neighbours):
                d_phi_dy_h[i] += coeffs_diffy[j]*phi_h[dof_j]
        
        return phi_h.array(), d_phi_dy_h
    
    A.finalize()
    b.finalize()
    return A.array(), b.array(), solve 


def shoebox_demo_fem(domain, order, k, h, dofs_layer1, neumann=False):
    """
    Solve the shoebox wave problem using FEM (FEniCS)
    """
    import dolfin as df
    from dolfin import grad, dot, dx, ds
    mesh = domain.to_fenics()
    V = df.FunctionSpace(mesh, 'CG', order)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    
    # Dirichlet and Neumann functions 
    gd = df.Expression("cosh(k*(x[1]+h))*sin(k*x[0])", k=k, h=h)
    gn = df.Expression(["k*cosh(k*(x[1]+h))*cos(k*x[0])",
                        "k*sinh(k*(x[1]+h))*sin(k*x[0])"],
                       k=k, h=h)
    
    # Define and assemble the weak form
    a = dot(grad(u), grad(v))*dx
    L = dot(gn, n)*v*ds
    A = df.assemble(a)
    b = df.assemble(L)
    print 'Number of unknowns: %d' % len(b)
    
    # Apply Dirichlet boundary condition
    def dirichlet_boundary(x, on_boundary):
        if neumann:
            return on_boundary and x[1] > h - 1e-8
        else:
            return on_boundary
    dbc = df.DirichletBC(V, gd, dirichlet_boundary)
    dbc.apply(A, b)
    
    def solve():
        q = df.Function(V)
        nit = df.solve(A, q.vector(), b)
        print 'Done in %d iterations' % nit
        phi_h = q.compute_vertex_values()
        
        # Find the derivative at the free surface
        a2 = u*v*dx
        L2 = df.Dx(q, 1)*v*dx
        A2 = df.assemble(a2)
        b2 = df.assemble(L2)
        q2 = df.Function(V)
        df.solve(A2, q2.vector(), b2)
        q2arr = q2.compute_vertex_values() 
        d_phi_dy_h = q2arr[dofs_layer1]
        
        return phi_h, d_phi_dy_h
    
    return A.array(), b.array(), solve


def get_dofs(coords, vert_pos, eps=1e-6):
    """
    Get the dofs at a given vertical position
    """
    # Indices of the top vertices
    idx = numpy.where((coords[:,1] > vert_pos - eps) &
                      (coords[:,1] < vert_pos + eps))[0]
    
    # Sort indices by x coordinate of the corresponding vertex
    idx = list(idx)
    idx.sort(key=lambda i: coords[i,0])
    idx = numpy.array(idx, int)
    
    return idx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=10,
                        help='number of elements over the height')
    parser.add_argument('-L', type=int, default=4,
                        help='size of the domain in x in (size in y is 1), must be an integer')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='show plots')
    parser.add_argument('--neumann', '-n', action='store_true',
                        help='include Neumann boundaries')
    parser.add_argument('--fem', '-f', type=int, default=0,
                        help='use FEM instead of HPC (specify order of method as argument)')
    parser.add_argument('--refine_y', type=float, default=0.0,
                        help='refine the mesh vertically by by ammount 0 to 1')
    
    parser.add_argument('--backend', choices=('auto', 'scipy', 'petsc', 'numpy'), default='auto')
    parser.add_argument('--solver', default='')
    parser.add_argument('--preconditioner', default='')
    
    args = parser.parse_args()
    
    hpc.parameters['linear_algebra_backend'] = args.backend
    if args.solver: hpc.parameters['solver'] = args.solver
    if args.preconditioner: hpc.parameters['preconditioner'] = args.preconditioner
    
    with hpc.Timer('Shoebox demo'):
        try:
            shoebox_demo(N=args.N,
                         L=args.L,
                         show_plot=args.plot,
                         neumann=args.neumann,
                         refine_y=args.refine_y,
                         fem=args.fem)
        except hpc.HPCError as e:
            print 'ERROR - '*9 + 'ERROR!!!\n'
            print '   ', e 
            print '\n' + 'ERROR - '*9 + 'ERROR!!!'
