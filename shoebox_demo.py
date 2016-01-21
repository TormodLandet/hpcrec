# encoding: utf8
from __future__ import division
import numpy
import hpc
from math import cosh, sinh, sin, cos, pi


def shoebox_demo(N, L, h=1, k=1, show_plot=True, neumann=False, refine=False, fem=0):
    print 'Calculating shoebox with N=%d, L=%d' % (N, L)
    
    # Create geometry
    with hpc.Timer('Geometry'):
        domain = hpc.rectangle_domain((0, 0), (L, h), N*L//h, N)
    
    # Apply refinement
    if refine:
        Nd = len(domain.dof_coordinates)
        alpha = 1
        beta_x = 1.0
        beta_y = 0.0
        for i in range(Nd):
            x, y = domain.dof_coordinates[i]
            xc = L/2*(1 - cos(pi*x/L)**alpha)
            yc = h/2*(1 - cos(pi*y/h)**alpha)
            domain.dof_coordinates[i] = (beta_x*xc + (1-beta_x)*x,
                                            beta_y*yc + (1-beta_y)*y)
    
    # Assemble global system
    with hpc.Timer('Assemble'):
        if fem > 0:
            A, b, solve = shoebox_demo_fem(domain, fem, k, h, neumann)
        else:
            A, b, solve = shoebox_demo_hpc(domain, L, k, h, neumann)
    
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
            phi_h = solve()
        except Exception as e:
            print 'ERROR:', e.message
            print 'The global system matrix cannot be inverted!'
            exit()
    
    # Analytical solution
    phi_a = numpy.zeros_like(phi_h)
    for dof, coord in enumerate(domain.dof_coordinates):
        x, y = coord
        phi_a[dof] = cosh(k*(y+h))*sin(k*x)
    
    # Print the error
    print 'Error: %15.8e' % numpy.linalg.norm(phi_h - phi_a)
    
    if show_plot:
        from matplotlib import pyplot
        pyplot.spy(A)
        hpc.plot(domain)
        hpc.plot(domain, phi_h)
        pyplot.show()


def shoebox_demo_hpc(domain, L, k, h, neumann=False):
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
        hpc.solve(A, phi_h, b)
        return phi_h
    
    return A, b, solve 


def shoebox_demo_fem(domain, order, k, h, neumann=False):
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
        df.solve(A, q.vector(), b)
        phi_h = q.compute_vertex_values()
        return phi_h
    
    return A.array(), b.array(), solve


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
    args = parser.parse_args()
    
    with hpc.Timer('Shoebox demo'):
        shoebox_demo(N=args.N,
                     L=args.L,
                     show_plot=args.plot,
                     neumann=args.neumann,
                     fem=args.fem)
