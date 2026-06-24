import numpy as np

from .mesh import HPCDomain


def plot(domain: HPCDomain, q=None):
    """
    Plot the solution phi given a domain and a set of polynomial coefficients. 
    If q is none then only the mesh is plotted
    """
    dim = domain.geometric_dimension
    
    if dim == 2:
        _plot_2D(domain, q)
    else:
        raise NotImplementedError('Cannot plot domain in %dD' % dim)


def interactive():
    from matplotlib import pyplot
    pyplot.show()


def _plot_2D(domain: HPCDomain, solution: None | np.ndarray = None):
    from matplotlib import pyplot
    
    # Create triangulation of cells
    x = domain.dof_coordinates[:,0]
    y = domain.dof_coordinates[:,1]
    triangles = domain.triangles
    
    if solution is None:
        # Plot the mesh
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        ax.triplot(x, y, triangles)
    
    else:
        # Plot the solution in the vertices (possibly interpolated)
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        tp = ax.tripcolor(x, y, triangles, solution, edgecolors='k', shading='gouraud')
        pyplot.colorbar(tp)
