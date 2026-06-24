import numpy as np

from .mesh import HPCDomain


def plot(domain: HPCDomain, phi: np.ndarray | None = None, ax=None, label: str | None = None):
    """
    Plot the solution phi given a domain and a set of polynomial coefficients.
    If phi is none then only the mesh is plotted
    """
    dim = domain.geometric_dimension

    if dim == 2:
        _plot_2D(domain, phi, ax=ax, label=label)
    else:
        raise NotImplementedError("Cannot plot domain in %dD" % dim)


def interactive():
    from matplotlib import pyplot

    pyplot.show()


def _plot_2D(domain: HPCDomain, phi: None | np.ndarray = None, ax=None, label=None):
    from matplotlib import pyplot as plt

    # Create triangulation of cells
    x = domain.dof_coordinates[:, 0]
    y = domain.dof_coordinates[:, 1]
    triangles = domain.triangles

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if phi is None:
        # Plot the mesh
        ax.triplot(x, y, triangles, color="k", linewidth=0.5)
    else:
        # Plot the solution phi in the vertices (possibly interpolated)
        ax.tripcolor(x, y, triangles, phi, shading="gouraud")
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        if label is not None:
            cbar.set_label(label)
