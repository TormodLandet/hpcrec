import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.tri import Triangulation


from .test_cases.wave_kinematics import (
    FreeSurfaceConformingDomain,
    compute_velocity,
    solve_wave_kinematics,
)
from .timer import Timer


@dataclasses.dataclass
class HpcWaveKinematics:
    """
    The results from a wave kinematics reconstruction using HPC.
    """

    #: Matplotlib Triangulation of the wave kinematics domain in the wave coordinate system
    triangulated_domain: Triangulation

    #: Length of the domain in the x-direction
    length: float

    #: Depth of the domain in the z-direction
    #: (positive value, z=0 at the mean free surface, z=-depth at the bottom; z>0 for crests)
    depth: float

    #: x-coordinates of the free surface in the wave coordinate system
    x_eta: np.ndarray

    #: z-coordinates of the free surface in the wave coordinate system
    z_eta: np.ndarray

    #: Reconstructed velocity potential at the DOFs in the wave coordinate system
    phi: np.ndarray

    #: Reconstructed horizontal velocity at the DOFs in the wave coordinate system
    u: np.ndarray

    #: Reconstructed vertical velocity at the DOFs in the wave coordinate system
    w: np.ndarray

    def z_profile(
        self, x: float, x0: float = 0.0, y0: float = 0.0, beta: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the vertical profile of the wave kinematics at a given x-coordinate.

        Parameters
        ----------
        x : float
            The x-coordinate at which to get the vertical profile (user coordinate system)
        x0 : float, optional
            The x-coordinate offset between the wave and user coordinate systems, by default 0.
        y0 : float, optional
            The y-coordinate offset between the wave and user coordinate systems, by default 0.
        beta : float, optional
            The rotation angle (in degrees) between the wave and user coordinate systems, by
            default 0. Only beta=0 and beta=180 make sense since the triangulation is 2D.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the z-coordinates, u-velocities, and w-velocities at the specified
            x-coordinate (input coord and output velocity are in user coordinate system).
        """
        beta_rad = beta * np.pi / 180.0
        x_wave = (x * np.cos(beta_rad) + x0) % self.length

        # Find the closest x-coordinate in the triangulated_domain to the specified x
        closest_x_idx = np.argmin(np.abs(self.triangulated_domain.x - x_wave))
        closest_x = self.triangulated_domain.x[closest_x_idx]

        # Find the indices of the points in the triangulated_domain that are closest to the specified x-coordinate
        idx = np.where(np.isclose(self.triangulated_domain.x, closest_x))[0]

        # Extract the corresponding z-coordinates and velocities
        z_profile = self.triangulated_domain.y[idx]
        u_profile = self.u[idx]
        w_profile = self.w[idx]

        # Rotate the velocity field by the specified angle beta (in degrees)
        beta_rad = beta * np.pi / 180.0
        v = 0.0  # Assuming v=0 for 2D wave
        u_rotated = u_profile * np.cos(beta_rad) - v * np.sin(beta_rad)

        return z_profile, u_rotated, w_profile

    def triangulation(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        beta: float = 0.0,
        mirror_twice: bool = False,
    ) -> Triangulation:
        """
        Get the triangulation of the wave kinematics triangulated_domain in the user coordinate
        system (after applying beta rotation and translation). Optionally, the triangulation can be
        concatenated before and after for plotting near the periodic boundaries.

        Parameters
        ----------
        x0 : float, optional
            The x-coordinate offset between the wave and user coordinate systems, by default 0.
        y0 : float, optional
            The y-coordinate offset between the wave and user coordinate systems, by default 0.
        beta : float, optional
            The rotation angle (in degrees) between the wave and user coordinate systems, by
            default 0. Only beta=0 and beta=180 make sense since the triangulation is 2D.
        mirror_twice : bool, optional
            Whether to concatenate the triangulation before and after for plotting
            (the domain is periodic in x), by default False.
        Returns
        -------
        Triangulation
            A matplotlib Triangulation object representing the wave kinematics triangulated_domain
            in the user coordinate system (after applying beta rotation and translation).
        """
        from matplotlib.tri import Triangulation

        x = self.triangulated_domain.x
        z = self.triangulated_domain.y
        triangles = self.triangulated_domain.triangles

        if mirror_twice:
            Ncoord = len(x)
            x = np.concatenate([x - self.length, x, x + self.length])
            z = np.concatenate([z, z, z])
            triangles = np.concatenate(
                [
                    triangles,
                    triangles + Ncoord,
                    triangles + 2 * Ncoord,
                ]
            )

        # Apply rotation and translation to the triangulated_domain coordinates
        # (Only makes sense for beta=0 and beta=180)
        beta_rad = beta * np.pi / 180.0
        y = 0.0
        x_rotated = (x - x0) * np.cos(beta_rad) - (y - y0) * np.sin(beta_rad)
        # y_rotated = (x - x0) * np.sin(beta_rad) + (y - y0) * np.cos(beta_rad)
        z_rotated = z

        return Triangulation(x_rotated, z_rotated, triangles)

    def velocity_field(
        self, beta: float = 0.0, mirror_twice: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the velocity field of the wave kinematics triangulated_domain after rotation.

        Parameters
        ----------
        beta : float, optional
            The rotation angle (in degrees) between the wave and user coordinate systems, by
            default 0. Only beta=0 and beta=180 make sense since the triangulation is 2D.
        mirror_twice : bool, optional
            Whether to concatenate the velocity field before and after for plotting
            (the domain is periodic in x), by default False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the rotated u and w velocities (in user coordinate system)
        """
        # Rotate the velocity field by the specified angle beta (in degrees)
        beta_rad = beta * np.pi / 180.0
        v = 0.0
        u_rotated = self.u * np.cos(beta_rad) - v * np.sin(beta_rad)
        w_rotated = self.w  # Vertical velocity remains unchanged in this rotation

        if mirror_twice:
            u_rotated = np.concatenate([u_rotated, u_rotated, u_rotated])
            w_rotated = np.concatenate([w_rotated, w_rotated, w_rotated])

        return u_rotated, w_rotated


def reconstruct_wave_kinematics_using_hpc(
    length: float,
    depth: float,
    eta: np.ndarray,
    psi: np.ndarray,
    oversample_factor: int = 4,
    verbose: bool = False,
) -> HpcWaveKinematics:
    """
    Reconstruct wave kinematics using HPC.

    Parameters
    ----------
    length : float
        Length of the domain.
    depth : float
        Depth of the domain.
    eta : np.ndarray
        Free surface elevation.
    psi : np.ndarray
        Stream function.
    oversample_factor : int, optional
        Factor by which to oversample the HOSM solution, by default 4.
    verbose : bool, optional
        Whether to print verbose output, by default False.
    Returns
    -------
    HpcWaveKinematics
        Dataclass containing the reconstructed wave kinematics.
    """
    from matplotlib.tri import Triangulation

    # Construct the free-surface-conforming wave domain
    with Timer("Creating free-surface-conforming domain", verbose=verbose):
        domain = FreeSurfaceConformingDomain(length, depth, eta, oversample=oversample_factor)
        if verbose:
            print_domain_info(domain)

    # Solve for the velocity potential (this is what takes time)
    with Timer("Solving for the velocity potential", verbose=verbose):
        phi = solve_wave_kinematics(domain, psi, oversample=oversample_factor)

    # Compute the velocity field (~ instant since we have coefficients in the cache)
    u, w = compute_velocity(domain, phi)

    # Create triangulation of cells (~ instant)
    x = domain.dof_coordinates[:, 0]
    y = domain.dof_coordinates[:, 1]
    triangles = domain.triangles
    tri = Triangulation(x, y, triangles)

    return HpcWaveKinematics(
        triangulated_domain=tri,
        length=length,
        depth=depth,
        x_eta=domain.x_fs,
        z_eta=domain.z_fs,
        phi=phi,
        u=u,
        w=w,
    )


def print_domain_info(domain: FreeSurfaceConformingDomain):
    """
    Print information about the free-surface-conforming domain.

    Parameters
    ----------
    domain : FreeSurfaceConformingDomain
        The domain to print information about.
    """
    Nx, Nz = domain.grid_shape
    print("  Domain Information:")
    print(f"    Length: {domain.length}")
    print(f"    Depth: {domain.depth}")
    print(f"    Constant  dx: {domain.dx}")
    print(f"    Depth avg dz: {domain.depth / Nz}")
    print(f"    Grid shape: {Nx} x {Nz}  (oversample factor: {domain.oversample})")
    print(f"    Number of DOFs: {domain.dof_coordinates.shape[0]}")
