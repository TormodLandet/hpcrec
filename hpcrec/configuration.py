from contextlib import contextmanager


def get_default_parameters():
    return {
        # numpy, scipy, petsc or auto (which is numpy for small matrices)
        "linear_algebra_backend": "auto",
        # Iterative KSP solver parameters
        "relative_tolerance": 1e-15,
        "absolute_tolerance": 1e-15,
        "divergence_limit": 10000,
        "max_iterations": 10000,
        "solver": "gmres",
        "preconditioner": "bjacobi",
    }


# Global configuration ala FEniCS
parameters = get_default_parameters()


@contextmanager
def local_parameters(**kwargs):
    """
    Locally set the configuration parameters for the HPC solver.
    """
    old_parameters = parameters.copy()
    parameters.update(kwargs)
    try:
        yield
    finally:
        parameters.update(old_parameters)
