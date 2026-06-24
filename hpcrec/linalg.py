from __future__ import annotations
from typing import TypeAlias, Literal

import numpy as np
import numpy.linalg
import numpy.typing

from hpcrec import parameters, HPCError


# Check for scipy
try:
    import scipy.sparse
    import scipy.sparse.linalg

    has_scipy = True
except ImportError:
    has_scipy = False


# Check for PETSc
try:
    import petsc4py

    petsc4py.init()
    from petsc4py import PETSc

    has_petsc = True
except Exception:
    has_petsc = False


ArrayLike: TypeAlias = numpy.ndarray | list[float] | list[list[float]] | numpy.typing.ArrayLike
LinalgBackendType: TypeAlias = Literal["numpy", "scipy", "petsc"]
LINALG_BACKEND_OPTIONS = ("numpy", "scipy", "petsc", "auto")


def get_linalg_backend_type() -> LinalgBackendType:
    available_backends = ["numpy"]
    if has_scipy:
        available_backends.append("scipy")
    if has_petsc:
        available_backends.append("petsc")

    param_backend = parameters["linear_algebra_backend"]
    if param_backend not in LINALG_BACKEND_OPTIONS:
        raise HPCError(
            f"Invalid linear algebra backend {param_backend!r}."
            f" Select one of {', '.join(repr(b) for b in LINALG_BACKEND_OPTIONS)}."
        )

    if param_backend == "auto":
        if has_petsc:
            return "petsc"
        elif has_scipy:
            return "scipy"
        else:
            return "numpy"
    else:
        return param_backend


def Matrix(
    N: int,
    M: int,
    data: list[float] | None = None,
    indices: list[int] | None = None,
    indptr: list[int] | None = None,
) -> GenericMatrix:
    """
    Get a matrix of the selected backend type
    """
    backend = get_linalg_backend_type()
    if backend == "numpy":
        return NumpyMatrix(N, M, data, indices, indptr)
    elif backend == "scipy":
        return ScipyMatrix(N, M, data, indices, indptr)
    elif backend == "petsc":
        return PetscMatrix(N, N, data, indices, indptr)


def Vector(N: int) -> GenericVector:
    """
    Get a vector of the selected backend type
    """
    backend = get_linalg_backend_type()
    if backend == "petsc":
        return PetscVector(N)
    else:
        return NumpyVector(N)


def LinearSolver(
    solver: str | None = None, preconditioner: str | None = None
) -> GenericLinearSolver:
    """
    Get a linear equation solver of the selected backend type
    """
    backend = get_linalg_backend_type()
    if backend == "numpy":
        return NumpyLinearSolver(solver, preconditioner)
    elif backend == "scipy":
        return ScipyLinearSolver(solver, preconditioner)
    elif backend == "petsc":
        return PetscLinearSolver(solver, preconditioner)


class GenericMatrix:
    def __init__(self):
        raise NotImplementedError("You cannot instantiate a GenericMatrix")

    def finalize(self):
        pass

    def array(self) -> np.ndarray:
        raise NotImplementedError("You cannot instantiate a GenericMatrix")

    @property
    def csr_matrix(self) -> ArrayLike:
        raise NotImplementedError("You cannot instantiate a GenericMatrix")

    @property
    def csc_matrix(self) -> ArrayLike:
        raise NotImplementedError("You cannot instantiate a GenericMatrix")

    def __setitem__(self, key: int | slice, value: float | ArrayLike):
        raise NotImplementedError("You cannot instantiate a GenericMatrix")


class ScipyMatrix(GenericMatrix):
    def __init__(
        self,
        N: int,
        M: int,
        data: list[float] | None = None,
        indices: list[int] | None = None,
        indptr: list[int] | None = None,
    ):
        """
        A scipy.sparse CSR matrix
        """
        from scipy.sparse import csr_matrix

        self.shape = (N, M)

        if data is not None:
            self._csr = csr_matrix((data, indices, indptr), self.shape)
        else:
            self._csr = csr_matrix(self.shape)

    def array(self) -> np.ndarray:
        return self._csr.toarray()

    @property
    def csr_matrix(self):
        return self._csr

    @property
    def csc_matrix(self):
        return self._csr.tocsc()

    def __setitem__(self, key, value):
        """
        Set an item (with global dof indexes)
        """
        i, j = key
        self._csr[i, j] = value

    def __repr__(self, *args, **kwargs):
        return "<ScipyMatrix %d by %d>" % self.shape


class PetscMatrix(GenericMatrix):
    def __init__(
        self,
        N: int,
        M: int,
        data: list[float] | None = None,
        indices: list[int] | None = None,
        indptr: list[int] | None = None,
        nnz: int = 9,
    ):
        """
        A sparse matrix using the PETSc library through petsc4py

        Can be initialised through CSR data, indices and indptr
        (see documentation of scipy.sparse.csr_matrix)
        """
        self.shape = (N, M)

        if data is not None:
            self._mat = PETSc.Mat().createAIJ(size=self.shape, csr=(indptr, indices, data))
        else:
            self._mat = PETSc.Mat().createAIJ([N, M], nnz=nnz)

    @classmethod
    def from_csr(cls, data, indices, indptr, shape):
        N, M = shape

        return cls(N, M, data, indices, indptr)

    def array(self):
        return self._mat.getValues(range(self.shape[0]), range(self.shape[1]))

    def finalize(self):
        self._mat.assemblyBegin()
        self._mat.assemblyEnd()

    def __setitem__(self, key, value):
        """
        Set an item (with global dof indexes)
        """
        i, j = key
        self._mat.setValue(i, j, value)

    def __repr__(self, *args, **kwargs):
        return "<PetscMatrix %d by %d>" % self.shape


class NumpyMatrix(GenericMatrix):
    def __init__(
        self,
        N: int,
        M: int,
        data: list[float] | None = None,
        indices: list[int] | None = None,
        indptr: list[int] | None = None,
    ):
        """
        A dense matrix, fast construction and fast enough calculation for small problems
        """
        self.shape = (N, M)
        self._data = numpy.zeros(self.shape, dtype=float)

        if data is not None:
            assert indices is not None and indptr is not None
            for irow in range(N):
                for j in range(indptr[irow], indptr[irow + 1]):
                    self._data[irow, indices[j]] = data[j]

    def array(self):
        return self._data

    def __setitem__(self, key, value):
        """
        Set an item (with global dof indexes)
        """
        i, j = key
        self._data[i, j] = value

    def __repr__(self, *args, **kwargs):
        return "<NumpyMatrix %d by %d>" % self.shape


class GenericVector:
    def __init__(self):
        raise NotImplementedError("You cannot instantiate a GenericVector")

    def finalize(self):
        pass

    def __getitem__(self, key: int | slice) -> float | ArrayLike:
        raise NotImplementedError("You cannot instantiate a GenericVector")

    def __setitem__(self, key: int | slice, value: float | ArrayLike):
        raise NotImplementedError("You cannot instantiate a GenericVector")

    def __len__(self) -> int:
        raise NotImplementedError("You cannot instantiate a GenericVector")

    def array(self) -> np.ndarray:
        raise NotImplementedError("You cannot instantiate a GenericVector")


class NumpyVector(GenericVector):
    def __init__(self, N: int):
        self._data = numpy.zeros(N, dtype=float)

    def array(self) -> np.ndarray:
        return self._data

    def __getitem__(self, key: int | slice) -> float | ArrayLike:
        return self._data[key]

    def __setitem__(self, key: int | slice, value: float | ArrayLike):
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)


class PetscVector(GenericVector):
    def __init__(self, N: int):
        self._vec = PETSc.Vec().createSeq(N)

    def finalize(self):
        self._vec.assemblyBegin()
        self._vec.assemblyEnd()

    def __getitem__(self, key: int | slice) -> float | ArrayLike:
        return self._vec.getValue(key)

    def __setitem__(self, key: int | slice, value: float | ArrayLike):
        self._vec.setValue(key, value)

    def __len__(self):
        return self._vec.getSize()

    def array(self) -> np.ndarray:
        return self._vec.getArray()


class GenericLinearSolver:
    def __init__(self, solver=None, preconditioner=None):
        self.solver = solver
        self.preconditioner = preconditioner
        self.reuse_preconditioner = False

    def solve(self, A: GenericMatrix, u: GenericVector, b: GenericVector) -> int:
        raise NotImplementedError("You cannot instantiate a GenericLinearSolver")


class ScipyLinearSolver(GenericLinearSolver):
    def solve(self, A: GenericMatrix, u: GenericVector, b: GenericVector) -> int:
        """
        Solve A u = b using SciPy sparse

        A must be a Matrix, u and b must be Vectors
        """
        assert isinstance(A, ScipyMatrix)
        solver = self.solver
        solver = parameters["solver"] if solver is None else solver

        if solver == "default_direct":
            solver = "splu"

        tol = min(parameters["absolute_tolerance"], parameters["relative_tolerance"])

        if solver == "gmres":
            u[:], info = scipy.sparse.linalg.gmres(A.csr_matrix, b.array(), rtol=tol)
            assert info == 0, "Got scipy gmres error %d" % info
        elif solver == "minres":
            u[:], info = scipy.sparse.linalg.minres(A.csr_matrix, b.array(), rtol=tol)
            assert info == 0, "Got scipy minres error %d" % info
        elif solver == "bcgs":
            u[:], info = scipy.sparse.linalg.bicgstab(A.csr_matrix, b.array(), rtol=tol)
            assert info == 0, "Got scipy bicgstab error %d" % info
        elif solver == "spsolve":
            u[:] = scipy.sparse.linalg.spsolve(A.csr_matrix, b.array())
        elif solver == "splu":
            if not self.reuse_preconditioner or not hasattr(self, "lu"):
                self.lu = scipy.sparse.linalg.splu(A.csc_matrix)
            u[:] = self.lu.solve(b.array())
        else:
            raise HPCError("Unsupported SciPy solver %r" % solver)

        return 1


class PetscLinearSolver(GenericLinearSolver):
    def setup(self, A: GenericMatrix):
        """
        Setup the solver
        """
        assert isinstance(A, PetscMatrix)

        if hasattr(self, "ksp"):
            # Already set up
            return

        assert isinstance(A, PetscMatrix)
        solver = self.solver
        precon = self.preconditioner

        solver = parameters["solver"] if solver is None else solver
        precon = parameters["preconditioner"] if precon is None else precon
        petsc_options = {}

        if solver == "default_direct":
            solver = "mumps"

        # Direct solvers are implemented as preconditioners
        setup_pc = lambda pc: None
        if solver == "mumps":
            solver = "preonly"
            precon = "lu"
            setup_pc = lambda pc: pc.setFactorSolverPackage("mumps")

        # Some preconditioners like jacobi, bjacobi, sor, asm, ilu,
        # cholesky etc work right out of the box. For others we need to
        # do some setup. See i.e. cbc.block for examples of configuring
        # PETSc preconditioners through petsc4py
        if precon == "hypre_amg":
            # When using finite element discretisations boomerAMG works
            # very well for the Poisson equation, so we want to test this
            # for the HPC method as well
            precon = PETSc.PC.Type.HYPRE
            petsc_options["pc_hypre_type"] = "boomeramg"

        with PetscOptions(petsc_options):
            ksp = PETSc.KSP().create()
            ksp.setType(solver)
            ksp.setTolerances(
                parameters["relative_tolerance"],
                parameters["absolute_tolerance"],
                parameters["divergence_limit"],
                parameters["max_iterations"],
            )

            pc = ksp.getPC()
            pc.setType(precon)
            setup_pc(pc)
            pc.setFromOptions()
            pc.setReusePreconditioner(self.reuse_preconditioner)

        self.ksp = ksp
        self.pc = pc

    def solve(self, A: GenericMatrix, u: GenericVector, b: GenericVector) -> int:
        """
        Solve A u = b using PETSc

        A must be a Matrix, u and b must be Vectors
        """
        assert isinstance(A, PetscMatrix)
        assert isinstance(u, PetscVector)
        assert isinstance(b, PetscVector)

        if self.solver == "hpc_richardson":
            return hpc_richardson(A, u, b)

        self.setup(A)
        assert isinstance(u, PetscVector)
        assert isinstance(b, PetscVector)

        # Finalize matrix and vector
        A.finalize()
        b.finalize()

        # Solve the linear system
        ksp, pc = self.ksp, self.pc
        ksp.setOperators(A._mat)
        pc.setUp()
        ksp.solve(b._vec, u._vec)

        # Check that the solver converged
        conv_code = ksp.getConvergedReason()
        if not conv_code > 0:
            raise PetscError(conv_code=conv_code)

        return ksp.getIterationNumber()


class NumpyLinearSolver(GenericLinearSolver):
    def solve(self, A: GenericMatrix, u: GenericVector, b: GenericVector) -> int:
        """
        Solve A u = b using numpy dense matrices

        A must be a Matrix, u and b must be Vectors
        """
        u[:] = numpy.linalg.solve(A.array(), b.array())
        return 1


def solve(A: GenericMatrix, u: GenericVector, b: GenericVector, *args):
    """
    Solve A u = b

    A must be a Matrix, u and b must be Vectors
    """
    return LinearSolver().solve(A, u, b)


class PetscOptions(object):
    def __init__(self, options):
        """
        PETSc options are global. This context manager handles
        setting and resetting options to avoid clobbering the
        global option database with non-default values
        """
        self.options = options

    def __enter__(self):
        if self.options:
            self.orig_options = PETSc.Options().getAll()
            for key, value in self.options.items():
                PETSc.Options().setValue(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.options:
            for key in self.options.iterkeys():
                PETSc.Options().delValue(key)
            for key, value in self.orig_options.iteritems():
                PETSc.Options().setValue(key, value)


class PetscError(Exception):
    def __init__(self, message=None, conv_code=None):
        if conv_code is not None:
            conv_reason = get_petsc_convergence_reason(conv_code)
            if conv_reason is None:
                conv_reason = "UNKNOWN REASON!"
            message = "KSP status %s" % conv_reason
        Exception.__init__(self, message)


def get_petsc_convergence_reason(conv_code):
    """
    Translate PETSc's numerical convergence codes to strings
    """
    for attr in dir(PETSc.KSP.ConvergedReason):
        if attr.startswith("DIVERGED") or attr.startswith("CONVERGED"):
            val = getattr(PETSc.KSP.ConvergedReason, attr)
            if val == conv_code:
                return attr


def hpc_richardson(
    A: PetscMatrix, u: PetscVector, b: PetscVector, tol: float = 1e-8, maxiter: int = 1000
):
    """
    An extremely basic implementation of Richardson iterations for solving
    Au=b for PETSc matrices. Used for debugging only
    """
    assert isinstance(A, PetscMatrix)
    assert isinstance(u, PetscVector)
    assert isinstance(b, PetscVector)

    r = PetscVector(len(u))

    A = A._mat
    u = u._vec
    b = b._vec
    r = r._vec

    for i in range(maxiter):
        A.mult(u, r)
        r.axpy(-1.0, b)
        # print i, r.array
        u.axpy(-1.0, r)
        norm = numpy.linalg.norm(r.array)
        if norm < tol:
            return i + 1
    else:
        raise HPCError("HPC richardson iteration did not converge. Norm = %r" % norm)
