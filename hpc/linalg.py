from hpc import parameters
import numpy.linalg

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
except:
    has_petsc = False


DENSE_LIMIT = 1000


def get_linalg_backend_type(N):
    backend = parameters['linear_algebra_backend'] 
    if backend != 'auto':
        assert backend in ('scipy', 'petsc', 'numpy')
        return backend 
    
    if has_petsc:
        sparse_backend = 'petsc'
    elif has_scipy:
        sparse_backend = 'scipy'
    else:
        sparse_backend = None
    
    if N <= DENSE_LIMIT or not sparse_backend:
        return 'numpy'
    else:
        return sparse_backend


def Matrix(N, M):
    """
    Try to make a smart selection of the matrix type to use
    """
    backend = get_linalg_backend_type(N)
    if backend == 'numpy':
        return NumpyMatrix(N, M)
    elif backend == 'scipy':
        return ScipyMatrix(N, M)
    elif backend == 'petsc':
        return PetscMatrix(N, N)


def Vector(N):
    """
    Try to make a smart selection of the matrix type to use
    """
    backend = get_linalg_backend_type(N)
    if backend == 'petsc':
        return PetscVector(N)
    else:
        return NumpyVector(N)


class GenericMatrix(object):
    def __init__(self):
        raise NotImplementedError('You cannot instantiate a GenericMatrix')
    

class GenericVector(object):
    def __init__(self):
        raise NotImplementedError('You cannot instantiate a GenericVector')


class ScipyMatrix(GenericMatrix):
    def __init__(self, N, M):
        """
        A sparse matrix (uses LIL for construction and CSR for calculations)
        """
        from scipy.sparse import lil_matrix
        self.shape = (N, M)
        self._lil = lil_matrix(self.shape)
        self._csr = None
    
    def array(self):
        return self._lil.toarray()
    
    @property
    def csr_matrix(self):
        if self._csr is None:
            self._csr = self._lil.tocsr()
        return self._csr
    
    def __setitem__(self, key, value):
        """
        Set an item (with global dof indexes)
        """
        i, j = key
        self._lil[i,j] = value
        
        # Invalidate cached CSR matrix
        self._csr = None
    
    def __repr__(self, *args, **kwargs):
        return '<ScipyMatrix %d by %d>' % self.shape


class PetscMatrix(GenericMatrix):
    def __init__(self, N, M):
        """
        A sparse matrix using the PETSc library through petsc4py
        """
        self.shape = (N, M)
        self._mat = PETSc.Mat().createAIJ([N, M], nnz=9)
    
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
        return '<PetscMatrix %d by %d>' % self.shape


class NumpyMatrix(GenericMatrix):
    def __init__(self, N, M):
        """
        A dense matrix, fast construction and fast enough calculation for small problems
        """
        self.shape = (N, M)
        self._data = numpy.zeros(self.shape, dtype=float)
    
    def array(self):
        return self._data
    
    def __setitem__(self, key, value):
        """
        Set an item (with global dof indexes)
        """
        i, j = key
        self._data[i,j] = value
    
    def __repr__(self, *args, **kwargs):
        return '<NumpyMatrix %d by %d>' % self.shape


class NumpyVector(numpy.ndarray, GenericVector):
    def __init__(self, N):
        numpy.ndarray.__init__(self, N)
        self[:] = 0
    
    def array(self):
        return self[:]


class PetscVector(GenericVector):
    def __init__(self, N):
        self._vec = PETSc.Vec().createSeq(N)
    
    def __setitem__(self, key, value):
        self._vec.setValue(key, value)
    
    def __len__(self):
        return self._vec.getSize()
    
    def array(self):
        return self._vec.getArray()


class LinearSolver(object):
    def __init__(self, solver=None, preconditioner=None):
        self.solver = solver
        self.preconditioner = None
    
    def solve(self, A, u, b):
        """
        Solve A u = b
        
        A must be a Matrix, u and b must be Vectors
        """
        solver = self.solver
        precon = self.preconditioner
        
        # Solve with scipy
        if isinstance(A, ScipyMatrix):
            solver = 'spsolve' if solver is None else solver
            u[:] = scipy.sparse.linalg.spsolve(A.csr_matrix, b.array())
            return 1
        
        # Solve with petsc4py
        elif isinstance(A, PetscMatrix):
            assert isinstance(u, PetscVector)
            assert isinstance(b, PetscVector)
            solver = parameters['solver'] if solver is None else solver
            precon = parameters['preconditioner'] if precon is None else precon
            
            # Finalize matrix
            A.finalize()
            
            ksp = PETSc.KSP().create()
            ksp.setOperators(A._mat)
            ksp.setType(solver)
            
            pc = ksp.getPC()
            pc.setType(precon)
            
            ksp.setTolerances(parameters['relative_tolerance'],
                              parameters['absolute_tolerance'],
                              parameters['divergence_limit'],
                              parameters['max_iterations'])
            ksp.solve(b._vec, u._vec)
            
            return ksp.getIterationNumber()
        
        # Solve with numpy
        elif isinstance(A, NumpyMatrix):
            u[:] = numpy.linalg.solve(A.array(), b.array())
            return 1


def solve(A, u, b, *args):
    """
    Solve A u = b
    
    A must be a Matrix, u and b must be Vectors
    """
    return LinearSolver().solve(A, u, b)
