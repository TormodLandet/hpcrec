from hpc import parameters, HPCError
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


def get_linalg_backend_type():
    available_backends = ['numpy']
    if has_scipy: available_backends.append('scipy')
    if has_petsc: available_backends.append('petsc') 
    
    param_backend = parameters['linear_algebra_backend'] 
    if param_backend != 'auto':
        if not param_backend in available_backends:
            raise HPCError('Linear algebra backend %r not available. Select one of %s.' % \
                           (param_backend, ', '.join(repr(b) for b in available_backends)))
        return param_backend 
    
    if has_petsc:
        return 'petsc'
    elif has_scipy:
        return 'scipy'
    else:
        return 'numpy'


def Matrix(N, M):
    """
    Get a matrix of the selected backend type
    """
    backend = get_linalg_backend_type()
    if backend == 'numpy':
        return NumpyMatrix(N, M)
    elif backend == 'scipy':
        return ScipyMatrix(N, M)
    elif backend == 'petsc':
        return PetscMatrix(N, N)


def Vector(N):
    """
    Get a vector of the selected backend type 
    """
    backend = get_linalg_backend_type()
    if backend == 'petsc':
        return PetscVector(N)
    else:
        return NumpyVector(N)


def LinearSolver(solver=None, preconditioner=None):
    """
    Get a linear equation solver of the selected backend type
    """
    backend = get_linalg_backend_type()
    if backend == 'numpy':
        return  NumpyLinearSolver(solver, preconditioner)
    elif backend == 'scipy':
        return ScipyLinearSolver(solver, preconditioner)
    elif backend == 'petsc':
        return PetscLinearSolver(solver, preconditioner)


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
        self._csr = self._csc = None
    
    def array(self):
        return self._lil.toarray()
    
    @property
    def csr_matrix(self):
        if self._csr is None:
            self._csr = self._lil.tocsr()
        return self._csr
    
    @property
    def csc_matrix(self):
        if self._csc is None:
            self._csc = self._lil.tocsc()
        return self._csc
    
    def __setitem__(self, key, value):
        """
        Set an item (with global dof indexes)
        """
        i, j = key
        self._lil[i,j] = value
        
        # Invalidate cached matrices
        self._csr = self._csc = None
    
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
    
    def finalize(self):
        self._vec.assemblyBegin()
        self._vec.assemblyEnd()
    
    def __getitem__(self, key):
        return self._vec.getValue(key)
    
    def __setitem__(self, key, value):
        self._vec.setValue(key, value)
    
    def __len__(self):
        return self._vec.getSize()
    
    def array(self):
        return self._vec.getArray()


class GenericLinearSolver(object):
    def __init__(self, solver=None, preconditioner=None):
        self.solver = solver
        self.preconditioner = None
        self.reuse_preconditioner = False


class ScipyLinearSolver(GenericLinearSolver):
    def solve(self, A, u, b):
        """
        Solve A u = b using SciPy sparse
        
        A must be a Matrix, u and b must be Vectors
        """
        assert isinstance(A, ScipyMatrix)
        solver = self.solver
        solver = parameters['solver'] if solver is None else solver
        
        if solver == 'default_direct':
            solver = 'splu'
        
        tol = min(parameters['absolute_tolerance'], parameters['relative_tolerance'])
        
        if solver == 'gmres':
            u[:], info = scipy.sparse.linalg.gmres(A.csr_matrix, b.array(), tol=tol)
            assert info == 0, 'Got scipy gmres error %d' % info
        elif solver == 'minres':
            u[:], info = scipy.sparse.linalg.minres(A.csr_matrix, b.array(), tol=tol)
            assert info == 0, 'Got scipy minres error %d' % info
        elif solver == 'bcgs':
            u[:], info = scipy.sparse.linalg.bicgstab(A.csr_matrix, b.array(), tol=tol)
            assert info == 0, 'Got scipy bicgstab error %d' % info
        elif solver == 'spsolve':
            u[:] = scipy.sparse.linalg.spsolve(A.csr_matrix, b.array())
        elif solver == 'splu':
            if not self.reuse_preconditioner or not hasattr(self, 'lu'):
                self.lu = scipy.sparse.linalg.splu(A.csc_matrix)
            u[:] = self.lu.solve(b)
        else:
            raise HPCError('Unsupported SciPy solver %r' % solver)
        
        return 1


class PetscLinearSolver(GenericLinearSolver):
    def setup(self, A):
        """
        Setup the solver
        """
        if hasattr(self, 'ksp'):
            # Already set up
            return
        
        assert isinstance(A, PetscMatrix)
        solver = self.solver
        precon = self.preconditioner
        
        solver = parameters['solver'] if solver is None else solver
        precon = parameters['preconditioner'] if precon is None else precon
        petsc_options = {}
        
        if solver == 'default_direct':
            solver = 'mumps'
        
        # Direct solvers are implemented as preconditioners
        setup_pc = lambda pc: None
        if solver == 'mumps':
            solver = 'preonly'
            precon = 'lu'
            setup_pc = lambda pc: pc.setFactorSolverPackage('mumps')
            
        # Some preconditioners like jacobi, bjacobi, sor, asm, ilu, 
        # cholesky etc work right out of the box. For others we need to
        # do some setup. See i.e. cbc.block for examples of configuring
        # PETSc preconditioners through petsc4py
        if precon == 'hypre_amg':
            # When using finite element discretisations boomerAMG works 
            # very well for the Poisson equation, so we want to test this
            # for the HPC method as well
            precon = PETSc.PC.Type.HYPRE
            petsc_options['pc_hypre_type']  = 'boomeramg'
        
        with PetscOptions(petsc_options):
            ksp = PETSc.KSP().create()
            ksp.setType(solver)
            ksp.setTolerances(parameters['relative_tolerance'],
                              parameters['absolute_tolerance'],
                              parameters['divergence_limit'],
                              parameters['max_iterations'])
            
            pc = ksp.getPC()
            pc.setType(precon)
            setup_pc(pc)
            pc.setFromOptions()
            pc.setReusePreconditioner(self.reuse_preconditioner)
        
        self.ksp = ksp
        self.pc = pc
    
    def solve(self, A, u, b):
        """
        Solve A u = b using PETSc
        
        A must be a Matrix, u and b must be Vectors
        """
        if self.solver == 'hpc_richardson':
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
    def solve(self, A, u, b):
        """
        Solve A u = b using numpy dense matrices
        
        A must be a Matrix, u and b must be Vectors
        """
        u[:] = numpy.linalg.solve(A.array(), b.array())
        return 1


def solve(A, u, b, *args):
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
            for key, value in self.options.iteritems():
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
                conv_reason = 'UNKNOWN REASON!'
            message = 'KSP status %s' % conv_reason
        Exception.__init__(self, message)


def get_petsc_convergence_reason(conv_code):
    """
    Translate PETSc's numerical convergence codes to strings
    """
    for attr in dir(PETSc.KSP.ConvergedReason):
        if attr.startswith('DIVERGED') or attr.startswith('CONVERGED'):
            val = getattr(PETSc.KSP.ConvergedReason, attr)
            if val == conv_code:
                return attr


def hpc_richardson(A, u, b, tol=1e-8, maxiter=1000):
    """
    An extremely basic implementation of Richardson iterations for solving
    Au=b for PETSc matrices. Used for debugging only
    """
    assert isinstance(A, PetscMatrix)
    r = PetscVector(len(u))
    
    A = A._mat
    u = u._vec
    b = b._vec
    r = r._vec
    
    for i in range(maxiter):
        A.mult(u, r)
        r.axpy(-1.0, b)
        #print i, r.array
        u.axpy(-1.0, r)
        norm = numpy.linalg.norm(r.array)
        if norm < tol:
            return i+1
    else:
        raise HPCError('HPC richardson iteration did not converge. Norm = %r' % norm)
