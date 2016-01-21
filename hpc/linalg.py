import numpy


DENSE_LIMIT = 1000


def get_linalg_backend_type(N):
    try:
        import scipy.sparse
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    backend = 'numpy' if (N <= DENSE_LIMIT or not has_scipy) else 'scipy'
    return backend


def Matrix(N, M):
    """
    Try to make a smart selection of the matrix type to use
    """
    backend = get_linalg_backend_type(N)
    if backend == 'numpy':
        return NumpyMatrix(N, M)
    elif backend == 'scipy':
        return ScipyMatrix(N, M)


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
    
    def insert_matrix(self, global_indices_cols, global_indices_rows, matrix):
        """
        Insert a submatrix into this matrix
        """
        N, M = matrix.shape
        assert N == len(global_indices_cols)
        assert M == len(global_indices_rows)
        
        for i in range(N):
            gi = global_indices_cols[i]
            for j in range(M):
                gj = global_indices_rows[j]
                self._lil[gi,gj] += matrix[i,j]
        
        # Invalidate cached CSR matrix
        self._csr = None
    
    def set_row_to_zero(self, i):
        """
        Set the row with the given index to be all zeros.
        Does not change the sparsity pattern
        """
        row = self._lil.data[i]
        row[:] = [0]*len(row)
        
        # Invalidate cached CSR matrix
        self._csr = None
    
    def set_col_to_zero(self, i):
        """
        Set the column with the given index to be all zeros.
        Does not change the sparsity pattern
        Returns the column as it was before deletion as a 1D array
        """
        N = self.shape[0]
        
        col = numpy.zeros(N, float)
        for irow, coljs in enumerate(self._lil.rows):
            for idx, icol in enumerate(coljs):
                if icol == i:
                    col[irow] = self._lil.data[irow][idx]
                    self._lil.data[irow][idx] = 0
                if icol >= i:
                    break
        
        # Invalidate cached CSR matrix
        self._csr = None
        
        return col
    
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


class NumpyMatrix(GenericMatrix):
    def __init__(self, N, M):
        """
        A dense matrix, fast construction and fast enough calculation for small problems
        """
        self.shape = (N, M)
        self._data = numpy.zeros(self.shape, dtype=float)
    
    def array(self):
        return self._data
    
    def insert_matrix(self, global_indices_cols, global_indices_rows, matrix):
        """
        Insert a submatrix into this matrix
        """
        N, M = matrix.shape
        assert N == len(global_indices_cols)
        assert M == len(global_indices_rows)
        
        for i in range(N):
            gi = global_indices_cols[i]
            for j in range(M):
                gj = global_indices_rows[j]
                self._data[gi,gj] += matrix[i,j]
    
    def set_row_to_zero(self, i):
        """
        Set the row with the given index to be all zeros.
        """
        self._data[i,:] = 0.0
    
    def set_col_to_zero(self, i):
        """
        Set the column with the given index to be all zeros.
        Returns the column as it was before deletion as a 1D array
        """
        # Copy previous data
        N = self.shape[0]
        col = numpy.array(self._data[:,i])
        col.shape = (N,)
        
        self._data[:,i] = 0.0
        return col
    
    def __setitem__(self, key, value):
        """
        Set an item (with global dof indexes)
        """
        i, j = key
        self._data[i,j] = value
    
    def __repr__(self, *args, **kwargs):
        return '<NumpyMatrix %d by %d>' % self.shape


class Vector(numpy.ndarray, GenericVector):
    def __init__(self, N):
        numpy.ndarray.__init__(self, N)
        self[:] = 0
    
    def array(self):
        return self[:]


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
        if solver is None:
            solver = 'spsolve' if isinstance(A, ScipyMatrix) else 'numpy'
        
        if solver == 'spsolve':
            import scipy.sparse.linalg
            u[:] = scipy.sparse.linalg.spsolve(A.csr_matrix, b.array())
        elif solver == 'numpy':
            import numpy.linalg
            u[:] = numpy.linalg.solve(A.array(), b.array())


def solve(A, u, b, *args):
    """
    Solve A u = b
    
    A must be a Matrix, u and b must be Vectors
    """
    LinearSolver().solve(A, u, b)
