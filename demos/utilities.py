# encoding: utf8
import sys, time, contextlib
import numpy
import scipy.sparse
import dolfin as df


RED = '\033[91m%s\033[0m'    # ANSI escape code Bright Red
YELLOW = '\033[91m%s\033[0m' # ANSI escape code Bright Yellow


class SimpleLog(object):
    def __init__(self, logfile=None, console=True):
        """
        SimpleLog lets you print messages to a file and to the 
        console at the same time. Errors will be written to
        console in red and warnings in yellow. The text "ERROR: "
        and "WARNING: " will be prepended to each line of the
        respective type of messages. 
        """
        self.files = []
        if logfile:
            self.files.append(open(logfile, 'wt'))
        if console:
            self.files.append(sys.stdout)
            
    def info(self, message):
        for f in self.files:
            f.write(message)
    
    def warning(self, message):
        message_warn = message[:-1].replace('\n', '\nWARNING: ')
        message = 'WARNING: %s%s' % (message_warn, message[-1])
        for f in self.files:
            if f.fileno() == 1:
                f.write(YELLOW % message)
            else:
                f.write(message)
    
    def error(self, message):
        message_err = message[:-1].replace('\n', '\nERROR: ')
        message = 'ERROR: %s%s' % (message_err, message[-1])
        for f in self.files:
            if f.fileno() == 1:
                f.write(RED % message)
            else:
                f.write(message)
    
    @contextlib.contextmanager
    def timer(self, pre_message='Timer: ', post_message='%4.2fs'):
        self.info(pre_message)
        t_start = time.time()
        yield
        duration = time.time() - t_start
        self.info(post_message % duration)


def mat_to_csr(dolfin_matrix):
    """
    Convert any dolfin.Matrix to csr matrix in scipy.
    Based on code by Miroslav Kuchta
    """
    assert df.MPI.size(df.mpi_comm_world()) == 1, 'mat_to_csr assumes single process'
    
    rows = [0]
    cols = []
    values = []
    for irow in range(dolfin_matrix.size(0)):
        indices, values_ = dolfin_matrix.getrow(irow)
        rows.append(len(indices)+rows[-1])
        cols.extend(indices)
        values.extend(values_)

    shape = dolfin_matrix.size(0), dolfin_matrix.size(1)
        
    return scipy.sparse.csr_matrix((numpy.array(values, dtype='float'),
                                    numpy.array(cols, dtype='int'),
                                    numpy.array(rows, dtype='int')),
                                    shape)


class StreamFunction(object):
    def __init__(self, u, boundary_is_streamline=False, degree=1):
        """
        Heavily based on
        https://github.com/mikaem/fenicstools/blob/master/fenicstools/Streamfunctions.py
        
        Stream function for a given general 2D velocity field.
        The boundary conditions are weakly imposed through the term
        
            inner(q, grad(psi)*n)*ds, 
        
        where grad(psi) = [-v, u] is set on all boundaries. 
        This should work for any collection of boundaries: 
        walls, inlets, outlets etc.    
        """
        Vu = u[0].function_space()
        mesh = Vu.mesh()
        
        # Check dimension
        if not mesh.geometry().dim() == 2:
            df.error("Stream-function can only be computed in 2D.")
    
        # Define the weak form 
        V = df.FunctionSpace(mesh, 'CG', degree)
        q = df.TestFunction(V)
        psi = df.TrialFunction(V)
        n = df.FacetNormal(mesh)
        a = df.dot(df.grad(q), df.grad(psi))*df.dx
        L = df.dot(q, df.curl(u))*df.dx 
        
        if boundary_is_streamline: 
            # Strongly set psi = 0 on entire domain boundary
            self.bcs = [df.DirichletBC(V, df.Constant(0), df.DomainBoundary())]
            self.normalize = False
        else:
            self.bcs = []
            self.normalize = True
            L = L + q*(n[1]*u[0] - n[0]*u[1])*df.ds
            
        # Create preconditioned iterative solver
        solver = df.PETScKrylovSolver('gmres', 'hypre_amg')
        solver.parameters['nonzero_initial_guess'] = True
        solver.parameters['relative_tolerance'] = 1e-10
        solver.parameters['absolute_tolerance'] = 1e-10
        solver.parameters['preconditioner']['structure'] = 'same'
        
        # Store for later computation
        self.psi = df.Function(V)
        self.A = df.assemble(a)
        self.L = L
        self.mesh = mesh
        self.solver = solver
        self._triangulation = None
    
    def compute(self):
        """
        Compute the stream function
        """
        b = df.assemble(self.L)
        
        if self.normalize:
            df.normalize(b)
        
        for bc in self.bcs:
            bc.apply(self.A, b)
        self.solver.solve(self.A, self.psi.vector(), b)
        
        if self.normalize: 
            df.normalize(self.psi.vector())
    
        return self.psi
    
    def plot(self, mpl_ax, levels=50):
        """
        Plot the function on a matplotlib axes. Call .compute() first
        to calculate the stream function
        """
        if self._triangulation is None:
            from matplotlib.tri import Triangulation
            coords = self.mesh.coordinates()
            triangles = []
            for cell in df.cells(self.mesh):
                cell_vertices = cell.entities(0)
                triangles.append(cell_vertices)
            self._triangulation = Triangulation(coords[:,0], coords[:,1], triangles)
        
        Z = self.psi.compute_vertex_values()
        mpl_ax.tricontour(self._triangulation, Z, levels, colors='#0000AA',
                          linewidths=0.3, linestyles='solid')
