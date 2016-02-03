from __future__ import division
import numpy
import hpc

class PotentialFlowDomain(object):
    def __init__(self, inp):
        assert inp.layout == 'I'
        hpc.parameters['linear_algebra_backend'] = 'scipy'
        self.input = inp
        
        p0 = (-inp.l1, -inp.h1/2)
        p1 = (0, inp.h1/2)
        Ny = self.input.N1
        Nx = int(round(Ny*self.input.l1/self.input.h1))
        
        self.domain = hpc.rectangle_domain(p0, p1, Nx, Ny)
        self.phi = numpy.zeros(len(self.domain.dof_coordinates), float)
        self.phi_old = numpy.zeros_like(self.phi)
    
    def get_dividing_line(self, line_number):
        """
        Get the dofs on the given line between the Navier-Stokes and potential 
        flow domains
        
          +---------------------------------
          |          
          |     +---line 1------------------
          |     |     
          |     |<--line 0
          |     |
          |     +---line 2------------------
          | 
          +---------------------------------
        
        The coordinate system is such that x = 0 on line 0 and y=+/- h2/2 on
        lines 1 and 2
        """
        assert line_number == 0
        
        dividing_line = []
        for dof, coord in enumerate(self.domain.dof_coordinates):
            if coord[0] > -1e-8:
                dividing_line.append((dof, coord))
        
        dividing_line.sort(key=lambda item: item[1][0]+item[1][1])
        return dividing_line
    
    def get_neumann_weights(self, component, coord, dof0, dof1):
        assert component in (0, 1)
        domain = self.domain
        coord0 = domain.dof_coordinates[dof0]
        coord1 = domain.dof_coordinates[dof1]
        
        d0 = ((coord0[0] - coord[0])**2 + (coord0[1] - coord[1])**2)**0.5
        d1 = ((coord1[0] - coord[0])**2 + (coord1[1] - coord[1])**2)**0.5
        fac = d1/(d0+d1)
        
        nbs0, _, cx0, cy0 = hpc.eval_phi(domain, dof0)
        nbs1, _, cx1, cy1 = hpc.eval_phi(domain, dof1)
        
        dofs = numpy.zeros(16, int)
        coeffs = numpy.zeros(16, float)
        cs = (cx0, cx1) if component == 0 else (cy0, cy1)
        dofs[:8] = nbs0
        dofs[8:] = nbs1
        coeffs[:8] = fac*cs[0]
        coeffs[8:] = (1-fac)*cs[1]
        
        #print 'u%d' % component, coord
        return dofs, coeffs
    
    def get_system(self, t):
        """
        Return linear system with normal BCs applied (not coupled)
        Matrix format is SciPy CSR
        """
        A, b = hpc.assemble(self.domain)
        bcs = self._get_boundary_conditions(t)
        hpc.apply_bcs(self.domain, A, b, bcs)
        return A.csr_matrix, b.array()
    
    def explicit_velocity_at_dof(self, dof):
        neighbours, _coeffs, coeffsdx, coeffsdy = hpc.eval_phi(self.domain, dof)
        u = v = 0
        for nb, wu, wv in zip(neighbours, coeffsdx, coeffsdy):
            u += wu*self.phi[nb]
            v += wv*self.phi[nb]
        return u, v
    
    def update(self, phi):
        self.phi_old[:] = self.phi
        self.phi[:] = phi
    
    def get_triangulation(self):
        coords = self.domain.dof_coordinates
        triangles = self.domain.triangles
        return coords, triangles
    
    def get_data(self, func_name):
        N = len(self.domain.dof_coordinates)
        values = numpy.zeros(N, float)
        rho = self.input.rho
        dt = self.input.dt
        for dof in range(N):
            vel = self.explicit_velocity_at_dof(dof)
            if func_name == 'u0':
                values[dof] = vel[0]
            elif func_name == 'u1':
                values[dof] = vel[1]
            elif func_name == 'p':
                values[dof] = -(vel[0]**2 + vel[1]**2)/2 - (self.phi[dof] - self.phi_old[dof])/dt
        
        if func_name == 'p':
            values *= rho
        
        return values
    
    def _get_boundary_conditions(self, t):
        inp, domain = self.input, self.domain
        U = inp.inlet_vel(t)
        
        bcs = []
        for dof, coord in enumerate(domain.dof_coordinates): 
            if domain.dof_type[dof] == hpc.DOF_TYPE_EXTERNAL:
                x, y = coord
                if x > -1e-8:
                    pass # Coupled to N-S
                elif x < -inp.l1 + 1e-8:
                    bcs.append(('Nx', dof, U))
                elif y > inp.h1/2 - 1e-8:
                    bcs.append(('Ny', dof, 0.0))
                elif y < -inp.h1/2 + 1e-8:
                    bcs.append(('Ny', dof, 0.0))
                else:
                    raise 'this should not happen!'
        
        return bcs
