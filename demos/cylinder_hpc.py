from __future__ import division
import numpy
import hpc

class PotentialFlowDomain(object):
    def __init__(self, inp):
        self.input = inp
        
        assert inp.layout == 'I'
        p0 = (-inp.l1, -inp.h1/2)
        p1 = (0, inp.h1/2)
        hpc.parameters['linear_algebra_backend'] = 'scipy'
        self.domain = hpc.rectangle_domain(p0, p1, inp.N, inp.N)
        self.bcs = self._get_boundary_conditions()
        self.phi = numpy.zeros(len(self.domain.dof_coordinates), float)
    
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
            if coord[0] == 0:
                dividing_line.append((dof, coord))
        
        dividing_line.sort(key=lambda item: item[1][0]+item[1][1])
        return dividing_line
    
    def get_neumann_weights(self, component, coord, dof0, dof1):
        assert component in (0, 1)
        domain = self.domain
        coord0 = domain.dof_coordinates[dof0]
        coord1 = domain.dof_coordinates[dof1]
        
        d0 = (coord0[0] - coord[0])**2 + (coord0[1] - coord[1])**2
        d1 = (coord1[0] - coord[0])**2 + (coord1[1] - coord[1])**2
        
        dof = dof0 if d0 < d1 else dof1
        neighbours, _coeffs, coeffs_diffx, coeffs_diffy = hpc.eval_phi(domain, dof)
        
        if component == 0:
            return neighbours, coeffs_diffx
        else:
            return neighbours, coeffs_diffy
    
    def assemble_csr(self):
        A, b = hpc.assemble(self.domain)
        hpc.apply_bcs(self.domain, A, b, self.bcs)
        return A.csr_matrix, b.array()
    
    def apply_dirichlet(self, A, dof):
        """
        Set row indicated by dof to zero except A[dof,dof] which should be 1
        """
        for nb in self.domain.dof_neighbours[dof]:
            A[dof,nb] = 0.0
        A[dof,dof] = 1.0
    
    def explicit_velocity_at_dof(self, dof):
        neighbours, _coeffs, coeffsdx, coeffsdy = hpc.eval_phi(self.domain, dof)
        u = v = 0
        for nb, wu, wv in zip(neighbours, coeffsdx, coeffsdy):
            u += wu*self.phi[nb]
            v += wv*self.phi[nb]
        return u, v
    
    def update(self, phi):
        self.phi[:] = phi
    
    def _get_boundary_conditions(self):
        bcs = []
        return bcs
