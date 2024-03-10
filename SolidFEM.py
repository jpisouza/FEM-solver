import numpy as np
from timeit import default_timer as timer
import scipy as sp
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, dok_matrix
import scipy.linalg  
import scipy.sparse.linalg 
import Elements

class FEM:
    
    @classmethod
    def set_parameters(cls, mesh, BC, h, E, dt, rho, nu):
        
        cls.E = E
        cls.rho = rho
        cls.mesh = mesh
        cls.nu = nu
        cls.BC = BC
        cls.h = h
        cls.dt = dt
        
        cls.D = (E/(1.0-nu**2))*np.array([[1.0,   nu,  0.0],
                                     [nu,  1.0,   0.0],
                                     [0.0,   0.0,   (1.0-nu)/2.0]]);
        
        cls.forces = np.zeros((2*cls.mesh.npoints), dtype='float')
        cls.u = np.zeros((2*cls.mesh.npoints), dtype='float')
        cls.u_prime = np.zeros((2*cls.mesh.npoints), dtype='float')
        
        cls.u_minus = cls.u - cls.u_prime*cls.dt
        
        cls.X_orig = cls.mesh.X.copy()
        cls.Y_orig = cls.mesh.Y.copy()

    @classmethod   
    def build_quad_GQ(cls):
        cls.K = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        cls.M = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        cls.Mb = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        
        quad = Elements.Quad(cls.mesh.X,cls.mesh.Y)
        
        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4,v5,v6 = cls.mesh.IEN[e]
            quad.getMSolid([v1,v2,v3,v4,v5,v6],cls.D)
            k_elem = quad.k
            m_elem = quad.mass
            for ilocal in range(0,6):
                  iglobal_x = 2*cls.mesh.IEN[e,ilocal]
                  iglobal_y = 2*cls.mesh.IEN[e,ilocal]+1
                  for jlocal in range(0,6):
                      jglobal_x = 2*cls.mesh.IEN[e,jlocal]
                      jglobal_y = 2*cls.mesh.IEN[e,jlocal]+1
            
                      cls.K[iglobal_x,jglobal_x] = cls.K[iglobal_x,jglobal_x] + k_elem[2*ilocal,2*jlocal]
                      cls.K[iglobal_x,jglobal_y] = cls.K[iglobal_x,jglobal_y] + k_elem[2*ilocal,2*jlocal+1]
                      cls.K[iglobal_y,jglobal_y] = cls.K[iglobal_y,jglobal_y] + k_elem[2*ilocal+1,2*jlocal+1]
                      cls.K[iglobal_y,jglobal_x] = cls.K[iglobal_y,jglobal_x] + k_elem[2*ilocal+1,2*jlocal]
                      
                      cls.M[iglobal_x,jglobal_x] = cls.M[iglobal_x,jglobal_x] + m_elem[2*ilocal,2*jlocal]
                      cls.M[iglobal_x,jglobal_y] = cls.M[iglobal_x,jglobal_y] + m_elem[2*ilocal,2*jlocal+1]
                      cls.M[iglobal_y,jglobal_y] = cls.M[iglobal_y,jglobal_y] + m_elem[2*ilocal+1,2*jlocal+1]
                      cls.M[iglobal_y,jglobal_x] = cls.M[iglobal_y,jglobal_x] + m_elem[2*ilocal+1,2*jlocal]
        
        for e in range(0,len(cls.mesh.IENbound)):
            for bound in cls.BC:
                if (cls.BC[bound][2] != 'None' or cls.BC[bound][3] != 'None')  and all(np.in1d(cls.mesh.IENbound[e,:],cls.mesh.bound_dict[bound])):
                    v1,v2,v3 = cls.mesh.IENbound[e]
                    
                    L = ((cls.mesh.X[v1] - cls.mesh.X[v2])**2 + (cls.mesh.Y[v1] - cls.mesh.Y[v2])**2)**0.5;
                    
                    Mbe = (L/30)* np.array([[4, 0, 2, 0, -1, 0],
                                               [0, 4, 0, 2, 0, -1],
                                               [2, 0, 16, 0, 2, 0],
                                               [0, 2, 0, 16, 0, 2],
                                               [-1, 0, 2, 0, 4, 0],
                                               [0, -1, 0, 2, 0, 4]])
                    
                    for ilocal in range(0,3):
                          iglobal_x = 2*cls.mesh.IENbound[e,ilocal]
                          iglobal_y = 2*cls.mesh.IENbound[e,ilocal]+1
                          for jlocal in range(0,3):
                              jglobal_x = 2*cls.mesh.IENbound[e,jlocal]
                              jglobal_y = 2*cls.mesh.IENbound[e,jlocal]+1
                              
                              cls.Mb[iglobal_x,jglobal_x] = cls.Mb[iglobal_x,jglobal_x] + Mbe[2*ilocal,2*jlocal]
                              cls.Mb[iglobal_x,jglobal_y] = cls.Mb[iglobal_x,jglobal_y] + Mbe[2*ilocal,2*jlocal+1]
                              cls.Mb[iglobal_y,jglobal_y] = cls.Mb[iglobal_y,jglobal_y] + Mbe[2*ilocal+1,2*jlocal+1]
                              cls.Mb[iglobal_y,jglobal_x] = cls.Mb[iglobal_y,jglobal_x] + Mbe[2*ilocal+1,2*jlocal]
                      
    @classmethod   
    def build_blocks(cls):
        
        cls.A = cls.K.tocsr() + cls.rho*cls.h*cls.M.tocsr()/(cls.dt**2)
        # cls.b = np.zeros((2*cls.mesh.npoints), dtype='float') 
        cls.b = sp.sparse.csr_matrix.dot(cls.rho*cls.h*cls.M.tocsr()/(cls.dt**2),2.0*cls.u - cls.u_minus)
        
    @classmethod   
    def set_BC(cls):
        #Calculates force vector
        for bound in cls.BC:
            for i in range(len(cls.mesh.bound_dict[bound])):
                if cls.BC[bound][2] != 'None':
                    if type(cls.BC[bound][2]) == np.ndarray:
                        cls.forces[2*cls.mesh.bound_dict[bound][i]] = cls.BC[bound][2][i]
                    else:
                        cls.forces[2*cls.mesh.bound_dict[bound][i]] = cls.BC[bound][2]
                if cls.BC[bound][3] != 'None':
                    if type(cls.BC[bound][2]) == np.ndarray:
                        cls.forces[2*cls.mesh.bound_dict[bound][i]+1] = cls.BC[bound][3][i]
                    else:
                        cls.forces[2*cls.mesh.bound_dict[bound][i]+1] = cls.BC[bound][3]

        cls.b = cls.b + sp.sparse.csr_matrix.dot(cls.h*cls.Mb.tocsr(),cls.forces)
        
        #sets prescribed displacements
        for bound in cls.BC:
            for node in cls.mesh.bound_dict[bound]:
                if cls.BC[bound][0] != 'None':
                    row = cls.A.getrow(2*node)
                    for col in row.indices:
                        cls.A[2*node,col] = 0
                    cls.A[2*node,2*node] = 1.0
                    cls.b[2*node] = cls.BC[bound][0]
                if cls.BC[bound][1] != 'None':
                    row = cls.A.getrow(2*node+1)
                    for col in row.indices:
                        cls.A[2*node+1,col] = 0
                    cls.A[2*node+1,2*node+1] = 1.0
                    cls.b[2*node+1] = cls.BC[bound][1]
                                
            
    @classmethod   
    def solve(cls,i):
        
        cls.build_quad_GQ()
        cls.build_blocks()
        cls.set_BC()
                
        cls.u_minus = cls.u.copy()
        
        cls.u = sp.sparse.linalg.spsolve(cls.h*cls.A.tocsr(),cls.b)
        
        cls.ux = np.array([cls.u[2*n] for n in range(cls.mesh.npoints)])
        cls.uy = np.array([cls.u[2*n+1] for n in range(cls.mesh.npoints)])
        
        cls.mesh.X = cls.X_orig + cls.ux
        cls.mesh.Y = cls.Y_orig + cls.uy
        
        cls.u_prime = (cls.u - cls.u_minus)/cls.dt
        
        if cls.mesh.FSI_flag:
            cls.mesh.fluidmesh.X[cls.mesh.IEN_orig] = cls.mesh.X[cls.mesh.IEN]
            cls.mesh.fluidmesh.Y[cls.mesh.IEN_orig] = cls.mesh.Y[cls.mesh.IEN]
            
            u_prime_x =  np.array([cls.u_prime[2*n] for n in range(cls.mesh.npoints)])
            u_prime_y = np.array([cls.u_prime[2*n+1] for n in range(cls.mesh.npoints)])
            cls.mesh.fluid.vx[cls.mesh.IEN_orig] = u_prime_x[cls.mesh.IEN]
            cls.mesh.fluid.vy[cls.mesh.IEN_orig] = u_prime_y[cls.mesh.IEN]
        
        return cls.u
        
        
        
        
        
        