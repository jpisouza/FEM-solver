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
        cls.ux = np.zeros((cls.mesh.npoints), dtype='float')
        cls.uy = np.zeros((cls.mesh.npoints), dtype='float')
        cls.u_prime = np.zeros((2*cls.mesh.npoints), dtype='float')
        
        cls.sigma = np.zeros((cls.mesh.npoints,3), dtype='float')
        cls.PK_stress = np.zeros((cls.mesh.npoints,3), dtype='float')
        cls.PK_stress_vect = np.zeros((3*cls.mesh.npoints), dtype='float')
        cls.sigma_x = np.zeros((cls.mesh.npoints), dtype='float')
        cls.PK_stress_x = np.zeros((cls.mesh.npoints), dtype='float')
        cls.sigma_y = np.zeros((cls.mesh.npoints), dtype='float')
        cls.PK_stress_y = np.zeros((cls.mesh.npoints), dtype='float')       
        cls.tau_xy = np.zeros((cls.mesh.npoints), dtype='float')
        cls.PK_stress_xy = np.zeros((cls.mesh.npoints), dtype='float')
        cls.sigma_VM = np.zeros((cls.mesh.npoints), dtype='float')
        
        cls.u_minus = cls.u - cls.u_prime*cls.dt
                
        cls.X_orig = cls.mesh.X.copy()
        cls.Y_orig = cls.mesh.Y.copy()
        
        cls.mesh.X_orig = cls.mesh.X.copy()
        cls.mesh.Y_orig = cls.mesh.Y.copy()
        
        cls.calc_dof()
        cls.u_w = np.zeros((2*cls.mesh.npoints), dtype = 'float')

    @classmethod   
    def build_quad_GQHE(cls):
        cls.K = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        cls.M = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        cls.Mb = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        
        #matrices for strain calculation
        cls.Ms = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' ) 
        cls.Gxs = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Gys = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        
        #vector for stress calculation
        # cls.M_stress = lil_matrix( (2*cls.mesh.npoints,3*cls.mesh.npoints),dtype='float' ) 
        cls.Res_stress = np.zeros((2*cls.mesh.npoints),dtype='float' ) 
        
        quad = Elements.Quad(cls.mesh.X_orig,cls.mesh.Y_orig)
        
        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4,v5,v6 = cls.mesh.IEN[e]
            quad.getMSolidHE([v1,v2,v3,v4,v5,v6],cls.D,cls.ux,cls.uy,cls.PK_stress)
            quad.getM([v1,v2,v3,v4,v5,v6])
            k_elem = quad.k_solid
            m_elem = quad.mass_solid
            m_strain = quad.mass
            gradx_strain = quad.gvx
            grady_strain = quad.gvy
            m_stress = quad.mass_stress
            res_stress = quad.res_stress
            
            k=0
            IENgdl_e = np.zeros((12), dtype='int')
            for i in range (len(cls.mesh.IEN[e])):
                IENgdl_e[k] = 2*cls.mesh.IEN[e,i]
                IENgdl_e[k+1] = 2*cls.mesh.IEN[e,i]+1
                k+=2
            for ilocal in range(0,12):
                  iglobal = IENgdl_e[ilocal]
                  for jlocal in range(0,12):
                      jglobal = IENgdl_e[jlocal]
            
                      cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + k_elem[ilocal,jlocal]                     
                      cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + m_elem[ilocal,jlocal]
                  cls.Res_stress[iglobal] = cls.Res_stress[iglobal] + res_stress[ilocal]
                      
            #Strain matrices calculation
            for ilocal in range(0,6):
                  iglobal_s = cls.mesh.IEN[e,ilocal]
                  for jlocal in range(0,6):
                      jglobal_s = cls.mesh.IEN[e,jlocal]
                      cls.Ms[iglobal_s,jglobal_s] = cls.Ms[iglobal_s,jglobal_s] + m_strain[ilocal,jlocal]
                      cls.Gxs[iglobal_s,jglobal_s] = cls.Gxs[iglobal_s,jglobal_s] + gradx_strain[ilocal,jlocal]
                      cls.Gys[iglobal_s,jglobal_s] = cls.Gys[iglobal_s,jglobal_s] + grady_strain[ilocal,jlocal]
            
            #Stress matrices calculation
            # k=0
            # IENgdl_col_e = np.zeros((18), dtype='int')
            # for i in range (len(cls.mesh.IEN[e])):
            #     IENgdl_col_e[k] = 2*cls.mesh.IEN[e,i]
            #     IENgdl_col_e[k+1] = 2*cls.mesh.IEN[e,i]+1
            #     IENgdl_col_e[k+2] = 2*cls.mesh.IEN[e,i]+2
            #     k+=3
            # k=0
            # IENgdl_row_e = np.zeros((12), dtype='int')
            # for i in range (len(cls.mesh.IEN[e])):
            #     IENgdl_row_e[k] = 2*cls.mesh.IEN[e,i]
            #     IENgdl_row_e[k+1] = 2*cls.mesh.IEN[e,i]+1
            #     k+=2
            
            # for ilocal in range(0,12):
            #       iglobal = IENgdl_row_e[ilocal]
            #       for jlocal in range(0,18):
            #           jglobal = IENgdl_col_e[jlocal]
            
            #           cls.M_stress[iglobal,jglobal] = cls.M_stress[iglobal,jglobal] + m_stress[ilocal,jlocal]                     
                      
        
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
    def build_quad_GQ(cls):
        cls.K = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        cls.M = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        cls.Mb = lil_matrix( (2*cls.mesh.npoints,2*cls.mesh.npoints),dtype='float' )
        
        #matrices for stress calculation
        cls.Ms = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' ) 
        cls.Gxs = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Gys = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        
        quad = Elements.Quad(cls.mesh.X,cls.mesh.Y)
        
        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4,v5,v6 = cls.mesh.IEN[e]
            quad.getMSolid([v1,v2,v3,v4,v5,v6],cls.D)
            quad.getM([v1,v2,v3,v4,v5,v6])
            k_elem = quad.k_solid
            m_elem = quad.mass_solid
            m_stress = quad.mass
            gradx_stress = quad.gvx
            grady_stress = quad.gvy
            
            k=0
            IENgdl_e = np.zeros((12), dtype='int')
            for i in range (len(cls.mesh.IEN[e])):
                IENgdl_e[k] = 2*cls.mesh.IEN[e,i]
                IENgdl_e[k+1] = 2*cls.mesh.IEN[e,i]+1
                k+=2
            for ilocal in range(0,12):
                  iglobal = IENgdl_e[ilocal]
                  for jlocal in range(0,12):
                      jglobal = IENgdl_e[jlocal]
            
                      cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + k_elem[ilocal,jlocal]                     
                      cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + m_elem[ilocal,jlocal]
                      
            #Strain matrices calculation
            for ilocal in range(0,6):
                  iglobal_s = cls.mesh.IEN[e,ilocal]
                  for jlocal in range(0,6):
                      jglobal_s = cls.mesh.IEN[e,jlocal]
                      cls.Ms[iglobal_s,jglobal_s] = cls.Ms[iglobal_s,jglobal_s] + m_stress[ilocal,jlocal]
                      cls.Gxs[iglobal_s,jglobal_s] = cls.Gxs[iglobal_s,jglobal_s] + gradx_stress[ilocal,jlocal]
                      cls.Gys[iglobal_s,jglobal_s] = cls.Gys[iglobal_s,jglobal_s] + grady_stress[ilocal,jlocal]
        
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
        
        cls.A = cls.h*cls.K.tocsr() + cls.rho*cls.h*cls.M.tocsr()/(cls.dt**2)
        cls.b = sp.sparse.csr_matrix.dot(cls.rho*cls.h*cls.M.tocsr()/(cls.dt**2),2.0*cls.u - cls.u_minus)
        
    @classmethod   
    def build_blocks_static(cls):
        
        cls.A = cls.h*cls.K.tocsr()
        cls.b = np.zeros((2*cls.mesh.npoints), dtype='float') 
    
    @classmethod   
    def calc_bound_force(cls):
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
    @classmethod   
    def set_BCHE(cls, Res):
        
        cls.b = Res  
        
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
    def set_BC(cls):
        #Calculates force vector
        # for bound in cls.BC:
        #     for i in range(len(cls.mesh.bound_dict[bound])):
        #         if cls.BC[bound][2] != 'None':
        #             if type(cls.BC[bound][2]) == np.ndarray:
        #                 cls.forces[2*cls.mesh.bound_dict[bound][i]] = cls.BC[bound][2][i]
        #             else:
        #                 cls.forces[2*cls.mesh.bound_dict[bound][i]] = cls.BC[bound][2]
        #         if cls.BC[bound][3] != 'None':
        #             if type(cls.BC[bound][2]) == np.ndarray:
        #                 cls.forces[2*cls.mesh.bound_dict[bound][i]+1] = cls.BC[bound][3][i]
        #             else:
        #                 cls.forces[2*cls.mesh.bound_dict[bound][i]+1] = cls.BC[bound][3]

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
    def calc_stressHE(cls):
        
        duxdx = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gxs,cls.ux))
        duxdy = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gys,cls.ux))
        
        duydx = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gxs,cls.uy))
        duydy = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gys,cls.uy))
        
        cls.E_x = duxdx +(1.0/2.0)*(duxdx**2 + duydx**2)
        cls.E_y = duydy + (1.0/2.0)*(duxdy**2 + duydy**2)
        cls.E_xy = duxdy + duydx + duxdy*duxdx + duydx*duydy
        
        cls.E_ = np.array([cls.E_x,cls.E_y,cls.E_xy])
        cls.PK_stress = np.transpose(cls.D@cls.E_)
        
        cls.PK_stress_x = cls.PK_stress[:,0]
        cls.PK_stress_y = cls.PK_stress[:,1]
        cls.PK_stress_xy = cls.PK_stress[:,2]
        
        #builds stress in a single 1D array
        k=0
        for i in range(cls.mesh.npoints):
            cls.PK_stress_vect[k] = cls.PK_stress_x[i]
            cls.PK_stress_vect[k+1] = cls.PK_stress_y[i]
            cls.PK_stress_vect[k+2] = cls.PK_stress_xy[i]
            k+=3
        
        F11 = 1.0 + duxdx
        F12 = duxdy
        F21 = duydx
        F22 = 1.0 + duydy
        
        detF = F11*F22 - F21*F12
        
        cls.sigma_x = 1.0/detF*(F11**2*cls.PK_stress_x + 2*F11*F12*cls.PK_stress_xy + F12**2*cls.PK_stress_y)
        cls.sigma_y = 1.0/detF*(F21**2*cls.PK_stress_x + 2*F21*F22*cls.PK_stress_xy + F22**2*cls.PK_stress_y)
        cls.tau_xy = 1.0/detF*(F21*F11*cls.PK_stress_x + (F22*F11 + F12*F21)*cls.PK_stress_xy + F22*F12*cls.PK_stress_y)
        
        cls.sigma[:,0] = cls.sigma_x
        cls.sigma[:,1] = cls.sigma_y
        cls.sigma[:,2] = cls.tau_xy
                    
        cls.sigma_VM = np.power(np.power(cls.sigma_x,2) + np.power(cls.sigma_y,2) - np.multiply(cls.sigma_x, cls.sigma_y) + 3.0*np.power(cls.tau_xy,2),0.5)
    
    @classmethod   
    def calc_stress(cls):
        
        duxdx = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gxs,cls.ux))
        duxdy = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gys,cls.ux))
        
        duydx = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gxs,cls.uy))
        duydy = sp.sparse.linalg.spsolve(cls.Ms,sp.sparse.csr_matrix.dot(cls.Gys,cls.uy))
        
        cls.eps_x = duxdx
        cls.eps_y = duydy
        cls.eps_xy = duxdy + duydx
        
        cls.eps = np.array([cls.eps_x,cls.eps_y,cls.eps_xy])
        cls.sigma = np.transpose(cls.D@cls.eps)
        
        cls.sigma_x = cls.sigma[:,0]
        cls.sigma_y = cls.sigma[:,1]
        cls.tau_xy = cls.sigma[:,2]
                            
        cls.sigma_VM = np.power(np.power(cls.sigma_x,2) + np.power(cls.sigma_y,2) - np.multiply(cls.sigma_x, cls.sigma_y) + 3.0*np.power(cls.tau_xy,2),0.5)
      
    @classmethod 
    def calc_freq(cls):
        for bound in cls.BC:
            for node in cls.mesh.bound_dict[bound]:
                if cls.BC[bound][0] != 'None':
                    cls.dof.remove(2*node)
                if cls.BC[bound][1] != 'None':
                    cls.dof.remove(2*node + 1)
                    
        cls.M_freq = lil_matrix( (len(cls.dof),len(cls.dof)),dtype='float' )
        cls.K_freq = lil_matrix( (len(cls.dof),len(cls.dof)),dtype='float' )
        for i in range(len(cls.dof)):
            for j in range(len(cls.dof)):
                cls.M_freq[i,j] = cls.M[cls.dof[i],cls.dof[j]]
                cls.K_freq[i,j] = cls.K[cls.dof[i],cls.dof[j]]
                
        # omega, modes = scipy.sparse.linalg.eigs(scipy.sparse.linalg.inv(cls.rho*cls.h*cls.M_freq.tocsc())*cls.h*cls.K_freq.tocsc(),6,None,None,'SM')
        omega, modes = np.linalg.eig((scipy.sparse.linalg.inv(cls.rho*cls.h*cls.M_freq.tocsc())*cls.h*cls.K_freq.tocsc()).toarray())        
        #omega, modes = scipy.sparse.linalg.eigs(cls.h*cls.K_freq.tocsc(),6,cls.rho*cls.h*cls.M_freq.tocsc(),None,'SM')
        
        indexes_omega = np.argsort(omega)
        cls.omega_sort = omega[indexes_omega]
        
        # print('---------------------Natural Frequencies--------------------------------')
        # print(cls.omega_sort)
        # print('------------------------------------------------------------------------')
        
        cls.u_w = np.zeros((2*cls.mesh.npoints,len(indexes_omega)), dtype = 'float')
        cls.u1 = modes[:,indexes_omega]
        cls.u_w[cls.dof,:] = 10*cls.u1

        
    @classmethod 
    def calc_dof(cls):
        cls.dof = []
        for i in range(cls.mesh.npoints):
            cls.dof.append(2*i)
            cls.dof.append(2*i + 1)
        
    @classmethod   
    def solve(cls,i, mesh_factor=0, nat_freq=False):
        
        cls.build_quad_GQ()
        cls.build_blocks()
        cls.calc_bound_force()
        cls.set_BC()
        
        if i==0 and nat_freq:
            cls.calc_freq()
                
        cls.u_minus = cls.u.copy()
        
        cls.u = sp.sparse.linalg.spsolve(cls.A.tocsr(),cls.b)
        
        cls.ux = np.array([cls.u[2*n] for n in range(cls.mesh.npoints)])
        cls.uy = np.array([cls.u[2*n+1] for n in range(cls.mesh.npoints)])
        
        cls.mesh.X = cls.X_orig + cls.ux
        cls.mesh.Y = cls.Y_orig + cls.uy
        
        cls.u_prime = (cls.u - cls.u_minus)/cls.dt
        
        cls.calc_stress()
        
        if cls.mesh.FSI_flag:
            cls.mesh.fluidmesh.mesh_displacement[cls.mesh.IEN_orig,0] = cls.ux[cls.mesh.IEN]
            cls.mesh.fluidmesh.mesh_displacement[cls.mesh.IEN_orig,1] = cls.uy[cls.mesh.IEN]
            
            cls.mesh.fluidmesh.X[cls.mesh.IEN_orig] = cls.mesh.X[cls.mesh.IEN]
            cls.mesh.fluidmesh.Y[cls.mesh.IEN_orig] = cls.mesh.Y[cls.mesh.IEN]
            
            u_prime_x =  np.array([cls.u_prime[2*n] for n in range(cls.mesh.npoints)])
            u_prime_y = np.array([cls.u_prime[2*n+1] for n in range(cls.mesh.npoints)])
            cls.mesh.fluid.vx[cls.mesh.IEN_orig] = u_prime_x[cls.mesh.IEN]
            cls.mesh.fluid.vy[cls.mesh.IEN_orig] = u_prime_y[cls.mesh.IEN]
            cls.mesh.fluidmesh.mesh_velocity[cls.mesh.IEN_orig,0] = u_prime_x[cls.mesh.IEN]
            cls.mesh.fluidmesh.mesh_velocity[cls.mesh.IEN_orig,1] = u_prime_y[cls.mesh.IEN]
            
            for node in cls.mesh.fluidmesh.node_list:
                if node.ID not in cls.mesh.IEN_orig and node.ID not in cls.mesh.fluidmesh.IENbound:
                    factor = -mesh_factor*node.FSI_dist[1]
                    cls.mesh.fluidmesh.mesh_velocity[node.ID,0] = cls.mesh.fluidmesh.mesh_velocity[node.FSI_dist[0],0]*np.exp(factor)
                    cls.mesh.fluidmesh.mesh_velocity[node.ID,1] = cls.mesh.fluidmesh.mesh_velocity[node.FSI_dist[0],1]*np.exp(factor)
                    cls.mesh.fluidmesh.mesh_displacement[node.ID,0] = cls.mesh.fluidmesh.mesh_velocity[node.ID,0]*cls.dt
                    cls.mesh.fluidmesh.mesh_displacement[node.ID,1] = cls.mesh.fluidmesh.mesh_velocity[node.ID,1]*cls.dt
                    cls.mesh.fluidmesh.X[node.ID] += cls.mesh.fluidmesh.mesh_displacement[node.ID,0]
                    cls.mesh.fluidmesh.Y[node.ID] += cls.mesh.fluidmesh.mesh_displacement[node.ID,1]
        
        return cls.u, cls.u_w
    
    @classmethod   
    def solve_static(cls, nat_freq):
        cls.build_quad_GQ()
        cls.build_blocks_static()
        cls.calc_bound_force()
        cls.set_BC()
        
        if nat_freq:
            cls.calc_freq()
        
        cls.u = sp.sparse.linalg.spsolve(cls.A.tocsr(),cls.b)
        
        cls.ux = np.array([cls.u[2*n] for n in range(cls.mesh.npoints)])
        cls.uy = np.array([cls.u[2*n+1] for n in range(cls.mesh.npoints)])
        
        # cls.mesh.X = cls.X_orig + cls.ux
        # cls.mesh.Y = cls.Y_orig + cls.uy
        
        cls.calc_stress()
        
        return cls.u, cls.u_w
    
    @classmethod   
    def solve_staticHE(cls):
        
        tol = 0.003
        error = 1.0
        cls.build_quad_GQHE()
        cls.build_blocks_static()
        cls.calc_bound_force()
        
        # Res = sp.sparse.csr_matrix.dot(cls.h*cls.Mb.tocsr(),cls.forces) - sp.sparse.csr_matrix.dot(cls.h*cls.M_stress.tocsr(), cls.PK_stress_vect)
        Res = sp.sparse.csr_matrix.dot(cls.h*cls.Mb.tocsr(),cls.forces) - cls.h*cls.Res_stress
        cls.set_BCHE(Res)
        du = np.ones((2*cls.mesh.npoints), dtype='float')
        
        i = 1
        while error > tol and i<=30:
            
            du_ant = du
            du = sp.sparse.linalg.spsolve(cls.A.tocsr(),cls.b)
            cls.u = cls.u + du
            
            cls.ux = np.array([cls.u[2*n] for n in range(cls.mesh.npoints)])
            cls.uy = np.array([cls.u[2*n+1] for n in range(cls.mesh.npoints)])
            
            cls.calc_stressHE()
            
            cls.build_quad_GQHE()
            cls.build_blocks_static()
            cls.calc_bound_force()
            
            Res_ant = Res
            # Res = sp.sparse.csr_matrix.dot(cls.h*cls.Mb.tocsr(),cls.forces) - sp.sparse.csr_matrix.dot(cls.h*cls.M_stress.tocsr(), cls.PK_stress_vect)
            Res = sp.sparse.csr_matrix.dot(cls.h*cls.Mb.tocsr(),cls.forces) - cls.h*cls.Res_stress          
            cls.set_BCHE(Res)
            
            
            error = np.sqrt(sum((du - du_ant)**2))/np.sqrt(sum((du_ant)**2))
            error = np.sqrt(sum((Res)**2))/np.sqrt(sum((Res_ant)**2))

            
            print ('Iteration ' + str(i) + '-----Error = ' + str(error))
            i+=1
            
        # cls.mesh.X = cls.X_orig + cls.ux
        # cls.mesh.Y = cls.Y_orig + cls.uy
            
            
        
        return cls.u
        
        
        