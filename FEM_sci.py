import numpy as np
from timeit import default_timer as timer
import scipy as sp
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, dok_matrix
import scipy.linalg  
import scipy.sparse.linalg 
import MathWrapper
from semi_lagrangiano import semi_lagrange2
import SL
import Elements
# import solveSys
# from julia import Main


class FEM:
    
    @classmethod
    def set_matrices(cls,mesh,fluid,dt,BC):

        start = timer()

        cls.fluid = fluid
        cls.Re = fluid.Re
        cls.Ga = fluid.Ga
        cls.Gr = fluid.Gr
        cls.Pr = fluid.Pr
        cls.dt = dt
        cls.mesh = mesh
        cls.BC = BC
        
        
        # Main.include('LinSolve.jl')
        
        cls.K = lil_matrix( (mesh.npoints,mesh.npoints),dtype='float' )
        cls.M = lil_matrix( (mesh.npoints,mesh.npoints),dtype='float' )
        cls.Gx = lil_matrix( (mesh.npoints,mesh.npoints_p),dtype='float' )
        cls.Gy = lil_matrix( (mesh.npoints,mesh.npoints_p),dtype='float' )
        cls.Dx = lil_matrix( (mesh.npoints_p,mesh.npoints),dtype='float' )
        cls.Dy = lil_matrix( (mesh.npoints_p,mesh.npoints),dtype='float' )
        cls.M_T = lil_matrix( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        cls.K_T = lil_matrix( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        cls.Gx_T = lil_matrix( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        cls.Gy_T = lil_matrix( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        
        if cls.mesh.mesh_kind == 'mini':
            cls.build_mini()
        elif cls.mesh.mesh_kind == 'quad':
            cls.build_quad_GQ()
        
        cls.set_block_matrices(BC)
        
        end = timer()
        print('time --> Build FEM matrices = ' + str(end-start) + ' [s]')

    @classmethod
    def build_quad_GQ(cls):
        quad = Elements.Quad(cls.mesh.X,cls.mesh.Y)
        lin = Elements.Linear(cls.mesh.X[:cls.mesh.npoints_p],cls.mesh.Y[:cls.mesh.npoints_p])
        # loop de elementos finitos
        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4,v5,v6 = cls.mesh.IEN[e]
            quad.getM([v1,v2,v3,v4,v5,v6])
            lin.getM([v1,v2,v3])
            kelem = quad.kxx + quad.kyy
            melem = quad.mass
            kelem_T = lin.kxx + lin.kyy
            melem_T = lin.mass
            for ilocal in range(0,6):
                 iglobal = cls.mesh.IEN[e,ilocal]
                 for jlocal in range(0,6):
                     jglobal = cls.mesh.IEN[e,jlocal]
            
                     cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + kelem[ilocal,jlocal]             
                     cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + melem[ilocal,jlocal]
                     
                 for jlocal in range(0,3):
                     jglobal = cls.mesh.IEN[e,jlocal]
                     
                     if ilocal <=2:                 
                         cls.M_T[iglobal,jglobal] = cls.M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                         cls.K_T[iglobal,jglobal] = cls.K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                         cls.Gx_T[iglobal,jglobal] = cls.Gx_T[iglobal,jglobal] + lin.gx[ilocal,jlocal]
                         cls.Gy_T[iglobal,jglobal] = cls.Gy_T[iglobal,jglobal] + lin.gy[ilocal,jlocal]
                         
                     cls.Gx[iglobal,jglobal] = cls.Gx[iglobal,jglobal] + quad.gx[ilocal,jlocal]
                     cls.Gy[iglobal,jglobal] = cls.Gy[iglobal,jglobal] + quad.gy[ilocal,jlocal]
                     cls.Dx[jglobal,iglobal] = cls.Dx[jglobal,iglobal] + quad.dx[jlocal,ilocal]
                     cls.Dy[jglobal,iglobal] = cls.Dy[jglobal,iglobal] + quad.dy[jlocal,ilocal]

        # cls.Dx = cls.Gx.transpose()
        # cls.Dy = cls.Gy.transpose()

    @classmethod
    def build_mini_GQ(cls):
        mini = Elements.Mini(cls.mesh.X,cls.mesh.Y)
        lin = Elements.Linear(cls.mesh.X[:cls.mesh.npoints_p],cls.mesh.Y[:cls.mesh.npoints_p])
        # loop de elementos finitos
        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4 = cls.mesh.IEN[e]
            mini.getM([v1,v2,v3,v4])
            lin.getM([v1,v2,v3])
            kelem = mini.kxx + mini.kyy
            melem = mini.mass
            kelem_T = lin.kxx + lin.kyy
            melem_T = lin.mass
            for ilocal in range(0,4):
                 iglobal = cls.mesh.IEN[e,ilocal]
                 for jlocal in range(0,4):
                     jglobal = cls.mesh.IEN[e,jlocal]
            
                     cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + kelem[ilocal,jlocal]             
                     cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + melem[ilocal,jlocal]
                     
                 for jlocal in range(0,3):
                     jglobal = cls.mesh.IEN[e,jlocal]
                     
                     if ilocal <=2:                 
                         cls.M_T[iglobal,jglobal] = cls.M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                         cls.K_T[iglobal,jglobal] = cls.K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                         cls.Gx_T[iglobal,jglobal] = cls.Gx_T[iglobal,jglobal] + lin.gx[ilocal,jlocal]
                         cls.Gy_T[iglobal,jglobal] = cls.Gy_T[iglobal,jglobal] + lin.gy[ilocal,jlocal]
                         
                     cls.Gx[iglobal,jglobal] = cls.Gx[iglobal,jglobal] + mini.gx[ilocal,jlocal]
                     cls.Gy[iglobal,jglobal] = cls.Gy[iglobal,jglobal] + mini.gy[ilocal,jlocal]

        cls.Dx = cls.Gx.transpose()
        cls.Dy = cls.Gy.transpose()

    @classmethod
    def build_mini(cls):
        # loop de elementos finitos
        for e in range(0,cls.mesh.ne):
             v0,v1,v2,v3 = cls.mesh.IEN[e]
        
             x0 = cls.mesh.X[v0]
             x1 = cls.mesh.X[v1]
             x2 = cls.mesh.X[v2]
             
             y0 = cls.mesh.Y[v0]
             y1 = cls.mesh.Y[v1]
             y2 = cls.mesh.Y[v2]
             
             b0 = y1 - y2
             b1 = y2 - y0
             b2 = y0 - y1
             
             c0 = x2 - x1
             c1 = x0 - x2
             c2 = x1 - x0
             
             Matriz = np.array([[1,x0,y0],
                                [1,x1,y1],
                                [1,x2,y2]],dtype='float')
             A = 0.5*np.linalg.det(Matriz)
             
             
            
             kx_lin = (1.0/(4.0*A))*np.array( [[b0**2,b0*b1,b0*b2],
                                             [b1*b0,b1**2,b1*b2],
                                             [b2*b0,b2*b1,b2**2]],dtype='float')
             ky_lin = (1.0/(4.0*A))*np.array( [[c0**2,c0*c1,c0*c2],
                                             [c1*c0,c1**2,c1*c2],
                                             [c2*c0,c2*c1,c2**2]],dtype='float')
             
             zx = (1.0/(4.0*A))*(b1**2+b1*b2+b2**2)
             zy = (1.0/(4.0*A))*(c1**2+c1*c2+c2**2)
             
             kx_aux = np.append(kx_lin +(9.0/10.0)*zx,[[-27.0*zx/10.0,-27.0*zx/10.0,-27.0*zx/10.0]],axis=0)
             kx = np.append(kx_aux,[[-27.0*zx/10.0],[-27.0*zx/10.0],[-27.0*zx/10.0],[81.0*zx/10.0]],axis=1)
                            
             ky_aux = np.append(ky_lin +(9.0/10.0)*zy,[[-27.0*zy/10.0,-27.0*zy/10.0,-27.0*zy/10.0]],axis=0)
             ky = np.append(ky_aux,[[-27.0*zy/10.0],[-27.0*zy/10.0],[-27.0*zy/10.0],[81.0*zy/10.0]],axis=1)
             
             kelem = kx + ky
             
             kelem_T = kx_lin + ky_lin
             
             melem = (A/840.0)*np.array( [[83,13,13,45],
                                         [13,83,13,45],
                                         [13,13,83,45],
                                            [45,45,45,243]],dtype='float')
             melem_T = (A/12.0)*np.array( [[2,1,1],
                                         [1,2,1],
                                         [1,1,2]],dtype='float')
             
             gx_lin=(1.0/6.0)*np.array([[b0,b1,b2],
                                    [b0,b1,b2],
                                    [b0,b1,b2]],dtype='float')
             
             gy_lin=(1.0/6.0)*np.array([[c0,c1,c2],
                                    [c0,c1,c2],
                                    [c0,c1,c2]],dtype='float')
        
             gx = np.append((9.0/20.0)*gx_lin + np.transpose(gx_lin),[[(-9.0/40.0)*b0,(-9.0/40.0)*b1,(-9.0/40.0)*b2]],axis=0)
             gy = np.append((9.0/20.0)*gy_lin + np.transpose(gy_lin),[[(-9.0/40.0)*c0,(-9.0/40.0)*c1,(-9.0/40.0)*c2]],axis=0)
             
             for ilocal in range(0,4):
                 iglobal = cls.mesh.IEN[e,ilocal]
                 for jlocal in range(0,4):
                     jglobal = cls.mesh.IEN[e,jlocal]
            
                     cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + kelem[ilocal,jlocal]             
                     cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + melem[ilocal,jlocal]
                     
                 for jlocal in range(0,3):
                     jglobal = cls.mesh.IEN[e,jlocal]
                     
                     if ilocal <=2:                 
                         cls.M_T[iglobal,jglobal] = cls.M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                         cls.K_T[iglobal,jglobal] = cls.K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                         cls.Gx_T[iglobal,jglobal] = cls.Gx_T[iglobal,jglobal] + gx_lin[ilocal,jlocal]
                         cls.Gy_T[iglobal,jglobal] = cls.Gy_T[iglobal,jglobal] + gy_lin[ilocal,jlocal]
                         
                     cls.Gx[iglobal,jglobal] = cls.Gx[iglobal,jglobal] + gx[ilocal,jlocal]
                     cls.Gy[iglobal,jglobal] = cls.Gy[iglobal,jglobal] + gy[ilocal,jlocal]

        cls.Dx = cls.Gx.transpose()
        cls.Dy = cls.Gy.transpose()

    @classmethod
    def set_block_matrices(cls,BC):
        
        A1 = (1.0/cls.dt)*cls.M + (1.0/cls.Re)*cls.K
        
        cls.Matriz = sp.sparse.bmat([ [ A1, None, -cls.Gx],
                                      [ None, A1, -cls.Gy],
                                      [ cls.Dx, cls.Dy, None]])
        
        cls.Matriz_T=(1.0/cls.dt)*cls.M_T + (1.0/(cls.Re*cls.Pr))*cls.K_T
        
        cls.Matriz = cls.Matriz.tocsr()
        cls.Matriz_T = cls.Matriz_T.tocsr()
        
        for i in range(len(BC)):
            
            for j in cls.mesh.boundary[i]:
                if BC[i]['vx'] != 'None':
                    row = cls.Matriz.getrow(j)
                    for col in row.indices:
                        cls.Matriz[j,col] = 0
                    cls.Matriz[j,j] = 1.0
                
                if BC[i]['vy'] != 'None':
                    row = cls.Matriz.getrow(j + cls.mesh.npoints)
                    for col in row.indices:
                        cls.Matriz[j + cls.mesh.npoints,col] = 0
                    cls.Matriz[j + cls.mesh.npoints,j + cls.mesh.npoints] = 1.0

                if j < cls.mesh.npoints_p:   
                    if BC[i]['T'] != 'None':
                        row = cls.Matriz_T.getrow(j)
                        for col in row.indices:
                            cls.Matriz_T[j,col] = 0
                        cls.Matriz_T[j,j] = 1.0
                        
                    if BC[i]['p'] != 'None':  
                        row = cls.Matriz.getrow(j + 2*cls.mesh.npoints)
                        for col in row.indices:
                            cls.Matriz[j + 2*cls.mesh.npoints,col] = 0
                        cls.Matriz = cls.Matriz.tolil()
                        cls.Matriz[j + 2*cls.mesh.npoints,j + 2*cls.mesh.npoints] = 1.0
                        cls.Matriz = cls.Matriz.tocsr()
        cls.Matriz = cls.Matriz.tocsr()
        cls.Matriz_T = cls.Matriz_T.tocsr()         

    @classmethod
    def set_block_vectors(cls,BC,forces):
        cls.M = cls.M.tocsr()
        cls.M_T = cls.M_T.tocsr()
        cls.vetor_vx = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M,cls.fluid.vxd) + sp.sparse.csr_matrix.dot(cls.M,forces[:,0])

        if cls.mesh.mesh_kind == 'mini':
            cls.vetor_vy = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M,cls.fluid.vyd) + sp.sparse.csr_matrix.dot(cls.M,cls.fluid.Gr*cls.fluid.T_mini + forces[:,1] -cls.fluid.Ga)
        elif cls.mesh.mesh_kind == 'quad':
            cls.vetor_vy = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M,cls.fluid.vyd) + sp.sparse.csr_matrix.dot(cls.M,cls.fluid.Gr*cls.fluid.T_quad + forces[:,1] -cls.fluid.Ga)
        cls.vetor_p = np.zeros((cls.mesh.npoints_p),dtype='float' )
        cls.vetor_T = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M_T,cls.fluid.Td[0:cls.mesh.npoints_p])

        
        for i in range(len(BC)):
            
            for j in cls.mesh.boundary[i]:
                if 'Profile' in BC[i]['vx']:
                    u_0 = float(BC[i]['vx'].split('-')[1])
                    ref = (np.min(cls.mesh.Y[cls.mesh.boundary[i]]) + np.max(cls.mesh.Y[cls.mesh.boundary[i]]))/2.0
                    h = np.abs(np.min(cls.mesh.Y[cls.mesh.boundary[i]]) - np.max(cls.mesh.Y[cls.mesh.boundary[i]]))
                    cls.vetor_vx[j] = 1.5*u_0*(1.0 - (2.0*(cls.mesh.Y[j]-ref)/h)**2)
                elif BC[i]['vx'] != 'None':
                    cls.vetor_vx[j] = float(BC[i]['vx'])

                if 'Profile' in BC[i]['vy']:
                    u_0 = float(BC[i]['vy'].split('-')[1])
                    ref = (np.min(cls.mesh.X[cls.mesh.boundary[i]]) + np.max(cls.mesh.X[cls.mesh.boundary[i]]))/2.0
                    h = np.abs(np.min(cls.mesh.X[cls.mesh.boundary[i]]) - np.max(cls.mesh.X[cls.mesh.boundary[i]]))
                    cls.vetor_vy[j] = 1.5*u_0*(1.0 - (2.0*(cls.mesh.X[j]-ref)/h)**2)
                elif BC[i]['vy'] != 'None':
                    cls.vetor_vy[j] = float(BC[i]['vy']) 

                if j < cls.mesh.npoints_p: 
                    if BC[i]['T'] != 'None':
                        cls.vetor_T[j] = float(BC[i]['T'])
                    if BC[i]['p'] != 'None':
                        cls.vetor_p[j] = float(BC[i]['p'])
        

        cls.vetor_vx = csr_matrix(cls.vetor_vx.reshape(-1,len(cls.vetor_vx)))
        cls.vetor_vy = csr_matrix(cls.vetor_vy.reshape(-1,len(cls.vetor_vy)))
        cls.vetor_p = csr_matrix(cls.vetor_p.reshape(-1,len(cls.vetor_p)))
        cls.vetor_T = csr_matrix(cls.vetor_T.reshape(-1,len(cls.vetor_T)))

        cls.vetor = sp.sparse.bmat([[cls.vetor_vx,cls.vetor_vy,cls.vetor_p]])
        cls.vetor = cls.vetor.tocsr()
        cls.vetor_T = cls.vetor_T.tocsr()
    
    @classmethod
    def solve_fields(cls,forces,SLG=False,neighborElem=[[]],oface=[]):

        if SLG:
            start = timer()
            sl = SL.Linear(cls.mesh.IEN,cls.mesh.X,cls.mesh.Y,neighborElem,oface,cls.fluid.vx,cls.fluid.vy)
            sl.compute(cls.dt)
            cls.fluid.vxd = sl.conv*cls.fluid.vx[0:cls.mesh.npoints_p]
            cls.fluid.vyd = sl.conv*cls.fluid.vy[0:cls.mesh.npoints_p]
            cls.fluid.Td = sl.conv*cls.fluid.T
            # interpolando no centroid
            [cls.fluid.vxd,cls.fluid.vyd] = sl.setCentroid(cls.fluid.vxd,cls.fluid.vyd)
            end = timer()
            print('time --> SL calculation = ' + str(end-start) + ' [s]')

        else:

            start = timer()
            cls.fluid.vxd, cls.fluid.vyd, cls.fluid.Td = semi_lagrange2(cls.mesh.node_list,cls.mesh.elem_list,cls.fluid.vx,cls.fluid.vy,cls.fluid.T,cls.dt,cls.mesh.IENbound)
            end = timer()
            print('time --> SL calculation = ' + str(end-start) + ' [s]')
        
        cls.fluid.T_mini[0:cls.mesh.npoints_p] = cls.fluid.T
        cls.fluid.T_quad[0:cls.mesh.npoints_p] = cls.fluid.T
        cls.fluid.p_quad[0:cls.mesh.npoints_p] = cls.fluid.p

        if cls.mesh.mesh_kind == 'mini':
            T_ = cls.fluid.T[cls.mesh.IEN_orig]
            centroids_T = T_.sum(axis=1)/3.0
            cls.fluid.T_mini[cls.mesh.npoints_p:] = centroids_T
        elif cls.mesh.mesh_kind == 'quad':
            for i in range (3,6):
                j=i-2
                if i == 5:
                    j = 0
                cls.fluid.T_quad[cls.mesh.IEN[:,i]]  = (cls.fluid.T_quad[cls.mesh.IEN[:,j]] + cls.fluid.T_quad[cls.mesh.IEN[:,i-3]])/2.0
                cls.fluid.p_quad[cls.mesh.IEN[:,i]]  = (cls.fluid.p_quad[cls.mesh.IEN[:,j]] + cls.fluid.p_quad[cls.mesh.IEN[:,i-3]])/2.0
                
        
        start = timer()
        cls.set_block_vectors(cls.BC,forces)
        end = timer()
        print('time --> Set boundaries = ' + str(end-start) + ' [s]')
        
        start = timer()
        sol = sp.sparse.linalg.spsolve(cls.Matriz,cls.vetor.transpose())
        # sol = solveSys.solveSystem(cls.vetor)
        # sol = Main.solve(cls.Matriz,cls.vetor)
        # sol = MathWrapper.linSolve(cls.Matriz,cls.vetor,cls.dll)
        end = timer()  
        print('time --> Flow solution = ' + str(end-start) + ' [s]')
        
        cls.fluid.vx = sol[0:cls.mesh.npoints]
        cls.fluid.vy = sol[cls.mesh.npoints:2*cls.mesh.npoints]
        cls.fluid.p = sol[2*cls.mesh.npoints:]
        
        start = timer()
        cls.fluid.T = sp.sparse.linalg.spsolve(cls.Matriz_T,cls.vetor_T.transpose())
        # cls.fluid.T = MathWrapper.linSolve(cls.Matriz_T,cls.vetor_T,cls.dll)
        end = timer()
        print('time --> Temperatute solution = ' + str(end-start) + ' [s]')

        return cls.fluid
    
   
        
            

