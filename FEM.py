import numpy as np
from timeit import default_timer as timer 
import MathWrapper
# import solveSys
# from julia import Main
from semi_lagrangiano import semi_lagrange2

class FEM:
    
    @classmethod
    def set_matrices(cls,mesh,fluid,dt,BC):
        
        cls.fluid = fluid
        cls.Re = fluid.Re
        cls.Ga = fluid.Ga
        cls.Gr = fluid.Gr
        cls.Pr = fluid.Pr
        cls.dt = dt
        cls.mesh = mesh
        cls.BC = BC
        
        
        # Main.include('LinSolve.jl')
        
        cls.K = np.zeros( (mesh.npoints,mesh.npoints),dtype='float' )
        cls.M = np.zeros( (mesh.npoints,mesh.npoints),dtype='float' )
        cls.Gx = np.zeros( (mesh.npoints,mesh.npoints_p),dtype='float' )
        cls.Gy = np.zeros( (mesh.npoints,mesh.npoints_p),dtype='float' )
        cls.M_T = np.zeros( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        cls.K_T = np.zeros( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        cls.Gx_T = np.zeros( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        cls.Gy_T = np.zeros( (mesh.npoints_p,mesh.npoints_p),dtype='float' )
        
        # loop de elementos finitos
        for e in range(0,mesh.ne):
             v0,v1,v2,v3 = mesh.IEN[e]
        
             x0 = mesh.X[v0]
             x1 = mesh.X[v1]
             x2 = mesh.X[v2]
             
             y0 = mesh.Y[v0]
             y1 = mesh.Y[v1]
             y2 = mesh.Y[v2]
             
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
                 iglobal = mesh.IEN[e,ilocal]
                 for jlocal in range(0,4):
                     jglobal = mesh.IEN[e,jlocal]
            
                     cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + kelem[ilocal,jlocal]             
                     cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + melem[ilocal,jlocal]
                     
                 for jlocal in range(0,3):
                     jglobal = mesh.IEN[e,jlocal]
                     
                     if ilocal <=2:                 
                         cls.M_T[iglobal,jglobal] = cls.M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                         cls.K_T[iglobal,jglobal] = cls.K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                         cls.Gx_T[iglobal,jglobal] = cls.Gx_T[iglobal,jglobal] + gx_lin[ilocal,jlocal]
                         cls.Gy_T[iglobal,jglobal] = cls.Gy_T[iglobal,jglobal] + gy_lin[ilocal,jlocal]
                         
                     cls.Gx[iglobal,jglobal] = cls.Gx[iglobal,jglobal] + gx[ilocal,jlocal]
                     cls.Gy[iglobal,jglobal] = cls.Gy[iglobal,jglobal] + gy[ilocal,jlocal]
             
        cls.Dx = np.transpose(cls.Gx)
        cls.Dy = np.transpose(cls.Gy)
        
        cls.set_block_matrices(BC)
        
        # start = timer()
        # solveSys.createMatrixFile(cls.Matriz)
        # end = timer()
        # print('time = ' + str(end-start))
        
    @classmethod
    def set_block_matrices(cls,BC):
        cls.Matriz_vx = np.block([[(1.0/cls.dt)*cls.M + (1.0/cls.Re)*cls.K,np.zeros( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' ), -cls.Gx]])
        cls.Matriz_vy = np.block([[np.zeros( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' ),(1.0/cls.dt)*cls.M + (1.0/cls.Re)*cls.K, -cls.Gy]])
        cls.Matriz_p =  np.block([[cls.Dx, cls.Dy, np.zeros( (cls.mesh.npoints_p,cls.mesh.npoints_p),dtype='float' )]])
        cls.Matriz_T=(1.0/cls.dt)*cls.M_T + (1.0/cls.Pr)*cls.K_T
        
        for i in range(len(BC)):
            
            for j in cls.mesh.boundary[i]:
                if BC[i]['vx'] != 'None':
                    cls.Matriz_vx[j,:] = 0
                    cls.Matriz_vx[j,j] = 1.0
                
                if BC[i]['vy'] != 'None':
                    cls.Matriz_vy[j,:] = 0
                    cls.Matriz_vy[j,j+cls.mesh.npoints] = 1.0
                    
                if BC[i]['T'] != 'None':
                    cls.Matriz_T[j,:] = 0
                    cls.Matriz_T[j,j] = 1.0
                    
                if BC[i]['p'] != 'None':  
                    cls.Matriz_p[j,:] = 0
                    cls.Matriz_p[j,j+2*cls.mesh.npoints] = 1.0
       
        cls.Matriz = np.block([[cls.Matriz_vx],[cls.Matriz_vy],[cls.Matriz_p]])
        
    @classmethod
    def set_block_vectors(cls,BC):
        cls.vetor_vx = np.dot((1.0/cls.dt)*cls.M,cls.fluid.vxd)
        cls.vetor_vy = np.dot((1.0/cls.dt)*cls.M,cls.fluid.vyd) + np.dot(cls.M,cls.fluid.Gr*cls.fluid.T_mini-cls.fluid.Ga)
        cls.vetor_p = np.zeros((cls.mesh.npoints_p),dtype='float' )
        cls.vetor_T = np.dot((1.0/cls.dt)*cls.M_T,cls.fluid.Td[0:cls.mesh.npoints_p])
        
        for i in range(len(BC)):
            
            for j in cls.mesh.boundary[i]:
                if BC[i]['vx'] != 'None':
                    cls.vetor_vx[j] = float(BC[i]['vx'])
                if BC[i]['vy'] != 'None':
                    cls.vetor_vy[j] = float(BC[i]['vy'])
                if BC[i]['T'] != 'None':
                    cls.vetor_T[j] = float(BC[i]['T'])
                if BC[i]['p'] != 'None':
                    cls.vetor_p[j] = float(BC[i]['p'])
        
        
        cls.vetor = np.block([cls.vetor_vx,cls.vetor_vy,cls.vetor_p])
        
    
    @classmethod
    def solve_fields(cls):
        
        start = timer()
        cls.fluid.vxd, cls.fluid.vyd, cls.fluid.Td = semi_lagrange2(cls.mesh.node_list,cls.mesh.elem_list,cls.fluid.vx,cls.fluid.vy,cls.fluid.T,cls.dt,cls.mesh.IENbound)
        end = timer()
        print('time SL= ' + str(end-start))
        
        cls.fluid.T_mini[0:cls.mesh.npoints_p] = cls.fluid.T
        for e in cls.mesh.IEN:
            v1,v2,v3,v4 = e
            cls.fluid.T_mini[v4] = (cls.fluid.T[v1] + cls.fluid.T[v2] + cls.fluid.T[v3])/3.0
            
        cls.set_block_vectors(cls.BC)
        
        start = timer()
        sol = np.linalg.solve(cls.Matriz,cls.vetor)
        # sol = solveSys.solveSystem(cls.vetor)
        # sol = Main.solve(cls.Matriz,cls.vetor)
        # sol = MathWrapper.linSolve(cls.Matriz,cls.vetor,cls.dll)
        end = timer()
        
        print('time solution python= ' + str(end-start))
        
        cls.fluid.vx = sol[0:cls.mesh.npoints]
        cls.fluid.vy = sol[cls.mesh.npoints:2*cls.mesh.npoints]
        cls.fluid.p = sol[2*cls.mesh.npoints:]
        
        cls.fluid.T = np.linalg.solve(cls.Matriz_T,cls.vetor_T)
        # cls.fluid.T = MathWrapper.linSolve(cls.Matriz_T,cls.vetor_T,cls.dll)
        
        return cls.fluid
    
   
        
            

