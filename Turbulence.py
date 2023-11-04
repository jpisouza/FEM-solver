import numpy as np
from timeit import default_timer as timer
import scipy as sp
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, dok_matrix
import scipy.linalg  
import scipy.sparse.linalg 
import Elements
import SL

class Turbulence:
    
    @classmethod
    def set_model(cls,solver):
        cls.solver = solver
        cls.C_mu = 0.09
        cls.C_e1 = 1.44
        cls.C_e2 = 1.92
        cls.sigma_k = 1.0
        cls.sigma_e = 1.3
        
    @classmethod
    def calc_turb(cls, neighborElem, oface):
        cls.calc_system(neighborElem, oface)
        cls.solver.fluid.nu_t = 0.1*np.ones((cls.solver.mesh.npoints), dtype='float') 
      
    @classmethod
    def calc_system(cls,neighborElem, oface):
        cls.calc_SL(neighborElem, oface)
        P = cls.calc_production()
        cls.Matriz = cls.solver.M.tocsr() + sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.solver.fluid.nu_t),cls.solver.K)
        cls.Matriz = cls.Matriz.tocsr()
        cls.vector = sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.solver.fluid.nu_t),sp.sparse.csr_matrix.dot(cls.solver.M.tocsr(),P))+\
            sp.sparse.csr_matrix.dot(cls.solver.M.tocsr(),cls.solver.fluid.kappad)
        
     
    @classmethod
    def calc_production(cls):
        dudx, dudy, dvdx, dvdy = cls.calc_derivatives()
        
        dudx2 = dudx**2
        dudy2 = dudy**2
        dvdx2 = dvdx**2
        dvdy2 = dvdy**2
        
        dudydvdx = dudy*dvdx
        
        P = 2*dudx2 + dudy2 + dvdx2 + 2*dvdy2 + 2*dudydvdx
        
        return P
        
     
    @classmethod
    def calc_derivatives(cls):
        dudx = sp.sparse.linalg.spsolve(cls.solver.M,sp.sparse.csr_matrix.dot(cls.solver.Gvx,cls.solver.fluid.vx))
        dudy = sp.sparse.linalg.spsolve(cls.solver.M,sp.sparse.csr_matrix.dot(cls.solver.Gvy,cls.solver.fluid.vx))
        
        dvdx = sp.sparse.linalg.spsolve(cls.solver.M,sp.sparse.csr_matrix.dot(cls.solver.Gvx,cls.solver.fluid.vy))
        dvdy = sp.sparse.linalg.spsolve(cls.solver.M,sp.sparse.csr_matrix.dot(cls.solver.Gvy,cls.solver.fluid.vy))
        
        return dudx, dudy, dvdx, dvdy
    
    @classmethod
    def calc_SL(cls, neighborElem, oface):
        if cls.solver.mesh.IEN.shape[1]==4:
            sl = SL.Linear(cls.solver.mesh.IEN,cls.solver.mesh.X,cls.solver.mesh.Y,neighborElem,oface,cls.solver.fluid.vx,cls.solver.fluid.vy)
            sl.compute(cls.solver.dt)
            cls.solver.fluid.kappad[0:cls.solver.mesh.npoints_p] = sl.conv*cls.solver.fluid.kappa[0:cls.solver.mesh.npoints_p]
            #Calculating for the centroid
            for e in range(cls.solver.mesh.IEN.shape[1]):
                [v1,v2,v3,v4] = cls.solver.mesh.IEN[e]
                cls.solver.fluid.kappad[v4] = (cls.solver.fluid.kappad[v1] + cls.solver.fluid.kappad[v2] + cls.solver.fluid.kappad[v3])/3.0
        else:
            sl = SL.Quad(cls.solver.mesh.IEN,cls.solver.mesh.X,cls.solver.mesh.Y,neighborElem,oface,cls.solver.fluid.vx,cls.solver.fluid.vy)
            sl.compute(cls.solver.dt)
            cls.solver.fluid.kappad = sl.conv*cls.solver.fluid.kappad
    
         
        

