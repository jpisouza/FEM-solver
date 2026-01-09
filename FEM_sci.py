import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, dok_matrix
import scipy.linalg  
import scipy.sparse.linalg 
# import MathWrapper
from semi_lagrangiano import semi_lagrange2
import SL
import clSemiLagrangian as SL_
import Elements
import clElement
from Turbulence import Turbulence
from FSISolidMesh import FSISolidMesh
from SolidFEM import FEM as SolidFEM
from multiprocessing import cpu_count
# import solveSys
# from julia import Main



class FEM:
    
    @classmethod
    def set_matrices(cls,mesh,fluid,dt,BC,IC,SolidProp,COO_flag = False,porous = False, turb = False):

        cls.fluid = fluid
        cls.Re = fluid.Re
        cls.Ga = fluid.Ga
        cls.Gr = fluid.Gr
        cls.Pr = fluid.Pr
        cls.Da = fluid.Da
        cls.Fo = fluid.Fo
        cls.dt = dt
        cls.mesh = mesh
        cls.BC = BC
        cls.turb = turb
        cls.porous = porous
        cls.COO_flag = COO_flag
        cls.SolidProp = SolidProp
        cls.IC = IC
        cls.int_i = 0 #internal count
        
        if len(cls.mesh.porous_elem)>0:
            cls.porous = True
        
        cls.fluid.FSIForces = np.zeros((cls.mesh.npoints,2), dtype='float')
                
        if len(cls.mesh.FSI) > 0:
            cls.solidMesh = FSISolidMesh(cls.mesh, cls.fluid)
            cls.BC_solid = cls.solidMesh.build_BCdict(cls.fluid.FSIForces)
            SolidFEM.set_parameters(cls.solidMesh, cls.BC_solid, cls.IC, float(SolidProp['h']), float(SolidProp['E']), dt, float(SolidProp['rho']), float(SolidProp['nu']), float(SolidProp['gamma']), float(SolidProp['beta']), [0,0], 1.0/cls.Ga**2)
        
        if len(cls.mesh.FSI) == 0:
            start = timer()
            if cls.mesh.mesh_kind == 'mini':
                cls.build_mini()
            elif cls.mesh.mesh_kind == 'quad':
                if cls.COO_flag:
                    cls.build_quad_GQ_COO()
                else:
                    cls.build_quad_GQ()
       
            cls.set_block_matrices(BC)
        
            end = timer()
            print('time --> Build FEM matrices = ' + str(end-start) + ' [s]')

    @classmethod
    def build_quad_GQ(cls):
        cls.K = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Kx = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Ky = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Kxy = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Kyx = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Gvx = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Gvy = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )

        cls.Gx = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints_p),dtype='float' )
        cls.Gy = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints_p),dtype='float' )
        cls.Dx = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints),dtype='float' )
        cls.Dy = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints),dtype='float' )
        
        cls.M_porous = lil_matrix((cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_porous_elem = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_porous_elem_F = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_T = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.K_T = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
            
        quad = Elements.Quad(cls.mesh.X,cls.mesh.Y)
        # lin = Elements.Linear(cls.mesh.X[list(cls.mesh.converter.keys())],cls.mesh.Y[list(cls.mesh.converter.keys())])
        
        # loop de elementos finitos
        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4,v5,v6 = cls.mesh.IEN[e]
            quad.getM([v1,v2,v3,v4,v5,v6])
            # lin.getM([cls.mesh.converter[v1],cls.mesh.converter[v2],cls.mesh.converter[v3]])
            kx_elem = quad.kxx
            ky_elem = quad.kyy
            k_elem = quad.kxx + quad.kyy
            kxy_elem = quad.kxy
            melem = quad.mass
            gvx_elem = quad.gvx
            gvy_elem = quad.gvy

            kelem_T = quad.kxx + quad.kyy
            melem_T = quad.mass
            for ilocal in range(0,6):
                  iglobal = cls.mesh.IEN[e,ilocal]
                  for jlocal in range(0,6):
                      jglobal = cls.mesh.IEN[e,jlocal]
                      
                      if e in cls.mesh.solid_elem:
                          cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + 10000*cls.Re*k_elem[ilocal,jlocal]
                      else:
                          cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + k_elem[ilocal,jlocal]
                          
                      cls.Kx[iglobal,jglobal] = cls.Kx[iglobal,jglobal] + kx_elem[ilocal,jlocal] 
                      cls.Ky[iglobal,jglobal] = cls.Ky[iglobal,jglobal] + ky_elem[ilocal,jlocal]
                      cls.Kxy[iglobal,jglobal] = cls.Kxy[iglobal,jglobal] + kxy_elem[ilocal,jlocal]  
                      cls.Gvx[iglobal,jglobal] = cls.Gvx[iglobal,jglobal] + gvx_elem[ilocal,jlocal]
                      cls.Gvy[iglobal,jglobal] = cls.Gvy[iglobal,jglobal] + gvy_elem[ilocal,jlocal]
                      # cls.M_full[iglobal,jglobal] = cls.M_full[iglobal,jglobal] + melem[ilocal,jlocal]
                      cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + melem[ilocal,jlocal]
                      cls.M_T[iglobal,jglobal] = cls.M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                      cls.K_T[iglobal,jglobal] = cls.K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                     
                      if len(cls.mesh.porous_list) > 0:
                          # cls.M_porous[iglobal,jglobal] = cls.M_porous[iglobal,jglobal] + melem_porous[ilocal,jlocal]
                          cls.M_porous_elem[iglobal,jglobal] = cls.M_porous_elem[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]
                          cls.M_porous_elem_F[iglobal,jglobal] = cls.M_porous_elem_F[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]**2
                      else:
                          cls.M_porous_elem[iglobal,jglobal] = cls.M_porous_elem[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porosity_array[e]
                          cls.M_porous_elem_F[iglobal,jglobal] = cls.M_porous_elem_F[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porosity_array[e]**2
                     
                  for jlocal in range(0,3):
                      jglobal = cls.mesh.IEN[e,jlocal]

                      cls.Gx[iglobal,cls.mesh.converter[jglobal]] = cls.Gx[iglobal,cls.mesh.converter[jglobal]] + quad.gx[ilocal,jlocal]
                      cls.Gy[iglobal,cls.mesh.converter[jglobal]] = cls.Gy[iglobal,cls.mesh.converter[jglobal]] + quad.gy[ilocal,jlocal]
                      cls.Dx[cls.mesh.converter[jglobal],iglobal] = cls.Dx[cls.mesh.converter[jglobal],iglobal] + quad.dx[jlocal,ilocal]
                      cls.Dy[cls.mesh.converter[jglobal],iglobal] = cls.Dy[cls.mesh.converter[jglobal],iglobal] + quad.dy[jlocal,ilocal]


    @classmethod
    def build_quad_GQ_COO(cls):
        dofVel, dofP, ngauss = 6, 3, 12 # degrees of fredom of velocity/pressure
        element_class = clElement.Triangle
        
        vel_nodes  = cls.mesh.IEN.astype(np.int32)       # (numElems, 6)
        pres_nodes = [[cls.mesh.converter[n] for n in line_IEN] for line_IEN in cls.mesh.IEN[:,0:dofP].astype(np.int32)]  # (numVerts, 4)
       
        iv  = np.repeat(vel_nodes, dofVel, axis=1).ravel()  # VV — linha (nó)
        jv  = np.tile   (vel_nodes, (1, dofVel)).ravel()    # VV — coluna (nó)
       
        ip  = np.repeat(vel_nodes, dofP,  axis=1).ravel()   # VP — linha (nó)
        jp  = np.tile   (pres_nodes, (1, dofVel)).ravel()   # VP — coluna (nó‑p)
       
        ipd = np.repeat(pres_nodes, dofVel, axis=1).ravel() # PV — linha (nó‑p)
        jpd = np.tile   (vel_nodes, (1, dofP )).ravel()     # PV — coluna (nó)
        
        n_vv = dofVel * dofVel * cls.mesh.ne          # 6×6 valores por elemento
        n_vp = dofVel * dofP   * cls.mesh.ne          # 6×3 valores por elemento
       
        m_data   = np.empty(n_vv, dtype=np.float64)
        kx_data = np.empty_like(m_data)
        ky_data = np.empty_like(m_data)
        k_data = np.empty_like(m_data)
        gvx_data  = np.empty_like(m_data)
        gvy_data  = np.empty_like(m_data)
       
        gx_data  = np.empty(n_vp, dtype=np.float64)
        gy_data  = np.empty_like(gx_data)
        dx_data  = np.empty_like(gx_data)
        dy_data  = np.empty_like(gx_data)
        
        offset_vv = dofVel * dofVel          # 6x6 = 36
        offset_vp = dofVel * dofP            # 6x3 = 18
        
        # Call the parallel engine:
        nprocs = max(1, cpu_count() - 1)
        local = clElement.parallel_get_matrices(
           element_class=element_class,# Triangle, Tetrahedron, Line...
           constructor_args=(cls.mesh.X, cls.mesh.Y, dofVel,dofP,ngauss),
           v_list=[cls.mesh.IEN[e] for e in range(cls.mesh.ne)],
           processes=nprocs,
           chunksize=max(1, cls.mesh.ne // (8 * nprocs)),
           use_slip=False
        )
        
        for e in range(cls.mesh.ne):
            s_vv = slice(e*offset_vv, (e+1)*offset_vv)
            s_vp = slice(e*offset_vp, (e+1)*offset_vp)
          
            m_data[s_vv]   = local[e].mass.ravel()
            kx_data[s_vv]  = local[e].kxx.ravel()
            ky_data[s_vv]  = local[e].kyy.ravel()
            k_data[s_vv]   = (local[e].kxx + local[e].kyy).ravel()
            gvx_data[s_vv]  = local[e].gvx.ravel()
            gvy_data[s_vv]  = local[e].gvy.ravel()
          
            gx_data[s_vp]  = local[e].gx.ravel()
            gy_data[s_vp]  = local[e].gy.ravel()
            dx_data[s_vp]  = local[e].dx.ravel()
            dy_data[s_vp]  = local[e].dy.ravel()
            
            
        cls.M    = coo_matrix((m_data,  (iv , jv )),shape=(cls.mesh.npoints, cls.mesh.npoints)).tocsr()
        cls.M_T = cls.M.copy()
        cls.Kx    = coo_matrix((kx_data,  (iv , jv )),shape=(cls.mesh.npoints, cls.mesh.npoints)).tocsr()
        cls.Ky    = coo_matrix((ky_data,  (iv , jv )),shape=(cls.mesh.npoints, cls.mesh.npoints)).tocsr()
        cls.K     = cls.Kx + cls.Ky
        cls.K_T = cls.K.copy()
        cls.Gvx   = coo_matrix((gvx_data, (iv , jv )),shape=(cls.mesh.npoints, cls.mesh.npoints)).tocsr()
        cls.Gvy   = coo_matrix((gvy_data, (iv , jv )),shape=(cls.mesh.npoints, cls.mesh.npoints)).tocsr()
        cls.Gx   = coo_matrix((gx_data,(ip , jp)),shape=(cls.mesh.npoints, cls.mesh.npoints_p)).tocsr()
        cls.Gy   = coo_matrix((gy_data,(ip , jp)),shape=(cls.mesh.npoints, cls.mesh.npoints_p)).tocsr()
        cls.Dx   = coo_matrix((dx_data, (ipd, jpd)),shape=(cls.mesh.npoints_p, cls.mesh.npoints)).tocsr()
        cls.Dy   = coo_matrix((dy_data, (ipd, jpd)),shape=(cls.mesh.npoints_p, cls.mesh.npoints)).tocsr()

    @classmethod
    def build_mini_GQ_COO(cls):
        mini = Elements.Mini(cls.mesh.X,cls.mesh.Y)
        lin = Elements.Linear(cls.mesh.X[:cls.mesh.npoints_p],cls.mesh.Y[:cls.mesh.npoints_p])
        
        rows = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        cols = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        rows_p = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        cols_p = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        rows_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        cols_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        
        M = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        K = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        M_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        K_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        Gx = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        Gy = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        M_porous_elem = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        M_porous_elem_F = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        
        npoints_elem = cls.mesh.IEN.shape[1]
        npoints_elem_p = cls.mesh.IEN_orig.shape[1]

        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4 = cls.mesh.IEN[e]
            mini.getM([v1,v2,v3,v4])
            kelem = mini.kxx + mini.kyy
            kelem_T = lin.kxx + lin.kyy
            melem = mini.mass
            melem_T = lin.mass
            gxelem = mini.gx
            gyelem = mini.gy
            melem_flat = melem.flatten()
            melem_T_flat = melem_T.flatten()
            kelem_flat = kelem.flatten()
            kelem_T_flat = kelem_T.flatten()
            gxelem_flat = gxelem.flatten()
            gyelem_flat = gyelem.flatten()
   
            rows[e*npoints_elem**2:(e+1)*npoints_elem**2] = npoints_elem*[v1] + npoints_elem*[v2] + npoints_elem*[v3] + npoints_elem*[v4]
            cols[e*npoints_elem**2:(e+1)*npoints_elem**2] = npoints_elem*[v1,v2,v3,v4]
            rows_p[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = npoints_elem_p*[v1] + npoints_elem_p*[v2] + npoints_elem_p*[v3] + npoints_elem_p*[v4] 
            cols_p[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = npoints_elem*[v1,v2,v3]
            rows_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = npoints_elem_p*[v1] + npoints_elem_p*[v2] + npoints_elem_p*[v3]
            cols_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = npoints_elem_p*[v1,v2,v3]
             
            Gx[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = gxelem_flat
            Gy[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = gyelem_flat
            
            M[e*npoints_elem**2:(e+1)*npoints_elem**2] = melem_flat
            M_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = melem_T_flat
            K[e*npoints_elem**2:(e+1)*npoints_elem**2] = kelem_flat
            K_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = kelem_T_flat
            
            if len(cls.mesh.porous_list) > 0:
                M_porous_elem[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e])
                M_porous_elem_F[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]**2)
            else:
                M_porous_elem[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porosity_array[e])
                M_porous_elem_F[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porosity_array[e]**2)
                
   
        cls.M = coo_matrix((M,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.M_T = coo_matrix((M_T,(rows_T,cols_T)), shape = (cls.mesh.npoints_p,cls.mesh.npoints_p)).tolil()
        cls.K = coo_matrix((K,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.K_T = coo_matrix((K_T,(rows_T,cols_T)), shape = (cls.mesh.npoints_p,cls.mesh.npoints_p)).tolil()
        cls.M_porous_elem = coo_matrix((M_porous_elem,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.M_porous_elem_F = coo_matrix((M_porous_elem_F,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.Gx = coo_matrix((Gx,(rows_p,cols_p)), shape = (cls.mesh.npoints,cls.mesh.npoints_p)).tolil()
        cls.Gy = coo_matrix((Gy,(rows_p,cols_p)), shape = (cls.mesh.npoints,cls.mesh.npoints_p)).tolil()
        cls.Dx = cls.Gx.transpose().tolil()
        cls.Dy = cls.Gy.transpose().tolil()
        
    @classmethod
    def build_mini_GQ(cls):
        
        cls.K = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        # cls.M_full = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Gx = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints_p),dtype='float' )
        cls.Gy = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints_p),dtype='float' )
        cls.Dx = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints),dtype='float' )
        cls.Dy = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints),dtype='float' )
        
        cls.M_porous = lil_matrix((cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_porous_elem = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_porous_elem_F = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        # cls.M_forchheimer = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_T = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints_p),dtype='float' )
        cls.K_T = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints_p),dtype='float' )
            
        mini = Elements.Mini(cls.mesh.X,cls.mesh.Y)
        lin = Elements.Linear(cls.mesh.X[:cls.mesh.npoints_p],cls.mesh.Y[:cls.mesh.npoints_p])
        # loop de elementos finitos
        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4 = cls.mesh.IEN[e]
            mini.getM([v1,v2,v3,v4])
            lin.getM([v1,v2,v3])
            kelem = mini.kxx + mini.kyy
            melem = mini.mass
            
            # if len(cls.mesh.porous_list) > 0:
            #     mini.getMass_porous([v1,v2,v3,v4], cls.mesh.porous_nodes)
            #     melem_porous = mini.mass_porous
                
            kelem_T = lin.kxx + lin.kyy
            melem_T = lin.mass
            for ilocal in range(0,4):
                 iglobal = cls.mesh.IEN[e,ilocal]
                 for jlocal in range(0,4):
                     jglobal = cls.mesh.IEN[e,jlocal]
            
                     cls.K[iglobal,jglobal] = cls.K[iglobal,jglobal] + kelem[ilocal,jlocal]           
                     # cls.M_full[iglobal,jglobal] = cls.M_full[iglobal,jglobal] + melem[ilocal,jlocal]
                     cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + melem[ilocal,jlocal]
                     if len(cls.mesh.porous_list) > 0:
                         # cls.M_porous[iglobal,jglobal] = cls.M_porous[iglobal,jglobal] + melem_porous[ilocal,jlocal]
                         cls.M_porous_elem[iglobal,jglobal] = cls.M_porous_elem[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]
                         cls.M_porous_elem_F[iglobal,jglobal] = cls.M_porous_elem_F[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]**2
                     else:
                         cls.M_porous_elem[iglobal,jglobal] = cls.M_porous_elem[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porosity_array[e]
                         cls.M_porous_elem_F[iglobal,jglobal] = cls.M_porous_elem_F[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porosity_array[e]**2
                     
                 for jlocal in range(0,3):
                     jglobal = cls.mesh.IEN[e,jlocal]
                     
                     if ilocal <=2:                 
                         cls.M_T[iglobal,jglobal] = cls.M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                         cls.K_T[iglobal,jglobal] = cls.K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                         
                     cls.Gx[iglobal,jglobal] = cls.Gx[iglobal,jglobal] + mini.gx[ilocal,jlocal]
                     cls.Gy[iglobal,jglobal] = cls.Gy[iglobal,jglobal] + mini.gy[ilocal,jlocal]

        cls.Dx = cls.Gx.transpose()
        cls.Dy = cls.Gy.transpose()

    @classmethod
    def build_mini_COO(cls):
   
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
             
        rows = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        cols = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        rows_p = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        cols_p = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        rows_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        cols_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        
        M = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        K = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        M_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        K_T = np.zeros((cls.mesh.ne*cls.mesh.IEN_orig.shape[1]**2), dtype='float')
        Gx = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        Gy = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]*cls.mesh.IEN_orig.shape[1]), dtype='float')
        M_porous_elem = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        M_porous_elem_F = np.zeros((cls.mesh.ne*cls.mesh.IEN.shape[1]**2), dtype='float')
        
        npoints_elem = cls.mesh.IEN.shape[1]
        npoints_elem_p = cls.mesh.IEN_orig.shape[1]

        for e in range(0,cls.mesh.ne):
            v1,v2,v3,v4 = cls.mesh.IEN[e]
            melem_flat = melem.flatten()
            melem_T_flat = melem_T.flatten()
            kelem_flat = kelem.flatten()
            kelem_T_flat = kelem_T.flatten()
            gxelem_flat = gx.flatten()
            gyelem_flat = gy.flatten()
   
            rows[e*npoints_elem**2:(e+1)*npoints_elem**2] = npoints_elem*[v1] + npoints_elem*[v2] + npoints_elem*[v3] + npoints_elem*[v4]
            cols[e*npoints_elem**2:(e+1)*npoints_elem**2] = npoints_elem*[v1,v2,v3,v4]
            rows_p[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = npoints_elem_p*[v1] + npoints_elem_p*[v2] + npoints_elem_p*[v3] + npoints_elem_p*[v4] 
            cols_p[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = npoints_elem*[v1,v2,v3]
            rows_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = npoints_elem_p*[v1] + npoints_elem_p*[v2] + npoints_elem_p*[v3]
            cols_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = npoints_elem_p*[v1,v2,v3]
             
            Gx[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = gxelem_flat
            Gy[e*npoints_elem*npoints_elem_p:(e+1)*npoints_elem*npoints_elem_p] = gyelem_flat
            
            M[e*npoints_elem**2:(e+1)*npoints_elem**2] = melem_flat
            M_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = melem_T_flat
            K[e*npoints_elem**2:(e+1)*npoints_elem**2] = kelem_flat
            K_T[e*npoints_elem_p**2:(e+1)*npoints_elem_p**2] = kelem_T_flat
            
            if len(cls.mesh.porous_list) > 0:
                M_porous_elem[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e])
                M_porous_elem_F[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]**2)
            else:
                M_porous_elem[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porosity_array[e])
                M_porous_elem_F[e*npoints_elem**2:(e+1)*npoints_elem**2] = (melem_flat*cls.mesh.porosity_array[e]**2)
                
   
        cls.M = coo_matrix((M,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.M_T = coo_matrix((M_T,(rows_T,cols_T)), shape = (cls.mesh.npoints_p,cls.mesh.npoints_p)).tolil()
        cls.K = coo_matrix((K,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.K_T = coo_matrix((K_T,(rows_T,cols_T)), shape = (cls.mesh.npoints_p,cls.mesh.npoints_p)).tolil()
        cls.M_porous_elem = coo_matrix((M_porous_elem,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.M_porous_elem_F = coo_matrix((M_porous_elem_F,(rows,cols)), shape = (cls.mesh.npoints,cls.mesh.npoints)).tolil()
        cls.Gx = coo_matrix((Gx,(rows_p,cols_p)), shape = (cls.mesh.npoints,cls.mesh.npoints_p)).tolil()
        cls.Gy = coo_matrix((Gy,(rows_p,cols_p)), shape = (cls.mesh.npoints,cls.mesh.npoints_p)).tolil()
        cls.Dx = cls.Gx.transpose().tolil()
        cls.Dy = cls.Gy.transpose().tolil()
             
             
    @classmethod
    def build_mini(cls):
        cls.K = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        # cls.M_full = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.Gx = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints_p),dtype='float' )
        cls.Gy = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints_p),dtype='float' )
        cls.Dx = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints),dtype='float' )
        cls.Dy = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints),dtype='float' )
        
        cls.M_porous = lil_matrix((cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_porous_elem = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        cls.M_porous_elem_F = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )
        # cls.M_forchheimer = lil_matrix( (cls.mesh.npoints,cls.mesh.npoints),dtype='float' )

        cls.M_T = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints_p),dtype='float' )
        cls.K_T = lil_matrix( (cls.mesh.npoints_p,cls.mesh.npoints_p),dtype='float' )

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
                     # cls.M_full[iglobal,jglobal] = cls.M_full[iglobal,jglobal] + melem[ilocal,jlocal]
                     cls.M[iglobal,jglobal] = cls.M[iglobal,jglobal] + melem[ilocal,jlocal]
                     if len(cls.mesh.porous_list) > 0:
                         # cls.M_porous[iglobal,jglobal] = cls.M_porous[iglobal,jglobal] + melem_porous[ilocal,jlocal]
                         cls.M_porous_elem[iglobal,jglobal] = cls.M_porous_elem[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]
                         cls.M_porous_elem_F[iglobal,jglobal] = cls.M_porous_elem_F[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porous_elem[e]*cls.mesh.porosity_array[e]**2
                     else:
                         cls.M_porous_elem[iglobal,jglobal] = cls.M_porous_elem[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porosity_array[e]
                         cls.M_porous_elem_F[iglobal,jglobal] = cls.M_porous_elem_F[iglobal,jglobal] + melem[ilocal,jlocal]*cls.mesh.porosity_array[e]**2
                     
                 for jlocal in range(0,3):
                     jglobal = cls.mesh.IEN[e,jlocal]
                     
                     if ilocal <=2:                 
                         cls.M_T[iglobal,jglobal] = cls.M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                         cls.K_T[iglobal,jglobal] = cls.K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                         
                     cls.Gx[iglobal,jglobal] = cls.Gx[iglobal,jglobal] + gx[ilocal,jlocal]
                     cls.Gy[iglobal,jglobal] = cls.Gy[iglobal,jglobal] + gy[ilocal,jlocal]

        cls.Dx = cls.Gx.transpose()
        cls.Dy = cls.Gy.transpose()
        
    @classmethod
    def set_Turb(cls):
        
        Turbulence.calc_turb()
        
        A11 = sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.nu_t),2.0*cls.Kx + cls.Ky)
        A22 = sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.nu_t),cls.Kx + 2.0*cls.Ky)
        A12 = sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.nu_t),cls.Kxy)
        
        block_Turb = sp.sparse.bmat([ [ A11, A12, sp.sparse.csr_matrix((cls.mesh.npoints, cls.mesh.npoints_p), dtype= 'float')],
                                      [ A12, A22, None],
                                      [ None, None, sp.sparse.csr_matrix((cls.mesh.npoints_p, cls.mesh.npoints_p), dtype= 'float')]])
        
        if cls.porous:
            cls.Matriz = cls.Matriz + block_Turb #in order to account set_Darcy_Forchheimer modifications in the main matrix
        else:
            cls.Matriz = cls.Matriz_orig + block_Turb 
        
    
    @classmethod
    def set_Darcy_Forchheimer(cls):
        

        if len(cls.mesh.porous_list) > 0:
                     
            
            v_diag = sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.vx),sp.sparse.diags(cls.fluid.vx)) + sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.vy),sp.sparse.diags(cls.fluid.vy))       
            v_diag = v_diag.power(0.5)
            
            A1 = (1.0/(cls.Re*cls.Da))*cls.M_porous_elem.tocsr() + (cls.Fo/(cls.Re*cls.Da))*sp.sparse.csr_matrix.dot(v_diag,cls.M_porous_elem_F.tocsr())# + (cls.Fo/(cls.Re*cls.Da))*cls.M_forchheimer
            
            block_DF = sp.sparse.bmat([[A1, None,sp.sparse.csr_matrix((cls.mesh.npoints, cls.mesh.npoints_p), dtype= 'float')],
                                       [None, A1,None],
                                       [None,None,sp.sparse.csr_matrix((cls.mesh.npoints_p, cls.mesh.npoints_p), dtype= 'float')]])
         
        
        
        else:
            v_diag = sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.vx),sp.sparse.diags(cls.fluid.vx)) + sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.vy),sp.sparse.diags(cls.fluid.vy))
        
            v_diag = v_diag.power(0.5)
            A1 = (1.0/(cls.Re*cls.Da))*cls.M_porous_elem.tocsr() + (cls.Fo/(cls.Re*cls.Da))*sp.sparse.csr_matrix.dot(v_diag,cls.M_porous_elem_F.tocsr())
            
            block_DF = sp.sparse.bmat([[A1, None,sp.sparse.csr_matrix((cls.mesh.npoints, cls.mesh.npoints_p), dtype= 'float')],
                                       [None, A1,None],
                                       [None,None,sp.sparse.csr_matrix((cls.mesh.npoints_p, cls.mesh.npoints_p), dtype= 'float')]])
         

        cls.Matriz = cls.Matriz_orig + block_DF
    
    @classmethod
    def set_Darcy_Forchheimer_(cls):
        
        v_diag = sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.vx),sp.sparse.diags(cls.fluid.vx)) + sp.sparse.csr_matrix.dot(sp.sparse.diags(cls.fluid.vy),sp.sparse.diags(cls.fluid.vy))
        v_diag = v_diag.power(0.5)
        
        A1 = (1.0/(cls.Re*cls.Da))*cls.M.tocsr() + (cls.Fo/(cls.Re*cls.Da))*sp.sparse.csr_matrix.dot(v_diag,cls.M.tocsr())
        
        block_DF = sp.sparse.bmat([[A1, None,sp.sparse.csr_matrix((cls.mesh.npoints, cls.mesh.npoints_p), dtype= 'float')],
                                   [None, A1,None],
                                   [None,None,sp.sparse.csr_matrix((cls.mesh.npoints_p, cls.mesh.npoints_p), dtype= 'float')]])
         
        
        if len(cls.mesh.porous_list) > 0:
            A2 = sp.sparse.diags(cls.mesh.porous_nodes)
            block_porous = sp.sparse.bmat([[A2, None,sp.sparse.csr_matrix((cls.mesh.npoints, cls.mesh.npoints_p), dtype= 'float')],
                                           [None, A2,None],
                                           [None,None,sp.sparse.csr_matrix((cls.mesh.npoints_p, cls.mesh.npoints_p), dtype= 'float')]])
         
            cls.Matriz = cls.Matriz_orig + sp.sparse.csr_matrix.dot(block_porous,block_DF)
        
        else:
            cls.Matriz = cls.Matriz_orig + block_DF    
        
        
    @classmethod
    def calc_v_modulus(cls):
        v = np.zeros((len(cls.fluid.vx)), dtype='float')
        for i in range (len(cls.fluid.vx)):
            v[i] = np.sqrt(cls.fluid.vx[i]**2 + cls.fluid.vy[i]**2)
              
        return v
        
    @classmethod
    def set_block_matrices(cls,BC):
        
        # A1 = (1.0/cls.dt)*cls.M + (1.0/cls.Re)*cls.K 
        
        # cls.Matriz = sp.sparse.bmat([ [ A1, None, -cls.Gx],
        #                               [ None, A1, -cls.Gy],
        #                               [ cls.Dx, cls.Dy, None]])
        
        A11 = (1.0/cls.dt)*cls.M + (1.0/cls.Re)*cls.K
        A22 = A11
        A12 = None
        
        cls.Matriz = sp.sparse.bmat([ [ A11, A12, -cls.Gx],
                                      [ A12, A22, -cls.Gy],
                                      [ cls.Dx, cls.Dy, None]])
        
        cls.Matriz_T=(1.0/cls.dt)*cls.M_T + (1.0/(cls.Re*cls.Pr))*cls.K_T
        
        cls.Matriz = cls.Matriz.tocsr()
        cls.Matriz_orig = cls.Matriz.copy()
        
        cls.Matriz_T = cls.Matriz_T.tocsr()
        
                       
        if not cls.porous and not cls.turb and not len(cls.mesh.FSI) > 0:
                
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
                    
                    if cls.mesh.mesh_kind == 'quad':
                        if BC[i]['T'] != 'None':
                            row = cls.Matriz_T.getrow(j)
                            for col in row.indices:
                                cls.Matriz_T[j,col] = 0
                            cls.Matriz_T[j,j] = 1.0
                      
                        if j in cls.mesh.IEN[:,:3]:
                            if BC[i]['p'] != 'None':
                                row = cls.Matriz.getrow(cls.mesh.converter[j] + 2*cls.mesh.npoints)
                                for col in row.indices:
                                    cls.Matriz[cls.mesh.converter[j] + 2*cls.mesh.npoints,col] = 0
                                cls.Matriz = cls.Matriz.tolil()
                                cls.Matriz[cls.mesh.converter[j] + 2*cls.mesh.npoints,cls.mesh.converter[j] + 2*cls.mesh.npoints] = 1.0
                                cls.Matriz = cls.Matriz.tocsr()
                            
                    elif cls.mesh.mesh_kind == 'mini':
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
    def set_block_vectors(cls,forces):
        cls.M = cls.M.tocsr()
        # cls.M_full = cls.M_full.tocsr()
        cls.M_T = cls.M_T.tocsr()
        cls.vetor_vx = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M,cls.fluid.vxd) + sp.sparse.csr_matrix.dot(cls.M,forces[:,0])

        if cls.mesh.mesh_kind == 'mini':
            cls.vetor_vy = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M,cls.fluid.vyd) + sp.sparse.csr_matrix.dot(cls.M,cls.fluid.Gr*cls.fluid.T_mini + forces[:,1] -cls.fluid.Ga)
            cls.vetor_T = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M_T,cls.fluid.Td[0:cls.mesh.npoints_p])
        elif cls.mesh.mesh_kind == 'quad':
            cls.vetor_vy = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M,cls.fluid.vyd) + sp.sparse.csr_matrix.dot(cls.M,cls.fluid.Gr*cls.fluid.T + forces[:,1] -cls.fluid.Ga)
            cls.vetor_T = sp.sparse.csr_matrix.dot((1.0/cls.dt)*cls.M_T,cls.fluid.Td)
        cls.vetor_p = np.zeros((cls.mesh.npoints_p),dtype='float' )
        

    @classmethod
    def set_BC(cls,BC):
                   
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
                    
                if cls.mesh.mesh_kind == 'quad':
                    if BC[i]['T'] != 'None':
                        cls.vetor_T[j] = float(BC[i]['T'])
                    
                    if j in cls.mesh.IEN[:,:3]:
                        if BC[i]['p'] != 'None':
                            cls.vetor_p[cls.mesh.converter[j]] = float(BC[i]['p'])
                
                elif cls.mesh.mesh_kind == 'mini':
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
    def set_BC_dynamic(cls,BC):
        for i in range(len(BC)):
            
            for j in cls.mesh.boundary[i]:
                if 'Profile' in BC[i]['vx']:
                    row = cls.Matriz.getrow(j)
                    for col in row.indices:
                        cls.Matriz[j,col] = 0
                    cls.Matriz[j,j] = 1.0
                    
                    u_0 = float(BC[i]['vx'].split('-')[1])
                    ref = (np.min(cls.mesh.Y[cls.mesh.boundary[i]]) + np.max(cls.mesh.Y[cls.mesh.boundary[i]]))/2.0
                    h = np.abs(np.min(cls.mesh.Y[cls.mesh.boundary[i]]) - np.max(cls.mesh.Y[cls.mesh.boundary[i]]))
                    cls.vetor_vx[j] = 1.5*u_0*(1.0 - (2.0*(cls.mesh.Y[j]-ref)/h)**2)
                elif BC[i]['vx'] != 'None':
                    row = cls.Matriz.getrow(j)
                    for col in row.indices:
                        cls.Matriz[j,col] = 0
                    cls.Matriz[j,j] = 1.0
                    
                    if len(cls.mesh.FSI) > 0 and BC[i] in cls.mesh.FSI:
                        cls.vetor_vx[j] = cls.fluid.vx[j]
                    else:
                        cls.vetor_vx[j] = float(BC[i]['vx'])

                if 'Profile' in BC[i]['vy']:
                    row = cls.Matriz.getrow(j + cls.mesh.npoints)
                    for col in row.indices:
                        cls.Matriz[j + cls.mesh.npoints,col] = 0
                    cls.Matriz[j + cls.mesh.npoints,j + cls.mesh.npoints] = 1.0
                    
                    u_0 = float(BC[i]['vy'].split('-')[1])
                    ref = (np.min(cls.mesh.X[cls.mesh.boundary[i]]) + np.max(cls.mesh.X[cls.mesh.boundary[i]]))/2.0
                    h = np.abs(np.min(cls.mesh.X[cls.mesh.boundary[i]]) - np.max(cls.mesh.X[cls.mesh.boundary[i]]))
                    cls.vetor_vy[j] = 1.5*u_0*(1.0 - (2.0*(cls.mesh.X[j]-ref)/h)**2)
                    
                elif BC[i]['vy'] != 'None':
                    row = cls.Matriz.getrow(j + cls.mesh.npoints)
                    for col in row.indices:
                        cls.Matriz[j + cls.mesh.npoints,col] = 0
                    cls.Matriz[j + cls.mesh.npoints,j + cls.mesh.npoints] = 1.0
                    
                    if len(cls.mesh.FSI) > 0 and BC[i] in cls.mesh.FSI:
                        cls.vetor_vy[j] = cls.fluid.vy[j]
                    else:
                        cls.vetor_vy[j] = float(BC[i]['vy']) 
                
                if cls.mesh.mesh_kind == 'quad':
                                                
                    if BC[i]['T'] != 'None':
                        row = cls.Matriz_T.getrow(j)
                        for col in row.indices:
                            cls.Matriz_T[j,col] = 0
                        cls.Matriz_T[j,j] = 1.0
                        
                        cls.vetor_T[j] = float(BC[i]['T'])
                        
                    if j in cls.mesh.IEN[:,:3]:
                        if BC[i]['p'] != 'None':
                            row = cls.Matriz.getrow(cls.mesh.converter[j] + 2*cls.mesh.npoints)
                            for col in row.indices:
                                cls.Matriz[cls.mesh.converter[j] + 2*cls.mesh.npoints,col] = 0
                            cls.Matriz = cls.Matriz.tolil()
                            cls.Matriz[cls.mesh.converter[j] + 2*cls.mesh.npoints,cls.mesh.converter[j] + 2*cls.mesh.npoints] = 1.0
                            cls.Matriz = cls.Matriz.tocsr()
                            
                            cls.vetor_p[cls.mesh.converter[j]] = float(BC[i]['p'])
                        
                elif cls.mesh.mesh_kind == 'mini':
                    if j < cls.mesh.npoints_p: 
                        if BC[i]['T'] != 'None':
                            row = cls.Matriz_T.getrow(j)
                            for col in row.indices:
                                cls.Matriz_T[j,col] = 0
                            cls.Matriz_T[j,j] = 1.0
                            
                            cls.vetor_T[j] = float(BC[i]['T'])
                        
                        if BC[i]['p'] != 'None':
                            row = cls.Matriz.getrow(j + 2*cls.mesh.npoints)
                            for col in row.indices:
                                cls.Matriz[j + 2*cls.mesh.npoints,col] = 0
                            cls.Matriz = cls.Matriz.tolil()
                            cls.Matriz[j + 2*cls.mesh.npoints,j + 2*cls.mesh.npoints] = 1.0
                            cls.Matriz = cls.Matriz.tocsr()
                            
                            cls.vetor_p[j] = float(BC[i]['p'])
        

        cls.vetor_vx = csr_matrix(cls.vetor_vx.reshape(-1,len(cls.vetor_vx)))
        cls.vetor_vy = csr_matrix(cls.vetor_vy.reshape(-1,len(cls.vetor_vy)))
        cls.vetor_p = csr_matrix(cls.vetor_p.reshape(-1,len(cls.vetor_p)))
        cls.vetor_T = csr_matrix(cls.vetor_T.reshape(-1,len(cls.vetor_T)))

        cls.vetor = sp.sparse.bmat([[cls.vetor_vx,cls.vetor_vy,cls.vetor_p]])
        cls.vetor = cls.vetor.tocsr()
        cls.vetor_T = cls.vetor_T.tocsr()
        
        cls.Matriz = cls.Matriz.tocsr()
        cls.Matriz_T = cls.Matriz_T.tocsr()
    
    @classmethod
    def calcFSIForces(cls, norm):
        dudx = sp.sparse.linalg.spsolve(cls.M,sp.sparse.csr_matrix.dot(cls.Gvx,cls.fluid.vx))
        dudy = sp.sparse.linalg.spsolve(cls.M,sp.sparse.csr_matrix.dot(cls.Gvy,cls.fluid.vx))
        
        dvdx = sp.sparse.linalg.spsolve(cls.M,sp.sparse.csr_matrix.dot(cls.Gvx,cls.fluid.vy))
        dvdy = sp.sparse.linalg.spsolve(cls.M,sp.sparse.csr_matrix.dot(cls.Gvy,cls.fluid.vy))
        
        cls.T11 = -cls.fluid.p_quad + 2*(1.0/cls.Re)*dudx
        cls.T22 = -cls.fluid.p_quad + 2*(1.0/cls.Re)*dvdy
        cls.T12 = (1.0/cls.Re)*(dudy + dvdx)
        cls.T21 = (1.0/cls.Re)*(dudy + dvdx)
        
        cls.FSIForces_x = np.multiply(cls.T11, norm[:,0]) + np.multiply(cls.T12, norm[:,1])
        cls.FSIForces_y = np.multiply(cls.T21, norm[:,0]) + np.multiply(cls.T22, norm[:,1])
        
        cls.fluid.FSIForces[:,0] = cls.FSIForces_x 
        cls.fluid.FSIForces[:,1] = cls.FSIForces_y
        
        # Cd_local = 2*np.multiply(-cls.fluid.p_quad,norm[:,0])
        # Cl_local = 2*np.multiply(-cls.fluid.p_quad,norm[:,1])
        
        Cd_local = 2*(np.multiply(cls.T11, norm[:,0]) + np.multiply(cls.T12, norm[:,1]))
        Cl_local = 2*(np.multiply(cls.T21, norm[:,0]) + np.multiply(cls.T22, norm[:,1]))
        
        #calculate the total force------------------------------------------------
        
        cls.Cd = 0
        cls.Cl = 0
        totalforce_x = 0
        totalforce_y = 0
        interp_matrix = np.array([1.0/6.0, 1.0/6.0, 4.0/6.0])
        for elem in cls.mesh.FSI_dict_list:
            nodes = elem['nodes']
            h1 = ((cls.mesh.X[nodes[2]] - cls.mesh.X[nodes[1]])**2 + (cls.mesh.Y[nodes[2]] - cls.mesh.Y[nodes[1]])**2)**0.5
            h2 = ((cls.mesh.X[nodes[2]] - cls.mesh.X[nodes[1]])**2 + (cls.mesh.Y[nodes[2]] - cls.mesh.Y[nodes[1]])**2)**0.5
            h = h1 + h2
            totalforce_x += h*interp_matrix@cls.FSIForces_x[nodes]
            totalforce_y += h*interp_matrix@cls.FSIForces_y[nodes]
            cls.Cd += h*interp_matrix@Cd_local[nodes]
            cls.Cl += h*interp_matrix@Cl_local[nodes]
        cls.totalforce = np.array([totalforce_x,totalforce_y])
            
    
    
    @classmethod
    def solve_fields(cls, i, forces, dt = 0.1, SL_matrix=False,neighborElem=[[]],oface=[]):
        if i > 0 and cls.int_i < 1 and type(cls.IC['mesh_X']) == np.ndarray:
            cls.mesh.X = cls.IC['mesh_X']
            cls.mesh.Y = cls.IC['mesh_Y']
        cls.int_i += 1
        
        cls.dt = dt
        if SL_matrix:
            start = timer()
            if cls.mesh.IEN.shape[1]==4:
                sl = SL.Linear(cls.mesh.IEN,cls.mesh.X,cls.mesh.Y,neighborElem,oface,cls.fluid.vx,cls.fluid.vy)
                sl.compute(cls.dt)
                cls.fluid.vxd = sl.conv*cls.fluid.vx[0:cls.mesh.npoints_p]
                cls.fluid.vyd = sl.conv*cls.fluid.vy[0:cls.mesh.npoints_p]
                cls.fluid.Td = sl.conv*cls.fluid.T
                # interpolating for the centroid
                [cls.fluid.vxd,cls.fluid.vyd] = sl.setCentroid(cls.fluid.vxd,cls.fluid.vyd)
                for i in range(cls.mesh.npoints):
                    cls.mesh.node_list[i].vx = cls.fluid.vx[i]
                    cls.mesh.node_list[i].vy = cls.fluid.vy[i]
                for i in range(cls.mesh.npoints_p):
                    cls.mesh.node_list[i].T = cls.fluid.T[i]
            else:
                #Using parallel ------------------------------------------------------
                element = 'Tri6'
                SLelem = getattr(SL_, element)
                vxALE = cls.mesh.mesh_displacement[:,0]/cls.dt
                vyALE = cls.mesh.mesh_displacement[:,1]/cls.dt
                vxEuler = cls.fluid.vx - vxALE
                vyEuler = cls.fluid.vy - vyALE
                sl1 = SLelem(cls.mesh.IEN,cls.mesh.X,cls.mesh.Y,neighborElem,oface,vxEuler,vyEuler)
                sl1.compute(dt)
                cls.fluid.vxd = sl1.conv*cls.fluid.vx
                cls.fluid.vyd = sl1.conv*cls.fluid.vy
                cls.fluid.Td = sl1.conv*cls.fluid.T
                #End of parallel ------------------------------------------------------
                # sl = SL.Quad(cls.mesh.IEN,cls.mesh.X,cls.mesh.Y,neighborElem,oface,cls.fluid.vx,cls.fluid.vy, cls.mesh.mesh_displacement)
                # sl.compute(cls.dt)
                # cls.fluid.vxd = sl.conv*cls.fluid.vx
                # cls.fluid.vyd = sl.conv*cls.fluid.vy
                # cls.fluid.Td = sl.conv*cls.fluid.T
                for i in range(cls.mesh.npoints):
                    cls.mesh.node_list[i].vx = cls.fluid.vx[i]
                    cls.mesh.node_list[i].vy = cls.fluid.vy[i]
                    cls.mesh.node_list[i].T = cls.fluid.T[i]
                
            end = timer()
            print('time --> SL calculation = ' + str(end-start) + ' [s]')

        else:

            start = timer()
            cls.fluid.vxd, cls.fluid.vyd, cls.fluid.Td = semi_lagrange2(cls.mesh.node_list,cls.mesh.elem_list,cls.fluid.vx,cls.fluid.vy,cls.fluid.T,cls.dt,cls.mesh.IENbound,cls.mesh.boundary_list)
            end = timer()
            print('time --> SL calculation = ' + str(end-start) + ' [s]')

        if cls.mesh.mesh_kind == 'mini':
            cls.fluid.T_mini[0:cls.mesh.npoints_p] = cls.fluid.T
            T_ = cls.fluid.T[cls.mesh.IEN_orig]
            centroids_T = T_.sum(axis=1)/3.0
            cls.fluid.T_mini[cls.mesh.npoints_p:] = centroids_T
        
        
        if cls.porous:
            start = timer()
            cls.set_Darcy_Forchheimer()
            end = timer()
            print('time --> Set Darcy/Forchheimer parcel = ' + str(end-start) + ' [s]')
        
        if cls.turb:
            start = timer()
            cls.calc_turb(neighborElem,oface)
            end = timer()
            print('time --> Set turbulent parcel = ' + str(end-start) + ' [s]')
        
        start = timer()
        
        if len(cls.mesh.FSI) > 0:
            cls.build_quad_GQ()
            cls.set_block_matrices(cls.BC)
            cls.set_block_vectors(forces)
            cls.set_BC_dynamic(cls.BC)
        if cls.porous or cls.turb:
            cls.set_block_vectors(forces)
            cls.set_BC_dynamic(cls.BC)
        elif len(cls.mesh.FSI) == 0:
            cls.set_block_vectors(forces)
            cls.set_BC(cls.BC)
        
        
        end = timer()
        print('time --> Set boundaries = ' + str(end-start) + ' [s]')
        
        start = timer()

        sol = sp.sparse.linalg.spsolve(cls.Matriz,cls.vetor.transpose())

        end = timer()  
        print('time --> Flow solution = ' + str(end-start) + ' [s]')
        
        cls.fluid.vx = sol[0:cls.mesh.npoints]
        cls.fluid.vy = sol[cls.mesh.npoints:2*cls.mesh.npoints]
        cls.fluid.p = sol[2*cls.mesh.npoints:]
     
        if cls.mesh.mesh_kind == 'quad':
            cls.fluid.p_quad[list(cls.mesh.converter.keys())] = cls.fluid.p
            for i in range (3,6):
                j=i-2
                if i == 5:
                    j = 0
                cls.fluid.p_quad[cls.mesh.IEN[:,i]]  = (cls.fluid.p_quad[cls.mesh.IEN[:,j]] + cls.fluid.p_quad[cls.mesh.IEN[:,i-3]])/2.0     
        
        
        start = timer()
        cls.fluid.T = sp.sparse.linalg.spsolve(cls.Matriz_T,cls.vetor_T.transpose())
        # cls.fluid.T = MathWrapper.linSolve(cls.Matriz_T,cls.vetor_T,cls.dll)
        end = timer()
        print('time --> Temperatute solution = ' + str(end-start) + ' [s]')
        
        cls.mesh.calc_normal()
        cls.calcFSIForces(cls.mesh.normal_vect)
        if len(cls.mesh.FSI) > 0:
            cls.solidMesh.update_forces(cls.fluid.FSIForces, cls.BC_solid)
        cls.mesh.mesh_displacement = np.zeros((cls.mesh.npoints,2), dtype = 'float')
            
        return cls.fluid
    
   
        
            

