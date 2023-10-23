import numpy as np
from timeit import default_timer as timer
import scipy as sp
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, dok_matrix
import scipy.linalg  
import scipy.sparse.linalg 
import Elements

class Turbulence:
    
    @classmethod
    def set_model(cls,solver):
        cls.solver = solver
        
    @classmethod
    def calc_turb(cls):
         cls.solver.fluid.nu_t = 0.1*np.ones((cls.solver.mesh.npoints), dtype='float')       
        

