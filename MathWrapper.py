import ctypes as ct
import numpy as np


def linSolve(Matrix,Vector,dll):
    dll.linSolve.argtypes = ([ct.c_ulong, ct.c_ulong,
                            np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')])
    
    
    
    A = Matrix.transpose().copy()
    b = Vector
    x = np.zeros((A.shape[1]), dtype='double')
    dll.linSolve(A.shape[0], A.shape[1], A, b, x)
    
    return x
