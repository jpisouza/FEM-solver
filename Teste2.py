import ctypes as ct
import _ctypes
import numpy as np
import os
from timeit import default_timer as timer
os.add_dll_directory(os.getcwd())

dll = ct.cdll.LoadLibrary("MathModule.dll")
dll.linSolve.argtypes = ([ct.c_ulong, ct.c_ulong,
                        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')])



A = np.array([[10.0,2.0],[3.0,1.0]], dtype='double').transpose().copy()
A_ = np.reshape(A,4)
b = np.array([1.0,3.0], dtype='double')
x = np.zeros((A.shape[1]), dtype='double')
start = timer()
dll.linSolve(A.shape[0], A.shape[1], A_, b, x)
end = timer()
print('time solution= ' + str(end-start))

print(x)

_ctypes.FreeLibrary(dll._handle)