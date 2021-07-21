import ctypes as ct
import _ctypes
import numpy as np
import os
os.add_dll_directory(os.getcwd())

dll = ct.cdll.LoadLibrary("MathModule.dll")
dll.matMul.argtypes = ([ct.c_ulong, ct.c_ulong, ct.c_ulong, ct.c_ulong,
                        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')])


A = np.array([[10,2],[3,1]], dtype='float')
B = np.array([[10,2],[3,1]], dtype='float')
M = np.zeros((A.shape[0],B.shape[1]), dtype='float')
dll.matMul(A.shape[0], A.shape[1], B.shape[0], B.shape[1], A, B, M)

print(M)

_ctypes.FreeLibrary(dll._handle)