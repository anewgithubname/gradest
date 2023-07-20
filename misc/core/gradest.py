# %%

import time
import ctypes
from numpy import *

libc = ctypes.cdll.LoadLibrary("core/PythonWrapper.so")
# libc = ctypes.CDLL("C:/Users/songa/Dropbox/animated-octo-sniffle/PythonWrapper.dll")
def version():
    libc.info()

def infer(xp, xq, x):    
    
    d = xp.shape[1]
    np = xp.shape[0]
    nq = xq.shape[0]
    n = x.shape[0]

    grad = zeros(n*d, dtype=float32)

    libc.GF.argtypes = [ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypeslib.ndpointer(dtype=float32)]
    libc.GF.restype = None

    start = time.perf_counter()
    libc.GF(xp.ravel('F'), xq.ravel('F'), x.ravel('F'), np, nq, n, d, grad)
    end = time.perf_counter()
    print(f"execution time: {end-start: 0.4f} seconds")

    grad = grad.reshape((n,d), order='F')
    
    return grad

# %%

# test code, main function 
if __name__ == "__main__":
    from util import *
    from readmat import readmat
    import sys

    # fix random seed
    random.seed(0)
    d = 2

    version()
    # generate random data
    xp = readmat("../../data/Xp.matrix")
    xq = readmat("../../data/Xq.matrix")
    
    np = xp.shape[0]
    nq = xq.shape[0]

    # call the function
    grad = infer(xp, xq, vstack((xq, xq)))

# %%
