import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, dok_matrix
import scipy.linalg  
import scipy.sparse.linalg 
import meshio
from SolidMesh import SolidMesh
from SolidFEM import FEM
from ExportSolid import export_data
import os

##Mesh reader------------------------------------------------------------------

E = 10**5
nu = 0.3
h = 1.0
rho = 10.0
dt = 0.1
end = 200

case = './Cases/Solid/'
output_dir = case + 'Results'
if not os.path.isdir(output_dir):
   os.mkdir(output_dir)

BC = {'upper_bound': ['None', 'None', 0, -1.0], 'left_bound': [0, 0, 'None', 'None']}

mesh = SolidMesh(case+'malhaTeste.msh', BC)

FEM.set_parameters(mesh, BC, h, E, dt, rho, nu)

umax = [0]
t = [0]
i = 0
while i < end:
    
    u = FEM.solve(i)

    export_data(mesh, output_dir, u, i)
    
    i+=1
    umax.append(np.max(np.abs(FEM.uy)))
    t.append(i*dt)
 
plt.plot(np.array(t),np.array(umax))
