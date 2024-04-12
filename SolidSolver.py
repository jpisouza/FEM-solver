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
from ExportSolid import export_nat
from ExportSolid import export_static
import os

##Mesh reader------------------------------------------------------------------

E = 10**6
nu = 0.3
h = 1.0
rho = 100.0
dt = 0.1
end = 5
dynamic = False

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

n_freq = 10

if dynamic:
    while i < end:
        
        u, u_w = FEM.solve(i,0,True)
    
        export_data(mesh, output_dir, u, u_w, FEM.sigma_x, FEM.sigma_y, FEM.tau_xy, FEM.sigma_VM, i)
        
        i+=1
        umax.append(np.max(np.abs(FEM.uy)))
        t.append(i*dt)
     
    plt.plot(np.array(t),np.array(umax))

else:
    u, u_w = FEM.solve_static()
    export_static(mesh, output_dir, u, u_w, FEM.sigma_x, FEM.sigma_y, FEM.tau_xy, FEM.sigma_VM)

print('--------------------'+str(n_freq)+' First Natural Frequencies [Hz]--------------------------------')
for i in range(n_freq):
    export_nat(mesh, output_dir, u_w, i)
    print(np.sqrt(FEM.omega_sort[i])/(2*np.pi))
print('--------------------------------------------------------------------------------------------------')
    
    
    
    