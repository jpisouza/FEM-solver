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

E = 10**5
nu = 0.3
h = 1.0
rho = 300.0
dt = 0.1
end = 1000
dynamic = True

case = './Cases/Solid_NewmarkHE/'
output_dir = case + 'Results'
if not os.path.isdir(output_dir):
   os.mkdir(output_dir)


BC = {'upper_bound': ['None', 'None', 0, -20], 'left_bound': [0, 0, 'None', 'None']}

mesh = SolidMesh(case+'malhaTeste.msh', BC)

FEM.set_parameters(mesh, BC, h, E, dt, rho, nu)

umax = [0]
t = [0]
i = 0

nat_freq = False
n_freq = 10
HE = True

if dynamic:
    while i < end:
        
        if HE:
            u = FEM.solve_HE(i,0,nat_freq)
        else:
            u, u_w = FEM.solve(i,0,nat_freq)    
    
        export_data(mesh, output_dir, u, FEM.sigma_x, FEM.sigma_y, FEM.tau_xy, FEM.sigma_VM, i)
        
        i+=1
        umax.append(np.max(np.abs(FEM.uy)))
        t.append(i*dt)
    
    plt.rc('font', family='serif', size=14)
    plt.plot(np.array(t),np.array(umax))
    plt.xlabel(r'Time [s]')
    plt.ylabel(r'Maximum displacement [m]')
    plt.show()

else:
    
    if HE:
        u = FEM.solve_staticHE() 
    else:
        u, u_w = FEM.solve_static(nat_freq)
    export_static(mesh, output_dir, u, FEM.sigma_x, FEM.sigma_y, FEM.tau_xy, FEM.sigma_VM)
    
if nat_freq and not HE:
    print('--------------------'+str(n_freq)+' First Natural Frequencies [Hz]--------------------------------')
    for i in range(n_freq):
        export_nat(mesh, output_dir, u_w, i)
        print(np.sqrt(FEM.omega_sort[i])/(2*np.pi))
    print('--------------------------------------------------------------------------------------------------')

 

