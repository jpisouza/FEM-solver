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

def Solid_solver(E,nu,rho,disp,HE,case,msh):

    g = [0,-2.0]
    h = 1.0
    dt = 0.02
    end = 1000
    
    gamma = 0.5
    beta = 0.5
    dynamic = True
    
    output_dir = case + 'Results'
    if not os.path.isdir(output_dir):
       os.mkdir(output_dir)
    
    
    # BC = {'fixed': [0,0, 'None', 'None']}
    BC = {'right_bound': ['None', 'None', 'None', 'None'], 'left_bound': [0, 0, 'None', 'None'], 'upper_bound': ['None', 'None', 'None', 'None']}
    # BC = {'right_bound': [disp, 'None', 'None', 'None'], 'left_bound': [0, 'None', 'None', 'None'], 'lower_bound': ['None', 0, 'None', 'None']}
    IC = None
    mesh = SolidMesh(case+msh, BC)
    
    FEM.set_parameters(mesh, BC, IC, h, E, dt, rho, nu, gamma, beta, g)
    
    umax = [0]
    t = [0]
    i = 0
    
    nat_freq = False
    n_freq = 10

    refconfig_force = True #Force calculated in the reference configuration (undeformed)
    
    if dynamic:
        while i < end:
            
            if HE:
                u = FEM.solve_HE(i,0,refconfig_force)
            else:
                u, u_w = FEM.solve(i,0,refconfig_force)    
        
            export_data(mesh, output_dir, u, FEM.u_prime, FEM.u_doubleprime, FEM.sigma_x, FEM.sigma_y, FEM.tau_xy, FEM.PK_stress_x, FEM.PK_stress_y, FEM.PK_stress_xy, FEM.sigma_VM, i)
            print ('--------Time step = ' + str(i) + ' --> saving solid solution (VTK)--------\n')
            
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
        export_static(mesh, output_dir, u, FEM.sigma_x, FEM.sigma_y, FEM.tau_xy, FEM.PK_stress_x, FEM.PK_stress_y, FEM.PK_stress_xy, FEM.sigma_VM)
        
    if nat_freq and not HE:
        print('--------------------'+str(n_freq)+' First Natural Frequencies [Hz]--------------------------------')
        for i in range(n_freq):
            export_nat(mesh, output_dir, u_w, i)
            print(np.sqrt(FEM.omega_sort[i])/(2*np.pi))
        print('--------------------------------------------------------------------------------------------------')

    return FEM

if __name__ == "__main__":
    E = 1.4e6
    nu = 0.4
    rho = 1000.0
    disp = 1.0
    HE = True
    case = './Cases/Solid_Turek/'
    # case = 'C:\\Users\\jpinn\\Downloads\\'
    msh = 'malhaTeste.msh'
    # msh = 'tri_quad.msh'
    FEM = Solid_solver(E, nu, rho, disp, HE, case, msh)

