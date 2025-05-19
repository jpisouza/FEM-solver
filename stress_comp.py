import SolidSolver
import numpy as np

disp_list = np.linspace(0,1,21)

i = 1
for disp in disp_list:
    E = 1.0e3
    nu = 0.4
    rho = 1000.0
    HE = False
    case = './Cases/Solid_traction/'
    FEM = SolidSolver.Solid_solver(E,nu,rho,disp,HE,case)
    output_dir = case + 'Results'
    
    f = open(output_dir + '/Stresses.txt', 'a')
    f.write(str(np.max(FEM.sigma_x)) + '\t' + str(np.max(FEM.sigma_y)) + '\t' + str(np.max(FEM.PK_stress_x)) + '\t' + str(np.max(FEM.PK_stress_y)) + '\n')
    print('Calculated stress ' + str(i) + '/' + str(len(disp_list)))
    i += 1
    f.close()
