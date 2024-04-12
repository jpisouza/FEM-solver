import numpy as np
import meshio

def export_data(mesh, output_dir, u, u_w, sigma_x, sigma_y, tau_xy, sigma_VM, i):
    
    print ('--------Time step = ' + str(i) + ' --> saving solid solution (VTK)--------\n')
    if mesh.mesh_kind == 'quad':
        
        #deformed solid
        cells = {}
        cells['line3'] = mesh.IENbound
        cells['triangle6'] = mesh.IEN
        ux = np.array([u[2*n] for n in range(mesh.npoints)])
        uy = np.array([u[2*n+1] for n in range(mesh.npoints)])
        points = np.block([[mesh.X],[mesh.Y],[np.zeros((mesh.npoints), dtype="float")]]).transpose()
        point_data = {'u' : np.transpose(np.block([[ux],[uy],[np.zeros((mesh.npoints),dtype='float')]]))}
        data_sigmax = {'sigma_x' : sigma_x}
        data_sigmay = {'sigma_y' : sigma_y}
        data_tauxy = {'tau_xy' : tau_xy}
        data_sigmaVM = {'sigma_VM' : sigma_VM}
        point_data.update(data_sigmax)
        point_data.update(data_sigmay)
        point_data.update(data_tauxy)
        point_data.update(data_sigmaVM)       
        meshio.write_points_cells(
        output_dir + '/solid_sol-'+str(i)+'.vtk',
        points,
        cells,
        point_data=point_data
        )

def export_static(mesh, output_dir, u, u_w, sigma_x, sigma_y, tau_xy, sigma_VM):
    
    print ('-------- saving solid solution (VTK)--------\n')
    if mesh.mesh_kind == 'quad':
        
        #deformed solid
        cells = {}
        cells['line3'] = mesh.IENbound
        cells['triangle6'] = mesh.IEN
        ux = np.array([u[2*n] for n in range(mesh.npoints)])
        uy = np.array([u[2*n+1] for n in range(mesh.npoints)])
        points = np.block([[mesh.X],[mesh.Y],[np.zeros((mesh.npoints), dtype="float")]]).transpose()
        point_data = {'u' : np.transpose(np.block([[ux],[uy],[np.zeros((mesh.npoints),dtype='float')]]))}
        data_sigmax = {'sigma_x' : sigma_x}
        data_sigmay = {'sigma_y' : sigma_y}
        data_tauxy = {'tau_xy' : tau_xy}
        data_sigmaVM = {'sigma_VM' : sigma_VM}
        point_data.update(data_sigmax)
        point_data.update(data_sigmay)
        point_data.update(data_tauxy)
        point_data.update(data_sigmaVM)       
        meshio.write_points_cells(
        output_dir + '/static_sol.vtk',
        points,
        cells,
        point_data=point_data
        )

def export_nat(mesh, output_dir, u_w,i):

        #natural modes
        cells = {}
        cells['line3'] = mesh.IENbound
        cells['triangle6'] = mesh.IEN
        u_wx = np.array([u_w[2*n,i] for n in range(mesh.npoints)])
        u_wy = np.array([u_w[2*n+1,i] for n in range(mesh.npoints)])
        points = np.block([[mesh.X_orig + u_wx],[mesh.Y_orig + u_wy],[np.zeros((mesh.npoints), dtype="float")]]).transpose()
        point_data = {'u' : np.transpose(np.block([[u_wx],[u_wy],[np.zeros((mesh.npoints),dtype='float')]]))}    
        meshio.write_points_cells(
        output_dir + '/naturalMode_sol-'+str(i)+'.vtk',
        points,
        cells,
        point_data=point_data
        )

