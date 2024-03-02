import numpy as np
import meshio

def export_data(mesh, output_dir, u, i):
    
    print ('--------Time step = ' + str(i) + ' --> saving solution (VTK)--------\n')
    if mesh.mesh_kind == 'quad':
        
        #deformed solid
        cells = {}
        cells['line3'] = mesh.IENbound
        cells['triangle6'] = mesh.IEN
        ux = np.array([u[2*n] for n in range(mesh.npoints)])
        uy = np.array([u[2*n+1] for n in range(mesh.npoints)])
        points = np.block([[mesh.X],[mesh.Y],[np.zeros((mesh.npoints), dtype="float")]]).transpose()
        point_data = {'ux' : ux}
        data_uy = {'uy' : uy}
        point_data.update(data_uy)
        meshio.write_points_cells(
        output_dir + '/sol-'+str(i)+'.vtk',
        points,
        cells,
        point_data=point_data
        )
