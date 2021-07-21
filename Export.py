import numpy as np
import meshio
from pyevtk.hl import pointsToVTK

def export_data(i,output_dir,fluid,MESH,x_part):
    # print(i)
    
    print ('--------Time step = ' + str(i) + ' --> saving solution (VTK)--------\n')
    if MESH.mesh_kind == 'mini':
        point_data = {'p' : fluid.p[0:MESH.npoints_p]}
        data_T  = {'T' : fluid.T}
        data_v = {'v' : np.transpose(np.block([[fluid.vx[0:MESH.npoints_p]],[fluid.vy[0:MESH.npoints_p]],[np.zeros((MESH.npoints_p),dtype='float')]]))}
        point_data.update(data_T)
        point_data.update(data_v)
        cell_data = {'line': {'v_c':np.zeros((len(MESH.msh.cells['line']),3),dtype='float')},'triangle':{'v_c': np.transpose(np.block([[fluid.vx[MESH.npoints_p:]],[fluid.vy[MESH.npoints_p:]],[np.zeros((len(MESH.msh.cells['triangle'])),dtype='float')]]))}}
        meshio.write_points_cells(
        output_dir + '/sol-'+str(i)+'.vtk',
        MESH.msh.points,
        MESH.msh.cells,
        #file_format="vtk-ascii",
        point_data=point_data,
        cell_data = cell_data
        )

    if MESH.mesh_kind == 'quad':
        points = np.block([[MESH.X],[MESH.Y],[np.zeros((MESH.npoints), dtype="float")]]).transpose()
        cells = {}
        cells['line3'] = MESH.IENbound
        cells['triangle6'] = MESH.IEN

        point_data = {'p' : fluid.p_quad}
        data_T  = {'T' : fluid.T_quad}
        data_v = {'v' : np.transpose(np.block([[fluid.vx],[fluid.vy],[np.zeros((MESH.npoints),dtype='float')]]))}
        point_data.update(data_T)
        point_data.update(data_v)
        meshio.write_points_cells(
        output_dir + '/sol-'+str(i)+'.vtk',
        points,
        cells,
        #file_format="vtk-ascii",
        point_data=point_data
        )
    
    if x_part.shape[0] != 0:
        x_p = x_part[:,0].copy()
        y_p = x_part[:,1].copy()
        pointsToVTK(output_dir + "/sol_particles"+str(i),x_p,y_p,0.01*np.ones((x_part.shape[0]), dtype='float'))
