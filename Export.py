import numpy as np
import meshio
from pyevtk.hl import pointsToVTK

def export_data(i,output_dir,fluid,MESH,particleCloud):
    # print(i)
    
    print ('--------Time step = ' + str(i) + ' --> saving solution (VTK)--------\n')
    if MESH.mesh_kind == 'mini':
        point_data = {'p' : fluid.p[0:MESH.npoints_p]}
        data_T  = {'T' : fluid.T}
        data_v = {'v' : np.transpose(np.block([[fluid.vx[0:MESH.npoints_p]],[fluid.vy[0:MESH.npoints_p]],[np.zeros((MESH.npoints_p),dtype='float')]]))}
        if particleCloud != 0:
            data_F = {'forces' : np.block([particleCloud.forces[0:MESH.npoints_p],np.zeros((MESH.npoints_p,1),dtype='float')])}
        else:
            data_F = {'forces' : np.zeros((MESH.npoints_p,3),dtype='float')}
        point_data.update(data_T)
        point_data.update(data_v)
        point_data.update(data_F)
        if particleCloud != 0:
            cell_data = {'line': {'v_c':np.zeros((len(MESH.msh.cells['line']),3),dtype='float'), 'forces_c':np.zeros((len(MESH.msh.cells['line']),3),dtype='float')},'triangle':{'v_c': np.transpose(np.block([[fluid.vx[MESH.npoints_p:]],[fluid.vy[MESH.npoints_p:]],[np.zeros((len(MESH.msh.cells['triangle'])),dtype='float')]])), 'forces_c' : np.block([particleCloud.forces[MESH.npoints_p:],np.zeros((len(MESH.msh.cells['triangle']),1),dtype='float')])}}
        else:
            cell_data = {'line': {'v_c':np.zeros((len(MESH.msh.cells['line']),3),dtype='float'), 'forces_c':np.zeros((len(MESH.msh.cells['line']),3),dtype='float')},'triangle':{'v_c': np.transpose(np.block([[fluid.vx[MESH.npoints_p:]],[fluid.vy[MESH.npoints_p:]],[np.zeros((len(MESH.msh.cells['triangle'])),dtype='float')]])), 'forces_c' : np.zeros((len(MESH.msh.cells['triangle']),3),dtype='float')}}
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
        data_T  = {'T' : fluid.T}
        data_v = {'v' : np.transpose(np.block([[fluid.vx],[fluid.vy],[np.zeros((MESH.npoints),dtype='float')]]))}
        data_normalFSI = {'FSI_normal' : np.transpose(np.block([[MESH.normal_vect[:,0]],[MESH.normal_vect[:,1]],[np.zeros((MESH.npoints),dtype='float')]]))}
        data_FSIForces = {'FSI_forces' : np.transpose(np.block([[fluid.FSIForces[:,0]],[fluid.FSIForces[:,1]],[np.zeros((MESH.npoints),dtype='float')]]))}
        if particleCloud != 0:
            data_F = {'forces' : np.block([particleCloud.forces,np.zeros((MESH.npoints,1),dtype='float')])}
        else:
            data_F = {'forces' : np.zeros((MESH.npoints,3),dtype='float')}
        point_data.update(data_T)
        point_data.update(data_v)
        point_data.update(data_F)
        point_data.update(data_normalFSI)
        point_data.update(data_FSIForces)
        meshio.write_points_cells(
        output_dir + '/sol-'+str(i)+'.vtk',
        points,
        cells,
        #file_format="vtk-ascii",
        point_data=point_data
        )
    
    if particleCloud != 0: 
        if particleCloud.x.shape[0] != 0:
            x_p = particleCloud.x[:,0].copy()
            y_p = particleCloud.x[:,1].copy()
            pointsToVTK(output_dir + "/sol_particles"+str(i),x_p,y_p,0.01*np.ones((particleCloud.x.shape[0]), dtype='float'),  data = {"d" : particleCloud.d, "vx" : particleCloud.v[:,0].copy(), "vy" : particleCloud.v[:,1].copy()})
       
