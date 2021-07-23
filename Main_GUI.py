import numpy as np
import os
import meshio
import pyvista as pv
import Export
from Mesh import mesh
from FEM_sci import FEM
import xml.etree.ElementTree as ET
from Fluid import Fluid
from particleCloud import ParticleCloud


class Main:
        
    @classmethod
    def def_MESH(cls,case):
        path = os.path.abspath(case)
        cls.mesh = meshio.read(path)
        
        cls.boundNames = list(cls.mesh.field_data.keys())
        
    @classmethod
    def set_BC(cls,table):
        BC_list = [None]*table.rowCount()
        for i in range (table.rowCount()):
            j = table.rowCount() - int(table.item(i,4).text())
            BC_list[j] = {}
            BC_list[j]['name'] = str(table.verticalHeaderItem(i).text())
            BC_list[j]['vx'] = str(table.item(i,0).text())
            BC_list[j]['vy'] = str(table.item(i,1).text())
            BC_list[j]['p'] = str(table.item(i,2).text())
            BC_list[j]['T'] = str(table.item(i,3).text())
        return BC_list
    
    @classmethod
    def set_IC(cls,case,textEdit_dt_2):
        if not textEdit_dt_2.toPlainText() == '':
            file = os.path.dirname(os.path.abspath(case)) + '/Results/sol-' + str(textEdit_dt_2.toPlainText()) + '.vtk'
            data = meshio.read(file)
            IC_ = data.point_data

            if cls.MESH.mesh_kind == 'mini':
                IC_c = data.cell_data['triangle']
                
                v = np.zeros((cls.MESH.npoints,3), dtype='float')
                # T_mini = np.zeros((cls.MESH.npoints), dtype='float')
                v[:cls.MESH.npoints_p,:] = IC_['v']
                v[cls.MESH.npoints_p:,:] = IC_c['v_c']
                # T_mini[:cls.MESH.npoints_p] = IC_['T']
                
                forces = np.zeros((cls.MESH.npoints,2), dtype='float')
                forces[:cls.MESH.npoints_p,:] = IC_['forces'][:,:2]
                forces[cls.MESH.npoints_p:,:] = IC_c['forces_c'][:,:2]
                
                # for e in cls.MESH.IEN:
                #     v1,v2,v3,v4 = e
                #     T_mini[v4] = (T_mini[v1] + T_mini[v2] + T_mini[v3])/3.0
                
                IC = {}
                IC['vx'] = v[:,0]
                IC['vy'] = v[:,1]
                IC['p'] = IC_['p']
                IC['T'] = IC_['T']
                # IC['T_mini'] = T_mini
            
            if cls.MESH.mesh_kind == 'quad':
                IC = {}
                v = IC_['v']
                forces = IC_['forces'][:,:2]
                IC['vx'] = v[:,0]
                IC['vy'] = v[:,1]
                IC['p'] = IC_['p'][:cls.MESH.npoints_p]
                IC['T'] = IC_['T'][:cls.MESH.npoints_p]

        else:
            xml = os.path.abspath(os.path.splitext(case)[0] + '.xml')
            if os.path.exists(xml):                      
                tree = ET.parse(xml)
                root = tree.getroot()
                IC = root.find('InitialCondition').attrib
            else:
                IC = {}
                IC['vx'] = 0
                IC['vy'] = 0
                IC['p'] = 0
                IC['T'] = 0
            forces = np.zeros((cls.MESH.npoints,2), dtype='float')

        return IC, forces
    
    @classmethod
    def set_OutFlow(cls,table):
        OF_list = []
        for i in range (table.rowCount()):
            if table.cellWidget(i, 5).isChecked():
                OF_list.append(table.verticalHeaderItem(i).text())
        return OF_list
                
    @classmethod
    def set_parameters(cls,textEdit_Re,textEdit_Pr,textEdit_Ga,textEdit_Gr,textEdit_dt,convection):
        if convection:
            Ga = float(textEdit_Ga.toPlainText())
        else:
            Ga = 1.0/float(textEdit_Ga.toPlainText())**2
        return  float(textEdit_Re.toPlainText()), float(textEdit_Pr.toPlainText()), Ga, float(textEdit_Gr.toPlainText()), float(textEdit_dt.toPlainText())
        
    @classmethod
    def set_simulation(cls,case,table,textEdit_Re,textEdit_Pr,textEdit_Ga,textEdit_Gr,textEdit_dt,textEdit_dt_2,convection):
        cls.BC = cls.set_BC(table)
        cls.Re,cls.Pr,cls.Ga,cls.Gr, cls.dt = cls.set_parameters(textEdit_Re,textEdit_Pr,textEdit_Ga,textEdit_Gr,textEdit_dt,convection)
        cls.outflow = cls.set_OutFlow(table)

        cls.MESH = mesh(os.path.splitext(case)[0])
        cls.MESH.set_boundary_prior(cls.BC,cls.outflow)      

        cls.IC, cls.forces = cls.set_IC(case,textEdit_dt_2)
        
        cls.fluid = Fluid(cls.MESH,cls.Re,cls.Pr,cls.Ga,cls.Gr,cls.IC)
        FEM.set_matrices(cls.MESH,cls.fluid,cls.dt,cls.BC)
        
        #----------------------------------------SL Gustavo ----------------------------------------------------------------
        # cls.neighborElem = [[]]
        # cls.neighborElem = [[] for i in range(cls.MESH.npoints_p)]
        # for i in range(0,cls.MESH.IEN_orig.shape[0]):
        #     for j in range(0,cls.MESH.IEN_orig.shape[1]):
        #         v = cls.MESH.IEN_orig[i][j]
        #         cls.neighborElem[v].append(i)
        
        # cls.oface = -1*np.ones( (cls.MESH.ne,3),dtype='int' )

        # for e in range(len(cls.MESH.IEN_orig)):
        #     for i in range(3):
        #         ID = cls.MESH.IEN_orig[e,i]
        #         point = cls.MESH.node_list[ID]
        #         for op in range (len(point.opostos)):
        #             if point.aresta_correspondente[op][0] in cls.MESH.IEN_orig[e] and  point.aresta_correspondente[op][1] in cls.MESH.IEN_orig[e]:
        #                 cls.oface[e,i] = point.opostos[op]
        #----------------------------------------------------------------------------------------------------------------------------------
                
        cls.output_dir = os.path.dirname(os.path.abspath(case)) + '/Results'
        if not os.path.isdir(cls.output_dir):
            os.mkdir(cls.output_dir)
            
    @classmethod
    def set_simulation_particles(cls,case,particles_D,particles_rho,particles_nLoop,particles_nparticles,particles_lim_inf_x,particles_lim_sup_x,particles_lim_inf_y,particles_lim_sup_y,textEdit_dt_2):
        
        cls.x_part = np.zeros((particles_nparticles,2), dtype='float')
        if textEdit_dt_2.toPlainText() == '':            
            cls.x_part[:,0] = particles_lim_inf_x + (particles_lim_sup_x - particles_lim_inf_x)*np.random.rand(particles_nparticles)
            cls.x_part[:,1] = particles_lim_inf_y + (particles_lim_sup_y - particles_lim_inf_y)*np.random.rand(particles_nparticles)
            cls.v_part = np.zeros((particles_nparticles,2), dtype='float')
            
        else:
            file = os.path.dirname(os.path.abspath(case)) + '/Results/sol_particles' + str(textEdit_dt_2.toPlainText()) + '.vtu'
            cls.x_part = np.array(pv.read(file).points)[:,:2]
            vx = np.array(pv.read(file)['vx'])
            vy = np.array(pv.read(file)['vy'])
            cls.v_part = np.block([[vx],[vy]]).transpose()
        
            
        cls.d_part = particles_D*np.ones( (cls.x_part.shape[0]),dtype='float' )
        cls.rho_part = particles_rho*np.ones( (cls.x_part.shape[0]),dtype='float' )
        cls.nLoop = particles_nLoop
        
        cls.particleCloud = ParticleCloud(cls.MESH.elem_list,cls.MESH.node_list,cls.x_part,cls.v_part,cls.d_part,cls.rho_part,cls.forces)
        
    @classmethod
    def play(cls,i):
        
        # cls.fluid = FEM.solve_fields(True,cls.neighborElem,cls.oface)
        
        # index = np.where(cls.forces != 0)
        # if len(index[0]) != 0:
        #     print(cls.forces[index[0]])
        
        if cls.particles:
            cls.fluid = FEM.solve_fields(cls.particleCloud.forces)
            cls.particleCloud.solve(cls.dt,cls.nLoop,cls.fluid.Re,1.0/np.sqrt(cls.fluid.Ga))
        else:
            cls.fluid = FEM.solve_fields(np.zeros((cls.MESH.npoints,2), dtype='float'))
            cls.particleCloud = 0
        
        Export.export_data(i,cls.output_dir,cls.fluid,cls.MESH,cls.particleCloud)
        
        
