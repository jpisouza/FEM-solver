import xml.etree.ElementTree as ET
import os
import numpy as np
import meshio
import pyvista as pv

class Case:
    
    @classmethod
    def read(cls,case,mesh):
        cls.case = case
        path = os.path.abspath(case + '.xml')
        tree = ET.parse(path)
        cls.root = tree.getroot()
        cls.mesh = mesh
        
    @classmethod
    def set_parameters(cls):
        
        par = cls.root.find('Parameters')
        
        if 'Re' in par.attrib:
            Re = float(par.attrib['Re'])
        else:
            Re = 1.0
        Pr = float(par.attrib['Pr']) 
        Gr = float(par.attrib['Gr'])
        particles = par.attrib['particles']
        if particles == 'True':
            particles = True
        else:
            particles = False
        if par.attrib['Fr_flag'] == 'True':
            Fr = float(par.attrib['Fr'])
            Ga = 1/Fr**2
        else:
            Fr = 0.0
            Ga = float(par.attrib['Ga'])
            
        return Re,Pr,Ga,Gr,Fr, particles
    
    @classmethod
    def set_BC(cls):
        BC_list = []
        for child in cls.root.find('BoundaryCondition'):
            BC_list.append(child.attrib)
        BC_list.reverse() 

        return BC_list
    
    @classmethod       
    def set_IC(cls,i):
        if i > 0:
            file = os.path.dirname(os.path.abspath(cls.case)) + '/Results/sol-' + str(i) + '.vtk'
            data = meshio.read(file)
            IC_ = data.point_data

            if cls.mesh.mesh_kind == 'mini':
                IC_c = data.cell_data['triangle']

                v = np.zeros((cls.mesh.npoints,3), dtype='float')
                v[:cls.mesh.npoints_p,:] = IC_['v']
                v[cls.mesh.npoints_p:,:] = IC_c['v_c']
                
                IC = {}
                IC['vx'] = v[:,0]
                IC['vy'] = v[:,1]
                IC['p'] = IC_['p']
                IC['T'] = IC_['T']

            elif cls.mesh.mesh_kind == 'quad':
                IC = {}
                v = IC_['v']                

                IC['vx'] = v[:,0]
                IC['vy'] = v[:,1]
                IC['p'] = IC_['p'][:cls.mesh.npoints_p]
                IC['T'] = IC_['T'][:cls.mesh.npoints_p]

        else:
            IC = cls.root.find('InitialCondition').attrib
        return IC
    
    @classmethod
    def set_OutFlow(cls):
        OF_list = []
        if cls.root.find('OutFlow') is not None:
            for child in cls.root.find('OutFlow'):
                OF_list.append(child.attrib['name'])
        return OF_list
    
    @classmethod
    def set_particles(cls,i):
        path = os.path.abspath(cls.case + '_particles.xml')
        tree = ET.parse(path)
        root = tree.getroot()
        part = root.find('Particles').attrib
        if i == 0:          
            nparticles = int(part['nparticles'])
            lims = np.zeros((2,2), dtype='float')
            lims[0,0] = float(part['lim_inf_x'])
            lims[0,1] = float(part['lim_sup_x'])
            lims[1,0] = float(part['lim_inf_y'])
            lims[1,1] = float(part['lim_sup_y'])
            
            x_part = np.zeros((nparticles,2), dtype='float')
            x_part[:,0] = lims[0,0] + (lims[0,1] - lims[0,0])*np.random.rand(nparticles)
            x_part[:,1] = lims[1,0] + (lims[1,1] - lims[1,0])*np.random.rand(nparticles)

        else:
            file = os.path.dirname(os.path.abspath(cls.case)) + '/Results/sol_particles' + str(i) + '.vtu'
            x_part = np.array(pv.read(file).points)[:,:2]

        d = float(part['diameter'])
        rho = float(part['rho'])
        nLoop = int(part['nLoop'])
              
        d_part = d*np.ones( (x_part.shape[0]),dtype='float' )
        rho_part = rho*np.ones( (x_part.shape[0]),dtype='float' )
        
        return x_part, d_part, rho_part, nLoop
        
    @classmethod
    def set_kind(cls, root):
       if 'kind' in root.find('Parameters').attrib:
           return root.find('Parameters').attrib['kind']
       else:
           return 'mini'
         
        