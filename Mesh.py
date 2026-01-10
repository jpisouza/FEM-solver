import numpy as np
import meshio
from node import Node
from element import Element
import os

class mesh:

    def __init__(self,case,kind='mini', porous_list = [], solid_list = [], limit_name = '', smooth_value = 1.0, porosity = 1.0):

        self.mesh_kind = kind
        path = os.path.abspath( case + '.msh')
        
        self.porous_list = porous_list
        self.solid_list = solid_list
        self.limit_name = limit_name
        self.smooth_value = smooth_value
        self.porosity = porosity
        self.msh = meshio.read(path)
        
        self.dict_boundary = {}
        self.dict_element = {}
        i = 0
        for key in self.msh.field_data:
            if self.msh.field_data[key][1] < 2:
                self.dict_boundary[self.msh.field_data[key][0]] = i
            else:
                self.dict_element[self.msh.field_data[key][0]] = i
            i+=1
            
        
        if 'triangle6' in self.msh.cells:
            self.mesh_kind = 'quad'
            
            self.X = self.msh.points[:,0]
            self.Y = self.msh.points[:,1]
            
            self.IEN = self.msh.cells['triangle6']
            self.IEN_orig = self.IEN[:,:3].copy()
            self.ne=len(self.IEN)
            self.IENbound = self.msh.cells['line3']
            
            self.npoints = np.max(self.IEN) + 1
            self.npoints_p = len(np.unique(self.IEN[:,:3]))
            
            points_p = np.arange(0,self.npoints_p,1)
            points_p_real = np.unique(self.IEN[:,:3])
            
            self.mesh_velocity = np.zeros((self.npoints,2), dtype = 'float')
            self.mesh_displacement = np.zeros((self.npoints,2), dtype = 'float')
            
            self.converter = dict(zip(points_p_real,points_p))
            
            self.IENboundTypeElem = list(self.msh.cell_data['line3']['gmsh:physical'])
            self.IENTypeElem = list(self.msh.cell_data['triangle6']['gmsh:physical'])
            self.boundNames = list(self.msh.field_data.keys())
            self.IENboundElem = [self.boundNames[self.dict_boundary[elem]] for elem in self.IENboundTypeElem]
            self.IENElem = []
            self.porous_elem = []
            self.porosity_array = []
            if len(self.porous_list) > 0:
                for elem in self.IENTypeElem:
                    self.IENElem.append(self.boundNames[self.dict_element[elem]])
                    if self.boundNames[self.dict_element[elem]] in porous_list:
                        self.porous_elem.append(1)
                        self.porosity_array.append(self.porosity)
                    else:
                        self.porous_elem.append(0)
                        self.porosity_array.append(1.0)
            else:
                self.porosity_array = self.ne*[1.0]
                
            #Detecting solid elements
            self.solid_elem = []
            self.solid_nodes = []
            if len(self.solid_list) > 0:
                for elem in range (len(self.IEN)):
                    if self.boundNames[self.dict_element[self.IENTypeElem[elem]]] in self.solid_list:
                        self.solid_elem.append(elem)
                        for i in self.IEN[elem]:
                            if i not in self.solid_nodes:
                               self.solid_nodes.append(i) 
                        
            #Removing non-boundary interfaces
            self.X_interf = []
            self.Y_interf = []
            if len(limit_name) > 0:
                self.boundNames.remove(limit_name)
                i = 0
                while i < len(self.IENboundElem):
                    if self.IENboundElem[i] == limit_name:
                        self.IENboundElem.pop(i)
                        self.IENboundTypeElem.pop(i)
                        self.IENbound = np.delete(self.IENbound, i, 0)
                        self.X_interf+=list(self.X[self.IENbound[i]])
                        self.Y_interf+=list(self.Y[self.IENbound[i]])
                    i+=1
                  
            self.boundary = []
            

        elif self.mesh_kind == 'mini':

            self.X = self.msh.points[:,0]
            self.Y = self.msh.points[:,1]

            self.IEN = self.msh.cells['triangle']
            self.IEN_orig = self.IEN.copy()
            self.ne=len(self.IEN)

            self.npoints_p = len(self.X)
            
            self.IENbound = self.msh.cells['line']
            self.IENboundTypeElem = list(self.msh.cell_data['line']['gmsh:physical'])
            self.IENTypeElem = list(self.msh.cell_data['triangle']['gmsh:physical'])
            self.boundNames = list(self.msh.field_data.keys())
            self.IENboundElem = [self.boundNames[self.dict_boundary[elem]] for elem in self.IENboundTypeElem]
            self.IENElem = []
            self.porous_elem = []
            self.porosity_array = []
            if len(self.porous_list) > 0:
                for elem in self.IENTypeElem:
                    self.IENElem.append(self.boundNames[self.dict_element[elem]])
                    if self.boundNames[self.dict_element[elem]] in porous_list:
                        self.porous_elem.append(1)
                        self.porosity_array.append(self.porosity)
                    else:
                        self.porous_elem.append(0)
                        self.porosity_array.append(1.0)
            else:
                self.porosity_array = self.ne*[1.0]
            
            #Removing non-boundary interfaces
            self.X_interf = []
            self.Y_interf = []
            if len(limit_name) > 0:
                self.boundNames.remove(limit_name)
                i = 0
                while i < len(self.IENboundElem):
                    if self.IENboundElem[i] == limit_name:
                        self.IENboundElem.pop(i)
                        self.IENboundTypeElem.pop(i)
                        self.IENbound = np.delete(self.IENbound, i, 0)
                        self.X_interf+=list(self.X[self.IENbound[i]])
                        self.Y_interf+=list(self.Y[self.IENbound[i]])
                    i+=1
            
            self.boundary = []
            
            #Acrescenta ponto central do elemento
            new_elements = np.arange(self.npoints_p,self.npoints_p+self.ne,1).reshape(self.ne,1)
            self.IEN = np.block([[self.IEN,new_elements]])
            
            X_ = self.X[self.IEN_orig]
            Y_ = self.Y[self.IEN_orig]

            centroids_x = (1.0/3.0)*X_.sum(axis=1)
            centroids_y = (1.0/3.0)*Y_.sum(axis=1)
            self.X = np.append(self.X,centroids_x)
            self.Y = np.append(self.Y,centroids_y)

            self.npoints = len(self.X)

        
        Node.node_list = []
        Element.elem_list = []
        
        self.porous_nodes = np.zeros((self.npoints), dtype='float')
        for ID in range (len(self.X)):
            self.node = Node(ID,self.IEN,self.IEN_orig, self.X[ID], self.Y[ID])
            print('Node ' + str(ID + 1) + ' --------> generated')
        for ID in range (len(self.IEN)):
            self.element = Element(ID,self.IEN,self.IEN_orig,Node.node_list,mesh = self)
            print('Element ' + str(ID + 1) + ' --------> generated')       
        print('\n')


        self.node_list = Node.node_list
        self.elem_list = Element.elem_list

    def set_boundary_prior(self,BC,outflow,FSI):
        
        self.BC = BC
        self.FSI = FSI
        k = 0
        bound_list = []
        self.FSI_list = []
        self.FSI_dict_list = []
        outflow_bound_list = []
        for bound in BC:
            bound_list.append([])
            for i in range (len(self.IENboundElem)):
                if self.IENboundElem[i] == bound['name']:
                    if bound in self.FSI or ('calc_force' in bound and bound['calc_force'] == 'True'):
                        self.FSI_dict_list.append({'bound_elem':i})
                        self.FSI_dict_list[-1]['nodes'] = self.IENbound[i]
                        self.FSI_dict_list[-1]['norm'] = np.array(([[0,0],[0,0],[0,0]]), dtype='float')
                        if 'calc_force' in bound and bound['calc_force'] == 'True':
                            bound['fluid_interface'] = "True"
                        self.FSI_dict_list[-1]['fluid_interface'] = bound['fluid_interface']
                    for l in range (len(self.IENbound[i])):
                        if bound in self.FSI or ('calc_force' in bound and bound['calc_force'] == 'True'):
                            if self.IENbound[i,l] not in self.FSI_list:
                                self.FSI_list.append(self.IENbound[i,l])
                        if self.IENbound[i,l] not in bound_list[k]:
                            bound_list[k].append(self.IENbound[i,l])
                            if bound['name'] in outflow:
                                outflow_bound_list.append(self.IENbound[i,l])
                                self.node_list[self.IENbound[i,l]].out_flow = True
            k+=1

        self.boundary = bound_list
        self.boundary_list = []
        for r in range(len(self.boundary)):
            self.boundary_list = self.boundary_list + self.boundary[r]
        self.outflow_boundary = outflow_bound_list
        
        #calculate distances to moving boundaries-----------------------------------------
        if len(self.FSI_list) > 0:
            for node in self.node_list:
                dist_min = 1e5
                for i in self.FSI_list:
                    dist = ((node.x - self.X[i])**2 + (node.y - self.Y[i])**2)**0.5
                    if dist < dist_min:
                        dist_min = dist
                        ID = i
                node.FSI_dist = [ID,dist_min]
        

    def calc_normal(self):
        
        self.normal_vect = np.zeros((len(self.X),2), dtype='float')
        for edge in self.FSI_dict_list:
            if edge['fluid_interface'] == 'True':
                if (self.X[edge['nodes'][1]] - self.X[edge['nodes'][0]]) != 0.0:
                    if (self.X[edge['nodes'][1]] - self.X[edge['nodes'][0]]) > 0.0:
                        edge['norm'][2][1] = -1.0        
                    else:
                        edge['norm'][2][1] = 1.0
                    edge['norm'][2][0] = -edge['norm'][2][1]*(self.Y[edge['nodes'][1]] - self.Y[edge['nodes'][0]])/(self.X[edge['nodes'][1]] - self.X[edge['nodes'][0]])

                else:
                    if (self.Y[edge['nodes'][1]] - self.Y[edge['nodes'][0]]) < 0.0:
                        edge['norm'][2][0] = -1.0
                        edge['norm'][2][1] = 0.0
                    else:
                        edge['norm'][2][0] = 1.0
                        edge['norm'][2][1] = 0.0
            
                mod = np.sqrt(edge['norm'][2][1]**2 + edge['norm'][2][0]**2)
                
                edge['norm'][2][1] = edge['norm'][2][1]/mod
                edge['norm'][2][0] = edge['norm'][2][0]/mod
               
                self.normal_vect[edge['nodes'][0],:] += edge['norm'][2]
                self.normal_vect[edge['nodes'][1],:] += edge['norm'][2]
                self.normal_vect[edge['nodes'][2],:] += edge['norm'][2]
        
        for i in self.FSI_list:
            mod = np.sqrt(self.normal_vect[i,0]**2 + self.normal_vect[i,1]**2)
            if mod != 0:
                self.normal_vect[i,:] = self.normal_vect[i,:]/mod
            


