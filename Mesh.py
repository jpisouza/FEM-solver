import numpy as np
import meshio
from node import Node
from element import Element
import os

class mesh:

    def __init__(self,case,kind='mini',porous_list = []):

        self.mesh_kind = kind
        path = os.path.abspath( case + '.msh')
        
        self.porous_list = porous_list
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
            
            self.converter = dict(zip(points_p_real,points_p))
            
            self.IENboundTypeElem = list(self.msh.cell_data['line3']['gmsh:physical'])
            self.IENTypeElem = list(self.msh.cell_data['triangle6']['gmsh:physical'])
            self.boundNames = list(self.msh.field_data.keys())
            self.IENboundElem = [self.boundNames[self.dict_boundary[elem]] for elem in self.IENboundTypeElem]
            self.IENElem = []
            self.porous_elem = []
            if len(self.porous_list) > 0:
                for elem in self.IENTypeElem:
                    self.IENElem.append(self.boundNames[self.dict_element[elem]])
                    if self.boundNames[self.dict_element[elem]] in porous_list:
                        self.porous_elem.append(1)
                    else:
                        self.porous_elem.append(0)
            
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
            if len(self.porous_list) > 0:
                for elem in self.IENTypeElem:
                    self.IENElem.append(self.boundNames[self.dict_element[elem]])
                    if self.boundNames[self.dict_element[elem]] in porous_list:
                        self.porous_elem.append(1)
                    else:
                        self.porous_elem.append(0)
            
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

        
        # elif self.mesh_kind == 'quad':

        #     self.X = self.msh.points[:,0]
        #     self.Y = self.msh.points[:,1]

        #     self.IEN = self.msh.cells['triangle']
        #     self.IEN_orig = self.IEN.copy()
        #     self.ne=len(self.IEN)

        #     self.npoints_p = len(self.X)

        #     self.IENbound = self.msh.cells['line']
        #     self.IENboundTypeElem = list(self.msh.cell_data['line']['gmsh:physical'])
        #     self.IENTypeElem = list(self.msh.cell_data['triangle']['gmsh:physical'])
        #     self.boundNames = list(self.msh.field_data.keys())
        #     self.IENboundElem = [self.boundNames[self.dict_boundary[elem]] for elem in self.IENboundTypeElem]
        #     self.IENElem = []
        #     self.porous_elem = []
        #     if len(self.porous_list) > 0:
        #         for elem in self.IENTypeElem:
        #             self.IENElem.append(self.boundNames[self.dict_element[elem]])
        #             if self.boundNames[self.dict_element[elem]] in porous_list:
        #                 self.porous_elem.append(1)
        #             else:
        #                 self.porous_elem.append(0)
            
        #     self.boundary = []

        #     #Modificar a IEN

        #     self.IEN = np.zeros((self.ne,6), dtype='int')
        #     self.IEN[:,:3] = self.IEN_orig
            
        #     num = self.npoints_p
        #     for e1 in range(len(self.IEN_orig)):
        #         for e2 in range(len(self.IEN_orig)):
        #             p = np.where(np.isin(self.IEN_orig[e1],self.IEN_orig[e2]))[0]
        #             p2 = np.where(np.isin(self.IEN_orig[e2],self.IEN_orig[e1]))[0]
        #             flag_plus1 = False
        #             if len(p) == 2:
        #                 if (np.abs(p[0] - p[1])) > 1:
        #                     if self.IEN[e1,5] == 0:
        #                         self.IEN[e1,5] = num
        #                         flag_plus1 = True
        #                 else:
        #                     if self.IEN[e1,np.min(p)+3] == 0:
        #                         self.IEN[e1,np.min(p)+3] = num
        #                         flag_plus1 = True
        #                 if (np.abs(p2[0] - p2[1])) > 1:
        #                     if self.IEN[e2,5] == 0:
        #                         self.IEN[e2,5] = num
        #                         flag_plus1 = True   
        #                 else:
        #                     if self.IEN[e2,np.min(p2)+3] == 0:
        #                         self.IEN[e2,np.min(p2)+3] = num
        #                         flag_plus1 = True
        #                 if flag_plus1:
        #                     num+=1
                        


        #     L = np.where(self.IEN[:,3:]==0)
            
        #     self.IEN[:,3:][L[0],L[1]] = np.arange(num,num+len(L[0]),1)
            
        #     X_ = np.zeros((np.max(self.IEN)+1), dtype='float')
        #     Y_ = np.zeros((np.max(self.IEN)+1), dtype='float')


        #     X_[self.IEN_orig] = self.X[self.IEN_orig]
        #     Y_[self.IEN_orig] = self.Y[self.IEN_orig]
        #     for i in range (3,6):
        #         j=i-2
        #         if i == 5:
        #             j = 0
        #         X_[self.IEN[:,i]]  = (self.X[self.IEN[:,j]] + self.X[self.IEN[:,i-3]])/2.0
        #         Y_[self.IEN[:,i]]  = (self.Y[self.IEN[:,j]] + self.Y[self.IEN[:,i-3]])/2.0
        #     self.X = X_
        #     self.Y = Y_

        #     # print(self.X[self.IEN])
        #     # print(self.Y[self.IEN])
        #     # print('\n')

        #     # L=np.isin(self.IEN_orig,self.IENbound)
        #     # index = np.where(L.sum(axis=1) == 2)[0]
        #     self.IENbound_orig = self.IENbound.copy()
        #     self.IENbound = np.zeros((self.IENbound_orig.shape[0],3), dtype='int')
        #     self.IENbound[:,:2] = self.IENbound_orig

        #     for i in range(len(self.IEN)):
        #         for j in range(len(self.IENbound_orig)):
        #             p = np.isin(self.IEN_orig[i],self.IENbound_orig[j])
        #             if np.sum(p) == 2:
        #                 p = np.where(p)[0]
        #                 if (np.abs(p[0] - p[1])) > 1:
        #                     self.IENbound[j,2] = self.IEN[i,5]
        #                 else:
        #                     self.IENbound[j,2] = self.IEN[i,np.min(p)+3]


        #     self.npoints = np.max(self.IEN) + 1
        
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

    def set_boundary_prior(self,BC,outflow):
        
        self.BC = BC
        k = 0
        bound_list = []
        outflow_bound_list = []
        for bound in BC:
            bound_list.append([])
            for i in range (len(self.IENboundElem)):
                if self.IENboundElem[i] == bound['name']:
                    for l in range (len(self.IENbound[i])):
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





