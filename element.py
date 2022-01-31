from node import Node
import numpy as np

class Element:
    
    elem_list = []
    
    def __init__(self,ID,IEN,IEN_orig, pontos,mesh):
        
        self.mesh = mesh
        self.ID = ID
        self.IEN = IEN
        self.IEN_orig = IEN_orig
        self.pontos = pontos
        self.nodes = self.nodes_elem()
        self.quad = False
        if self.IEN.shape[1] > 4:
            self.quad = True
            self.edges, self.edges_dict = self.nodes_edges()
            for i in range (len(self.edges)):
                self.mesh.porous_nodes[self.edges[i].ID] = (self.mesh.porous_nodes[self.edges_dict[self.edges[i].ID][0]] + self.mesh.porous_nodes[self.edges_dict[self.edges[i].ID][1]])/2.0                 
        else:
            x_c = 0.0
            y_c = 0.0
            
            for i in range (len(self.nodes)):
                x_c += self.nodes[i].x
                y_c += self.nodes[i].y
            x_c = x_c/3.0
            y_c = y_c/3.0
            
            self.centroide = Node(IEN[ID,3],IEN,IEN_orig,x_c,y_c)
            if len(self.mesh.porous_list) > 0:
                if self.mesh.porous_elem[self.ID] == 1:
                    self.mesh.porous_nodes[IEN[ID,3]] = np.sum(self.mesh.porous_nodes[IEN[ID]])/3.0
        
        Element.elem_list.append(self)
        
    def nodes_elem(self):       
        lista = []
        for i in range(len(self.IEN_orig[self.ID])):
            lista.append(self.pontos[self.IEN_orig[self.ID,i]])
            if len(self.mesh.porous_list) > 0:
                if self.mesh.porous_elem[self.ID] == 1:
                    self.mesh.porous_nodes[self.IEN_orig[self.ID,i]] = 1
                    if self.mesh.X[self.IEN_orig[self.ID,i]] == 0.0:
                        self.mesh.porous_nodes[self.IEN_orig[self.ID,i]] = 0.5
                    
        return lista
    
    def nodes_edges(self):
        dic = {}
        lista = []
        for i in range(3,6):
            j=i-3
            lista.append(self.pontos[self.IEN[self.ID,i]])
            dic[self.IEN[self.ID,i]] = (self.IEN[self.ID,j],self.IEN[self.ID,j+1])
            if len(self.mesh.porous_list) > 0:
                if self.mesh.porous_elem[self.ID] == 1:
                    self.mesh.porous_nodes[self.IEN[self.ID,i]] = 1
                    if self.mesh.X[self.IEN[self.ID,i]] == 0.0:
                        self.mesh.porous_nodes[self.IEN[self.ID,i]] = 0.5
        return lista, dic
        
