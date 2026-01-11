import numpy as np
import meshio

class SolidMesh:
    def __init__(self, meshname, BC):
        
        self.msh = meshio.read(meshname)
        self.BC = BC
        self.FSI_flag = False
        
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
            
            points = np.arange(0,self.npoints,1)
            points_real = np.unique(self.IEN)
            
            self.converter = dict(zip(points_real,points))
            
            self.IENboundTypeElem = list(self.msh.cell_data['line3']['gmsh:physical'])
            self.IENTypeElem = list(self.msh.cell_data['triangle6']['gmsh:physical'])
            self.boundNames = list(self.msh.field_data.keys())
            self.IENboundElem = [self.boundNames[self.dict_boundary[elem]] for elem in self.IENboundTypeElem]
        
        self.build_bounddict()
            
    def build_bounddict(self):
        
        self.bound_dict = {}
        for bound in self.BC:
            self.bound_dict[bound] = []
        
        for e in range(len(self.IENboundElem)):
            if self.IENboundElem[e] in self.BC:
                for node in  self.IENbound[e]:
                    if node not in self.bound_dict[self.IENboundElem[e]]:
                        self.bound_dict[self.IENboundElem[e]].append(node)
        

