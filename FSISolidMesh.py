import numpy as np

class FSISolidMesh:
    
    def __init__(self, fluidmesh, fluid):
        
        self.mesh_kind = 'quad'
        
        self.fluid = fluid
        
        self.FSI_flag = True
        self.fluidmesh = fluidmesh
        
        self.IEN_orig = []
        for e in range (len(self.fluidmesh.IEN)):
            if e in self.fluidmesh.solid_elem:
                self.IEN_orig.append(self.fluidmesh.IEN[e])
        
        self.IEN_orig = np.array(self.IEN_orig)
        
        self.IENbound_orig = []
        self.IENboundElem = []
        for e in range (len(self.fluidmesh.IENbound)):
            if self.fluidmesh.IENbound[e,0] in self.fluidmesh.solid_nodes and self.fluidmesh.IENbound[e,1] in self.fluidmesh.solid_nodes:
                self.IENbound_orig.append(self.fluidmesh.IENbound[e])
                self.IENboundElem.append(self.fluidmesh.IENboundElem[e])
        
        self.IENbound_orig = np.array(self.IENbound_orig)
        
        self.points_real = np.unique(self.IEN_orig)
        self.points = np.arange(0,len(self.points_real),1)
        self.npoints = len(self.points)
               
        self.converter = dict(zip(self.points_real,self.points))
        self.inv_converter = dict(zip(self.points,self.points_real))
        
        self.X = self.fluidmesh.X[self.points_real]
        self.Y = self.fluidmesh.Y[self.points_real]
        
        self.IEN = np.zeros((len(self.IEN_orig), len(self.IEN_orig[0])), dtype = 'int')
        for e in range(len(self.IEN_orig)):
            for i in range (len(self.IEN_orig[e])):
                self.IEN[e,i] = self.converter[self.IEN_orig[e,i]]
                
        self.ne=len(self.IEN)
        
        self.IENbound = np.zeros((len(self.IENbound_orig), len(self.IENbound_orig[0])), dtype = 'int')
        for e in range(len(self.IENbound_orig)):
            for i in range (len(self.IENbound_orig[e])):
                self.IENbound[e,i] = self.converter[self.IENbound_orig[e,i]]  
        
        self.build_bounddict()
        
    def build_bounddict(self):
        
        self.bound_dict = {}
        for bound in self.fluidmesh.FSI:
            self.bound_dict[bound['name']] = []
        
        for e in range(len(self.IENboundElem)):
            for bound in self.fluidmesh.FSI:
                if self.IENboundElem[e] == bound['name']:
                    for node in  self.IENbound[e]:
                        if node not in self.bound_dict[self.IENboundElem[e]]:
                            self.bound_dict[self.IENboundElem[e]].append(node)
    
    def build_BCdict(self,FSIForces):
        #builds the dictionary for BC inputs
        BC_dict = {}
        for i in range(len(self.fluidmesh.FSI)):
            bc_list = [self.fluidmesh.FSI[i]['ux']]
            bc_list.append(self.fluidmesh.FSI[i]['uy'])
            bc_list.append([])
            bc_list.append([])
            for j in range(len(self.bound_dict[self.fluidmesh.FSI[i]['name']])):
                bc_list[-2].append(FSIForces[self.inv_converter[self.bound_dict[self.fluidmesh.FSI[i]['name']][j]],0])
                bc_list[-1].append(FSIForces[self.inv_converter[self.bound_dict[self.fluidmesh.FSI[i]['name']][j]],1])
            bc_list[-2] = np.array(bc_list[-2])
            bc_list[-1] = np.array(bc_list[-1])
            bc_list.append(FSIForces[self.bound_dict[self.fluidmesh.FSI[i]['name']],1])
            BC_dict[self.fluidmesh.FSI[i]['name']] = bc_list
            
        return BC_dict
    
    def update_forces(self, FSIForces, BC_dict):
        for i in range(len(self.fluidmesh.FSI)):
            for j in range(len(self.bound_dict[self.fluidmesh.FSI[i]['name']])):               
                BC_dict[self.fluidmesh.FSI[i]['name']][2][j] = FSIForces[self.inv_converter[self.bound_dict[self.fluidmesh.FSI[i]['name']][j]],0]
                BC_dict[self.fluidmesh.FSI[i]['name']][3][j] = FSIForces[self.inv_converter[self.bound_dict[self.fluidmesh.FSI[i]['name']][j]],1]

