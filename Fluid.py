import numpy as np

class Fluid:
    
    def __init__(self,mesh,Re,Pr,Ga,Gr,IC,Da=0,Fo=0):
        
        self.Re = Re
        self.Pr = Pr
        self.Ga = Ga
        self.Gr = Gr
        self.Da = Da
        self.Fo = Fo
        
        if not type(IC['vx']) == np.ndarray:
            self.vx = float(IC['vx'])*np.ones((mesh.npoints), dtype='float')
            self.vy = float(IC['vy'])*np.ones((mesh.npoints), dtype='float')
            self.p = float(IC['p'])*np.ones((mesh.npoints_p), dtype='float')
            if mesh.mesh_kind == 'quad':
                self.T = float(IC['T'])*np.ones((mesh.npoints), dtype='float')
            if mesh.mesh_kind == 'mini':
                self.T = float(IC['T'])*np.ones((mesh.npoints_p), dtype='float')
        else:           
            self.vx = IC['vx']
            self.vy = IC['vy']
            self.p = IC['p']
            self.T = IC['T']


        self.T_mini = np.zeros((mesh.npoints), dtype='float')
        self.p_quad = np.zeros((mesh.npoints), dtype='float')

        self.vxd = np.zeros((mesh.npoints), dtype='float')
        self.vyd = np.zeros((mesh.npoints), dtype='float')
        self.Td = np.zeros((mesh.npoints), dtype='float')