import numpy as np

class Particle:
    
    def __init__(self,ID,d,rho,pos,v=[0,0]):
        
        self.ID = ID
        self.d = d        
        self.rho = rho
        self.vol = (4.0/3.0)*np.pi*(0.5*self.d)**3
        self.m = self.rho*self.vol
        self.pos = pos
        self.v = np.array(v)
        
        self.Cd = 0
        self.D = 0
        self.E = 0
        self.F = 0
        self.v_f = 0
        self.element = 0
        self.stop = False
        self.delete = False
        
    def calc_Cd(self,Re):
        
        self.Re_p = Re*np.linalg.norm(self.v-self.v_f)*self.d
        
        if self.Re_p > 0.0:
            if self.Re_p<1000:
                self.Cd = (24.0/self.Re_p)*(1+0.15*self.Re_p**0.687)
            else:
                self.Cd = 0.44
        else:
            self.Cd = 0.0

    def calc_Drag(self):
        self.D = (3.0/4.0)*self.Cd*(1.0/self.rho)*(self.m/self.d)*np.linalg.norm(self.v-self.v_f)*(self.v_f-self.v)
        
    def calc_E(self,Fr):
        self.E = (1.0/Fr**2)*np.array([0,-1.0*(np.pi/6.0)*(self.rho-1.0)*self.d**3])
        
    def calc_v(self,Re,Fr,dt):
        self.calc_vf()
        self.calc_Cd(Re)
        self.calc_Drag()
        self.calc_E(Fr)
        self.F = self.D + self.E
  
        self.v = self.v + (self.F)*dt/self.m
    def calc_pos(self,dt):
        self.pos = self.pos + self.v*dt

        
    def calc_vf(self):
        self.v_f = baricentric_v(self.pos,self.element)
        
def baricentric_v(point,element):
    point1 = element.nodes[0]
    point2 = element.nodes[1]
    point3 = element.nodes[2]
    
    p = point
    p1 = np.array([point1.x,point1.y])
    p2 = np.array([point2.x,point2.y])
    p3 = np.array([point3.x,point3.y])
    
    A = Area(p1,p2,p3)
    A1 = Area(p,p2,p3)
    A2 = Area(p,p1,p3)
    A3 = Area(p,p1,p2)
    
    vx = (A1*point1.vx + A2*point2.vx + A3*point3.vx)/A
    vy = (A1*point1.vy + A2*point2.vy + A3*point3.vy)/A
    
    v_f = np.array([vx,vy])
    return v_f
    
def Area(p1,p2,p3):
    Matriz = np.array([[1,p1[0],p1[1]],
                        [1,p2[0],p2[1]],
                        [1,p3[0],p3[1]]],dtype='float')
    A = 0.5*np.linalg.det(Matriz)
    return np.abs(A)
            
    
