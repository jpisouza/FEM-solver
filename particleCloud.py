import numpy as np
import math
from particle import Particle
from timeit import default_timer as timer

class ParticleCloud:
    
    def __init__(self,elements,nodes,pos_vector,v_vector,d_vector,rho_vector,forces,two_way=True):
        
        self.x = pos_vector
        self.v = v_vector
        self.d = d_vector
        self.rho = rho_vector
        self.elements = elements
        self.nodes = nodes
        
        self.forces = forces
        self.two_way = two_way
        
        self.particle_list = []
        for i in range(self.x.shape[0]):
            particle = Particle(i,self.d[i],self.rho[i],self.x[i],self.v[i])
            self.particle_list.append(particle)

           
        self.set_element()
    
    def set_element(self):
        
        for part in self.particle_list:
            # print(str(part.pos[0]) + ' =  ' + str(type(part.pos[0])))
            if math.isnan(part.pos[0]):
                part.stop = True
                part.delete = True
                continue
            if part.element == 0:
                for e in self.elements:
                    if pointInElement(part,e):
                        part.element = e
                        break
            elif not part.delete:
                dist_min = 1e20
                for point in part.element.nodes:
                    dist = dist_point(part.pos,point)
                    if dist < dist_min:
                        dist_min = dist
                        p = point
                        
                p_ant = 0
                p_2ant = 0
                count = 0
                while count < 100:
                    count+=1
                    flag_break = False
                    for e in p.lista_elem:
                        if pointInElement(part,self.elements[e]):
                            part.element = self.elements[e]
                            if len(part.element.mesh.porous_elem) > 0 and part.element.mesh.porous_elem[e] == 1:
                                part.v = [0,0]
                                part.stop = True
                            flag_break = True
                            break
                    if flag_break:
                        break
                    
                    dist_min = 1e20
                    p_2ant = p_ant
                    p_ant = p
                    p_aux = p
                    for vizinho in np.reshape(p_aux.lista_pontos,2*len(p_aux.lista_pontos)):
                        dist = dist_point(part.pos,self.nodes[vizinho])
                        if dist < dist_min:
                            dist_min = dist
                            p = self.nodes[vizinho]
                            
                    if p == p_2ant:
                        for e in p.lista_elem:
                            if p_2ant in self.elements[e].nodes:
                                part.element = self.elements[e]
                                # part.pos[0] = np.abs((part.pos[0]-p.x)/(p_2ant.x-p.x))*p_2ant.x + np.abs((part.pos[0]-p_2ant.x)/(p_2ant.x-p.x))*p.x
                                # part.pos[1] = np.abs((part.pos[1]-p.y)/(p_2ant.y-p.y))*p_2ant.y + np.abs((part.pos[1]-p_2ant.y)/(p_2ant.y-p.y))*p.y
                                part.pos[0] = (p.x + p_2ant.x)/2.0
                                part.pos[1] = (p.y + p_2ant.y)/2.0
                                part.v = [0,0]
                                part.stop = True
                                if p.out_flow:
                                    part.delete = True
                                flag_break = True
                                break
                        if flag_break:
                            break

    
    def calc_force_vector(self):
        for part in self.particle_list:
            if not math.isnan(part.pos[0]):
                self.calc_F(part.pos,part.element,part.F,part.vol)
            
            
            
    def solve(self,dt,nLoop,Re,Fr):
        dt_ = dt/float(nLoop)
        self.x = np.zeros((len(self.particle_list),2), dtype='float') 
        # print(self.particle_list[0].v_f - self.particle_list[0].v)
        # print(self.particle_list[0].pos)
        # print(self.particle_list[0].m)
        # print('\n')
        for n in range(nLoop):             
            # list_del = []
            # count = 0
            for particle in self.particle_list:
                if not particle.stop:
                    particle.calc_v(Re,Fr,dt_)
                    particle.calc_pos(dt_)
                if particle.delete:
                    particle.pos = [float('NaN'),float('NaN')]

                self.x[particle.ID,:] = particle.pos
                self.v[particle.ID,:] = particle.v

                # print (self.particle_list[0].Re_p)
                # print (self.particle_list[0].D)
                # print (self.particle_list[0].E)
                # print (self.particle_list[0].Cd)
                # print (self.particle_list[0].v_f - self.particle_list[0].v)
                # print(self.particle_list[0].pos)
                # print('\n')
            if n == (nLoop - 1):                
                self.forces = np.zeros((np.max(self.nodes[0].IEN)+1,2), dtype = 'float')
                if self.two_way:
                    start = timer()
                    self.calc_force_vector()
                    end = timer()
                    print('time --> Calculate two-way forces = ' + str(end-start) + ' [s]')
            self.set_element()
        
                        
    def calc_F(self,point,element,force,volume):
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
        
        L1 = A1/A
        L2 = A2/A
        L3 = A3/A
        
        if element.IEN.shape[1] <= 4:
            N1 = L1 - 9.0*L1*L2*L3
            N2 = L2 - 9.0*L1*L2*L3
            N3 = L3 - 9.0*L1*L2*L3
            N4 = 27.0*L1*L2*L3
            
            N = [N1,N2,N3,N4]
            
            for i in range (3):
                self.forces[element.nodes[i].ID,:] -= N[i]*force/volume
            self.forces[element.centroide.ID,:] -= N4*force/volume
        
        else:
            N1 = (2*L1-1.0)*L1
            N2 = (2*L2-1.0)*L2
            N3 = (2*L3-1.0)*L3
            N4 = 4*L1*L2
            N5 = 4*L2*L3
            N6 = 4*L1*L3
            
            N = [N1,N2,N3,N4,N5,N6]
            
            for i in range (3):
                self.forces[element.nodes[i].ID,:] -= N[i]*force/volume
                self.forces[element.edges[i].ID,:] -= N[i+3]*force/volume
 


def pointInElement(particle,element):
    p = particle.pos
    v1 = element.nodes[0]
    v2 = element.nodes[1]
    v3 = element.nodes[2]
    
    a = np.array([v1.x,v1.y])
    b = np.array([v2.x,v2.y])
    c = np.array([v3.x,v3.y])
    if sameSide(p,a,b,c) and sameSide(p,b,a,c) and sameSide(p,c,a,b):
        return True
    else:
        return False

def sameSide(p1,p2,a,b):
    cp1 = np.cross(b-a,p1-a)
    cp2 = np.cross(b-a,p2-a)
    if np.dot(cp1,cp2)>=0:
        return True
    else:
        return False

def dist_point(x,point):
    dist = np.sqrt((x[0]-point.x)**2 + (x[1]-point.y)**2)
    return dist

def Area(p1,p2,p3):
    Matriz = np.array([[1,p1[0],p1[1]],
                        [1,p2[0],p2[1]],
                        [1,p3[0],p3[1]]],dtype='float')
    A = 0.5*np.linalg.det(Matriz)
    return np.abs(A)