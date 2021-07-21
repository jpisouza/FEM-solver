import numpy as np
import math
from particle import Particle

class ParticleCloud:
    
    def __init__(self,elements,nodes,pos_vector,d_vector,rho_vector):
        
        self.x = pos_vector
        self.d = d_vector
        self.rho = rho_vector
        self.elements = elements
        self.nodes = nodes
        
        self.particle_list = []
        for i in range(self.x.shape[0]):
            particle = Particle(i,self.d[i],self.rho[i],self.x[i])
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
                    if pointInElement(part.pos,e):
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
                        if pointInElement(part.pos,self.elements[e]):
                            part.element = self.elements[e]
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

    
    def solve(self,dt,nLoop,Re,Fr):
        dt_ = dt/float(nLoop)
        x = np.zeros((len(self.particle_list),2), dtype='float') 
        for n in range(nLoop):             
            # list_del = []
            # count = 0
            for particle in self.particle_list:
                if not particle.stop:
                    particle.calc_v(Re,Fr,dt_)
                    particle.calc_pos(dt_)
                if particle.delete:
                    particle.pos = [float('NaN'),float('NaN')]
                    # list_del.append(count)
                x[particle.ID,:] = particle.pos
                # count += 1
                
            # self.particle_list = np.delete(self.particle_list,list_del)
            # x = np.delete(x,list_del,axis = 0)
            self.set_element()
        return x
                    
            

def pointInElement(p,element):
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