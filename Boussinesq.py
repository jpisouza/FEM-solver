import numpy as np
from semi_lagrangiano import calc_vd, semi_lagrange, semi_lagrange2, semi_lagrange3
from node import Node
from element import Element
from particleCloud import ParticleCloud
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import os
from pyevtk.hl import pointsToVTK

plt.close('all')

Pr = 70.0
Ga = 10
Gr = 100.0
rho = 1.0

# geracao de malha

msh = meshio.read('cavity.msh')
X = msh.points[:,0]
Y = msh.points[:,1]

t = np.linspace(0,30.0,300)
dt=t[1]-t[0]


npoints_p = len(X)
    
IEN = msh.cells['triangle']
IEN_orig = IEN.copy()
ne=len(IEN)




IENbound = msh.cells['line']
IENboundTypeElem = list(msh.cell_data['line']['gmsh:physical'] - 1)
boundNames = list(msh.field_data.keys())
IENboundElem = [boundNames[elem] for elem in IENboundTypeElem]

upper_bound = np.array((),dtype='int')
lower_bound = np.array((),dtype='int')
left_bound = np.array((),dtype='int')
right_bound = np.array((),dtype='int')



for i in range (len(IENboundElem)):
    if IENboundElem[i] == 'upper_bound':
        if IENbound[i,0] not in upper_bound:
            upper_bound= np.append(upper_bound, IENbound[i,0])
        if IENbound[i,1] not in upper_bound:
            upper_bound = np.append(upper_bound, IENbound[i,1])
    elif IENboundElem[i] == 'lower_bound':
        if IENbound[i,0] not in lower_bound:
            lower_bound= np.append(lower_bound, IENbound[i,0])
        if IENbound[i,1] not in lower_bound:
            lower_bound = np.append(lower_bound, IENbound[i,1])
    elif IENboundElem[i] == 'left_bound':
        if IENbound[i,0] not in left_bound:
            left_bound= np.append(left_bound, IENbound[i,0])
        if IENbound[i,1] not in left_bound:
            left_bound = np.append(left_bound, IENbound[i,1])
    elif IENboundElem[i] == 'right_bound':
        if IENbound[i,0] not in right_bound:
            right_bound= np.append(right_bound, IENbound[i,0])
        if IENbound[i,1] not in right_bound:
            right_bound = np.append(right_bound, IENbound[i,1])

#Acrescenta ponto central do elemento
new_elements = np.arange(npoints_p,npoints_p+ne,1).reshape(ne,1)
IEN = np.block([[IEN,new_elements]])

for e in range (ne):
    X = np.append(X,(X[IEN[e,0]]+X[IEN[e,1]]+X[IEN[e,2]])/3.0)
    Y = np.append(Y,(Y[IEN[e,0]]+Y[IEN[e,1]]+Y[IEN[e,2]])/3.0)
    
for ID in range (len(X)):
    node = Node(ID,IEN,IEN_orig, X[ID],Y[ID])
for ID in range (len(IEN)):
    element = Element(ID,IEN,IEN_orig,Node.node_list)

npoints = len(X)

# alocacao de espaco para matriz global de rigidez K
K = np.zeros( (npoints,npoints),dtype='float' )
M = np.zeros( (npoints,npoints),dtype='float' )
Gx = np.zeros( (npoints,npoints_p),dtype='float' )
Gy = np.zeros( (npoints,npoints_p),dtype='float' )
M_T = np.zeros( (npoints_p,npoints_p),dtype='float' )
K_T = np.zeros( (npoints_p,npoints_p),dtype='float' )
Gx_T = np.zeros( (npoints_p,npoints_p),dtype='float' )
Gy_T = np.zeros( (npoints_p,npoints_p),dtype='float' )

# loop de elementos finitos
for e in range(0,ne):
     v0,v1,v2,v3 = IEN[e]

     x0 = X[v0]
     x1 = X[v1]
     x2 = X[v2]
     
     y0 = Y[v0]
     y1 = Y[v1]
     y2 = Y[v2]
     
     b0 = y1 - y2
     b1 = y2 - y0
     b2 = y0 - y1
     
     c0 = x2 - x1
     c1 = x0 - x2
     c2 = x1 - x0
     
     Matriz = np.array([[1,x0,y0],
                        [1,x1,y1],
                        [1,x2,y2]],dtype='float')
     A = 0.5*np.linalg.det(Matriz)
     
     
    
     kx_lin = (1.0/(4.0*A))*np.array( [[b0**2,b0*b1,b0*b2],
                                     [b1*b0,b1**2,b1*b2],
                                     [b2*b0,b2*b1,b2**2]],dtype='float')
     ky_lin = (1.0/(4.0*A))*np.array( [[c0**2,c0*c1,c0*c2],
                                     [c1*c0,c1**2,c1*c2],
                                     [c2*c0,c2*c1,c2**2]],dtype='float')
     
     zx = (1.0/(4.0*A))*(b1**2+b1*b2+b2**2)
     zy = (1.0/(4.0*A))*(c1**2+c1*c2+c2**2)
     
     kx_aux = np.append(kx_lin +(9.0/10.0)*zx,[[-27.0*zx/10.0,-27.0*zx/10.0,-27.0*zx/10.0]],axis=0)
     kx = np.append(kx_aux,[[-27.0*zx/10.0],[-27.0*zx/10.0],[-27.0*zx/10.0],[81.0*zx/10.0]],axis=1)
                    
     ky_aux = np.append(ky_lin +(9.0/10.0)*zy,[[-27.0*zy/10.0,-27.0*zy/10.0,-27.0*zy/10.0]],axis=0)
     ky = np.append(ky_aux,[[-27.0*zy/10.0],[-27.0*zy/10.0],[-27.0*zy/10.0],[81.0*zy/10.0]],axis=1)
     
     kelem = kx + ky
     
     kelem_T = kx_lin + ky_lin
     
     melem = (A/840.0)*np.array( [[83,13,13,45],
                                 [13,83,13,45],
                                 [13,13,83,45],
                                    [45,45,45,243]],dtype='float')
     melem_T = (A/12.0)*np.array( [[2,1,1],
                                 [1,2,1],
                                 [1,1,2]],dtype='float')
     
     gx_lin=(1.0/6.0)*np.array([[b0,b1,b2],
                            [b0,b1,b2],
                            [b0,b1,b2]],dtype='float')
     
     gy_lin=(1.0/6.0)*np.array([[c0,c1,c2],
                            [c0,c1,c2],
                            [c0,c1,c2]],dtype='float')

     gx = np.append((-9.0/20.0)*gx_lin - np.transpose(gx_lin),[[(9.0/40.0)*b0,(9.0/40.0)*b1,(9.0/40.0)*b2]],axis=0)
     gy = np.append((-9.0/20.0)*gy_lin - np.transpose(gy_lin),[[(9.0/40.0)*c0,(9.0/40.0)*c1,(9.0/40.0)*c2]],axis=0)
     
     for ilocal in range(0,4):
         iglobal = IEN[e,ilocal]
         for jlocal in range(0,4):
             jglobal = IEN[e,jlocal]
    
             K[iglobal,jglobal] = K[iglobal,jglobal] + kelem[ilocal,jlocal]             
             M[iglobal,jglobal] = M[iglobal,jglobal] + melem[ilocal,jlocal]
             
         for jlocal in range(0,3):
             jglobal = IEN[e,jlocal]
             
             if ilocal <=2:                 
                 M_T[iglobal,jglobal] = M_T[iglobal,jglobal] + melem_T[ilocal,jlocal]
                 K_T[iglobal,jglobal] = K_T[iglobal,jglobal] + kelem_T[ilocal,jlocal]
                 Gx_T[iglobal,jglobal] = Gx_T[iglobal,jglobal] + gx_lin[ilocal,jlocal]
                 Gy_T[iglobal,jglobal] = Gy_T[iglobal,jglobal] + gy_lin[ilocal,jlocal]
                 
             Gx[iglobal,jglobal] = Gx[iglobal,jglobal] + gx[ilocal,jlocal]
             Gy[iglobal,jglobal] = Gy[iglobal,jglobal] + gy[ilocal,jlocal]
     
Dx = np.transpose(Gx)
Dy = np.transpose(Gy)



vx=np.zeros( (npoints),dtype='float' )
vy=np.zeros( (npoints),dtype='float' )
p = np.zeros( (npoints_p),dtype='float' )
T=np.zeros( (npoints_p),dtype='float' )
T_mini=np.zeros( (npoints),dtype='float' )



y_min=np.min(Y)
y_max = np.max(Y)
delta_y = y_max - y_min

#-------------------------Partículas-----------------------------------
nparticles = 50
x_part = np.zeros((nparticles,2), dtype='float')
x_part[:,0] = 0.1 + 0.8*np.random.rand(nparticles)
x_part[:,1] = 0.1 + 0.8*np.random.rand(nparticles)

d_part = 0.1*np.ones( (x_part.shape[0]),dtype='float' )
rho_part = 5.0*np.ones( (x_part.shape[0]),dtype='float' )

nLoop = 100
particleCloud = ParticleCloud(Element.elem_list,Node.node_list,x_part,d_part,rho_part)

#-----------------------------------------------------------------------

if not os.path.isdir('C:\VTK\\NS-cavity_conv'):
    os.mkdir('C:\VTK\\NS-cavity_conv')
for i in range(len(t)):
    print(i)

    vxd, vyd, Td = semi_lagrange2(Node.node_list,Element.elem_list,vx,vy,T,dt,IENbound)
         
    T_mini[0:npoints_p] = T
    for e in IEN:
        v1,v2,v3,v4 = e
        T_mini[v4] = (T[v1] + T[v2] + T[v3])/3.0
    
    Matriz_vx = np.block([[(1.0/dt)*M + K,np.zeros( (npoints,npoints),dtype='float' ), Gx]])
    vetor_vx = np.dot((1.0/dt)*M,vxd) 
    
    Matriz_vy = np.block([[np.zeros( (npoints,npoints),dtype='float' ),(1.0/dt)*M + K, Gy]])
    vetor_vy = np.dot((1.0/dt)*M,vyd) + np.dot(M,Gr*T_mini-Ga)
    
    Matriz_p =  np.block([[-Dx,-Dy,np.zeros( (npoints_p,npoints_p),dtype='float' )]])
    vetor_p = np.zeros((npoints_p),dtype='float' )
    
    # Matriz_T=(1.0/dt)*M_T + (1.0/(Re*Pr))*K_T + np.dot(np.diag(vx[0:npoints_p]),Gx_T) + np.dot(np.diag(vy[0:npoints_p]),Gy_T)
    # vetor_T = np.dot((1.0/dt)*M_T,T)
    
    Matriz_T=(1.0/dt)*M_T + (1.0/Pr)*K_T
    vetor_T = np.dot((1.0/dt)*M_T,Td[0:npoints_p])
    
    
        
    for j in lower_bound:
        vetor_vx[j] = 0
        Matriz_vx[j,:] = 0
        Matriz_vx[j,j] = 1.0
        
        vetor_vy[j] = 0
        Matriz_vy[j,:] = 0
        Matriz_vy[j,j+npoints] = 1.0
        
        # Matriz_T[j,:] = 0
        # Matriz_T[j,j] = 1.0
        # vetor_T[j] = 0.0
        
    for j in upper_bound:
        vetor_vx[j] = 0
        Matriz_vx[j,:] = 0
        Matriz_vx[j,j] = 1.0
        
        vetor_vy[j] = 0
        Matriz_vy[j,:] = 0
        Matriz_vy[j,j+npoints] = 1.0
        
        # Matriz_T[j,:] = 0
        # Matriz_T[j,j] = 1.0
        # vetor_T[j] = 0.0
    for j in left_bound:
        vetor_vx[j] = 0.0
        Matriz_vx[j,:] = 0
        Matriz_vx[j,j] = 1.0
        
        vetor_vy[j] = 0
        Matriz_vy[j,:] = 0
        Matriz_vy[j,j+npoints] = 1.0
        
        Matriz_T[j,:] = 0
        Matriz_T[j,j] = 1.0
        vetor_T[j] = -0.5
        
    for j in right_bound:
        # vetor_p[j] = 0
        # Matriz_p[j,:] = 0
        # Matriz_p[j,j+2*npoints] = 1.0
        vetor_vx[j] = 0.0
        Matriz_vx[j,:] = 0
        Matriz_vx[j,j] = 1.0
        
        vetor_vy[j] = 0
        Matriz_vy[j,:] = 0
        Matriz_vy[j,j+npoints] = 1.0
        
        Matriz_T[j,:] = 0
        Matriz_T[j,j] = 1.0
        vetor_T[j] = 0.5
        
    
    Matriz = np.block([[Matriz_vx],[Matriz_vy],[Matriz_p]])
    vetor = np.block([vetor_vx,vetor_vy,vetor_p])
    
    sol = np.linalg.solve(Matriz,vetor)
    
    T = np.linalg.solve(Matriz_T,vetor_T)
    
    vx = sol[0:npoints]
    vy = sol[npoints:2*npoints]
    p = sol[2*npoints:]

    x_part = particleCloud.solve(dt,nLoop,rho,1.0,1.0/np.sqrt(Ga))
    
    print (' --> saving solution: VTK ')
    point_data = {'p' : p[0:npoints_p]}
    data_T  = {'T' : T}
    data_v = {'v' : np.transpose(np.block([[vx[0:npoints_p]],[vy[0:npoints_p]],[np.zeros((npoints_p),dtype='float')]]))}
    point_data.update(data_T)
    point_data.update(data_v)
    meshio.write_points_cells(
    'C:\VTK\\NS-cavity_conv\sol-'+str(i)+'.vtk',
    msh.points,
    msh.cells,
    #file_format="vtk-ascii",
    point_data=point_data,
    )
    
    if x_part.shape[0] != 0:
        x_p = x_part[:,0].copy()
        y_p = x_part[:,1].copy()
        pointsToVTK("C:\VTK\\NS-cavity_conv\sol_particles"+str(i),x_p,y_p,0.1*np.ones((x_part.shape[0]), dtype='float'))




    


