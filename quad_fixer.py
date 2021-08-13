import meshio
import matplotlib.pyplot as plt
import time
import numpy as np

plt.close('all')

msh = meshio.read('mixeddarcy.msh')
Xo = np.array(msh.points[:,0])
Yo = np.array(msh.points[:,1])
numNodes = len(Xo)
IENo= np.array(msh.cells['triangle6'])
numElems = len(IENo)
IENboundo= msh.cells['line3']
numElemsb = len(IENboundo)

npoints = np.max(IENo) + 1

converter = []
IEN = IENo.copy()
IENbound = IENboundo.copy()
X = Xo.copy()
Y = Yo.copy()

for i in range(numElemsb):    
    p = np.where(IENo == np.max(IENo[:,0:3]) - i)
    p2 = np.where(IENo == IENboundo[numElemsb - 1 - i,2])
    converter.append([np.max(IENo[:,0:3]) - i,IENboundo[numElemsb - 1 - i,2]])
    for j in range(len(p[0])):
        if p[1][j] > 2:
            print("Refine mesh!")
        index = [p[0][j],p[1][j]]
        IEN[index[0],index[1]] = converter[i][1]
    for j in range(len(p2[0])):
        index = [p2[0][j],p2[1][j]]
        IEN[index[0],index[1]] = converter[i][0]
    
    X[converter[i][0]] = Xo[converter[i][1]]
    X[converter[i][1]] = Xo[converter[i][0]]
    Y[converter[i][0]] = Yo[converter[i][1]]
    Y[converter[i][1]] = Yo[converter[i][0]]
    IENbound[numElemsb - 1 - i,2] = converter[i][0]

npoints = np.max(IEN) + 1
npoints_p = np.max(IEN[:,0:3]) + 1
    
IENboundTypeElem = list(msh.cell_data['line3']['gmsh:physical'] - 8)
boundNames = list(msh.field_data.keys())
IENboundElem = [boundNames[elem] for elem in IENboundTypeElem]

# cria lista de nos do contorno
cc = np.unique(IENbound.reshape(IENbound.size))

# plot 
ax = plt.triplot(X[0:npoints_p],Y[0:npoints_p],IEN[:,0:3],color='k',linewidth=0.5)
ax = plt.plot(X,Y,'bo')
ax = plt.plot(X[cc],Y[cc],'ko')
plt.gca().set_aspect('equal')
plt.show()   
    
    
    
    
