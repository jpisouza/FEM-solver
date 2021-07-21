## =================================================================== ##
#  this is file navierStokes2dFEMGauss.py, created at 22-Fev-2021       #
#  maintained by Gustavo Rabello dos Anjos                              #
#  e-mail: gustavo.rabello@gmail.com                                    #
## =================================================================== ##

import meshio
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import time
import numpy as np

# -----------------------------------------------------------------------------
# leitura de malha e classificacao de contorno por nome (ccName)
start = time.time()

filename = 'channelHole.msh'
msh = meshio.read(filename)
Xo = np.array(msh.points[:,0])
Yo = np.array(msh.points[:,1])
numNodes = len(Xo)
IENo= np.array(msh.cells['triangle6'])
numElems = len(IENo)
IENboundo= msh.cells['line3']
numElemsb = len(IENboundo)

# ----- INICIO: conversao TRI6 (Gmsh) --> TRI6 (Navier-Stokes) ------------- #
# malha de triangulos
# numeracao: 
#  1) vertices dos triangulos
#  2) nos das arestas (edge)
# cria lista de vertices e arestas (malha triangulo)
vertlist = np.unique( IENo[:,0:3].flatten() )
edgelist = np.unique( IENo[:,3:6].flatten() )
numVerts = len(vertlist) 
numEdges = len(edgelist) 
convertp = -1*np.ones(( vertlist.max()+1 ),dtype='int')
converte = -1*np.ones(( edgelist.max()+1 ),dtype='int')

# vetores coordenadas X,Y
X = np.zeros( (numNodes),dtype='float' )
Y = np.zeros( (numNodes),dtype='float' )
count = 0
for v in vertlist:
 convertp[v] = count
 X[count] = Xo[v]
 Y[count] = Yo[v]
 count += 1

for e in edgelist:
 converte[e] = count
 X[count] = Xo[e]
 Y[count] = Yo[e]
 count += 1

# matriz de conectividade IEN
IEN = np.zeros( (numElems,6),dtype='int' )
for i in range(0,numElems):
 for j in range(0,3):
  IEN[i,j]   = convertp[ IENo[i,j] ]
  IEN[i,j+3] = converte[ IENo[i,j+3] ]

# cria lista de vertices e arestas (malha contorno)
# numeracao: 
#  1) vertices dos segmentos de reta
#  2) nos das arestas (edge)
bvertlist = np.unique( IENboundo[:,0:2].flatten() )
bedgelist = np.unique( IENboundo[:,2:3].flatten() )
numVertsb = len(bvertlist) 
numEdgesb = len(bedgelist) 
numNodesb = numVertsb+numEdgesb
convertbp = -1*np.ones(( bvertlist.max()+1 ),dtype='int')
convertbe = -1*np.ones(( bedgelist.max()+1 ),dtype='int')
countb = 0
for v in bvertlist:
 convertbp[v] = countb
 countb += 1

countb = 0
for e in bedgelist:
 convertbe[e] = countb + numVerts
 countb += 1

# matriz de conectividade de contorno IENbound
IENbound = np.zeros( (numElemsb,3),dtype='int' )
for i in range(0,numElemsb):
 for j in range(0,2):
  IENbound[i,j]   = convertbp[ IENboundo[i,j] ]
  IENbound[i,j+1] = convertbe[ IENboundo[i,j+1] ]
# ----- FIM: conversao TRI6 (Gmsh) --> TRI6 (Navier-Stokes) ---------------- #


IENboundTypeElem = list(msh.cell_data['line3']['gmsh:physical'] - 1)
boundNames = list(msh.field_data.keys())
IENboundElem = [boundNames[elem] for elem in IENboundTypeElem]

# cria lista de nos do contorno
cc = np.unique(IENbound.reshape(IENbound.size))

# plot 
ax = plt.triplot(X[0:numVerts],Y[0:numVerts],IEN[:,0:3],color='k',linewidth=0.5)
ax = plt.plot(X,Y,'bo')
plt.gca().set_aspect('equal')
plt.show()


