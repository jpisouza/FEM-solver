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

filename = 'channelHole.msh'
msh = meshio.read('./' + filename)
Xo = np.array(msh.points[:,0])
Yo = np.array(msh.points[:,1])
numNodes = len(Xo)
IENo= np.array(msh.cells['triangle6'])
numElems = len(IENo)
IENboundo= msh.cells['line3']
numElemsb = len(IENboundo)

# cria lista de vertices e arestas (malha triangulo)
flatv = IENo[:,0:3].flatten()
flate = IENo[:,3:6].flatten()
vertlist = np.unique(flatv)
edgelist = np.unique(flate)
numVerts = len(vertlist) 
numEdges = len(edgelist) 
convplen = vertlist.max()+1
convelen = edgelist.max()+1
convertp = -1*np.ones((convplen),dtype='int')
converte = -1*np.ones((convelen),dtype='int')

# malha de triangulos
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

IEN = np.zeros( (numElems,6),dtype='int' )
for i in range(0,numElems):
 for j in range(0,3):
  IEN[i,j] = convertp[ IENo[i,j] ]
  IEN[i,j+3] = converte[ IENo[i,j+3] ]

# cria lista de vertices e arestas (malha contorno)
flatbv = IENboundo[:,0:2].flatten()
flatbe = IENboundo[:,2:3].flatten()
bvertlist = np.unique(flatbv)
bedgelist = np.unique(flatbe)
numVertsb = len(bvertlist) 
numEdgesb = len(bedgelist) 
numNodesb = numVertsb+numEdgesb
convbplen = bvertlist.max()+1
convbelen = bedgelist.max()+1
convertbp = -1*np.ones((convbplen),dtype='int')
convertbe = -1*np.ones((convbelen),dtype='int')
countb = 0
for v in bvertlist:
 convertbp[v] = countb
 countb += 1

countb = 0
for e in bedgelist:
 convertbe[e] = countb + numVerts
 countb += 1

IENbound = np.zeros( (numElemsb,3),dtype='int' )
for i in range(0,numElemsb):
 for j in range(0,2):
  IENbound[i,j] = convertbp[ IENboundo[i,j] ]
  IENbound[i,j+1] = convertbe[ IENboundo[i,j+1] ]


IENboundTypeElem = list(msh.cell_data['line3']['gmsh:physical'] - 1)
boundNames = list(msh.field_data.keys())
IENboundElem = [boundNames[elem] for elem in IENboundTypeElem]

# cria lista de nos do contorno
cc = np.unique(IENbound.reshape(IENbound.size))

#--------------------------------------------------
# # plot 
# ax = plt.triplot(X[0:numVerts],Y[0:numVerts],IEN[:,0:3],color='k',linewidth=0.5)
# ax = plt.plot(X,Y,'bo')
# for e in range(0,IENbound.shape[0]):
#  v1,v2,v3 = IENbound[e]
#  x1 = X[v1];y1 = Y[v1]
#  x2 = X[v2];y2 = Y[v2]
#  x3 = X[v3];y3 = Y[v3]
#  xMid = (x1+x2)/2.0
#  yMid = (y1+y2)/2.0
#  plt.text(xMid,yMid,IENboundElem[e],color='r')
# plt.gca().set_aspect('equal')
# plt.show()
#-------------------------------------------------- 

ccName = [[] for i in range( len(X) )]
for elem in range(0,len(IENbound)):
 if IENboundElem[elem] == 'wallInflow' or \
    IENboundElem[elem] == 'wallOutflow' or \
    IENboundElem[elem] == 'wallRotating':
  for j in range(0,IENbound.shape[1]):
   ccName[ IENbound[elem][j] ] = IENboundElem[elem]
 if IENboundElem[elem] == 'wallRotating':
  for j in range(0,IENbound.shape[1]):
   ccName[ IENbound[elem][j] ] = IENboundElem[elem]
for elem in range(0,len(IENbound)):
 if IENboundElem[elem] == 'wallNoSlip':
  for j in range(0,IENbound.shape[1]):
   ccName[ IENbound[elem][j] ] = IENboundElem[elem]
for elem in range(0,len(IENbound)):
 if IENboundElem[elem] == 'wallNoSlipPressure':
  for j in range(0,IENbound.shape[1]):
   ccName[ IENbound[elem][j] ] = IENboundElem[elem]
for elem in range(0,len(IENbound)):
 if IENboundElem[elem] == 'wallSlip':
  for j in range(0,IENbound.shape[1]):
   ccName[ IENbound[elem][j] ] = IENboundElem[elem]

# plot 
ax = plt.triplot(X[0:numVerts],Y[0:numVerts],IEN[:,0:3],color='k',linewidth=0.5)
ax = plt.plot(X,Y,'bo')
for i in cc:
 x = X[i];y = Y[i]
 plt.text(x,y,str(i),color='k')
 #plt.text(x,y,ccName[i],color='r')
 ax = plt.plot(X[i],Y[i],'ro')
plt.gca().set_aspect('equal')
plt.show()

# -----------------------------------------------------------------------------

