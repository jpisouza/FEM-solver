## =================================================================== ##
#  this is file SL.py, created at 19-Feb-2021                           #
#  maintained by Gustavo Rabello dos Anjos                              #
#  e-mail: gustavo.rabello@gmail.com                                    #
## =================================================================== ##

# Classe SL para navierStokes2dFEM, navierStokes2dFEMGauss and 3D version
# compute -> gera matriz de conveccao conv(numVerts,numVerts)
# getDepartElem -> define a trajetoria para tras e inicia busca
# jumpToElem -> busca de trajetoria
# computeIntercept -> se ponto cair fora do dominio, interpola no contorno
# testElement -> verifica se eh o elemento que contem o ponto
# setCentroid -> interpola valores de numVerts nos centroids
# setBC -> ajuste de c.c. de velocidade em vxd,vyd

import numpy as np
from scipy.sparse import lil_matrix,csr_matrix

class Linear:
 def __init__(_self,_IEN,_X,_Y,_neighborElem,_oface,_velU,_velV):

  _self.X = _X
  _self.Y = _Y
  _self.IEN = _IEN
  _self.numNodes = len(_X)
  _self.numElems = len(_IEN)
  _self.numVerts = _self.numNodes-_self.numElems
  _self.neighborElem = _neighborElem
  _self.oface = _oface
  _self.velU = _velU
  _self.velV = _velV

  _self.l1 = 0.0
  _self.l2 = 0.0
  _self.l3 = 0.0

  _self.conv  = lil_matrix((_self.numVerts,_self.numVerts), dtype='float32')

 def compute(_self,_dt):
  _self.getDepartElem(_dt)

 def getDepartElem(_self,_dt):
  for i in range(0,_self.numVerts):
   mele = _self.neighborElem[i][0]
   xp = _self.X[i] - _self.velU[i]*_dt
   yp = _self.Y[i] - _self.velV[i]*_dt
   
   _self.jumpToElem(mele,i,xp,yp)

 def jumpToElem(_self,_destElem,_iiVert,_R2X,_R2Y):
  v1 = _self.IEN[_destElem][0]
  v2 = _self.IEN[_destElem][1]
  v3 = _self.IEN[_destElem][2]
 
  if _self.testElement(_destElem,_R2X,_R2Y):
   _self.conv[_iiVert,v1] = _self.l1
   _self.conv[_iiVert,v2] = _self.l2
   _self.conv[_iiVert,v3] = _self.l3
  else:
   if (_self.l1<=_self.l2) and (_self.l1<=_self.l3):
    vjump=0
    ib1=v2
    ib2=v3
   if (_self.l2<=_self.l1) and (_self.l2<=_self.l3):
    vjump=1
    ib1=v1
    ib2=v3
   if (_self.l3<=_self.l1) and (_self.l3<=_self.l2): 
    vjump=2 
    ib1=v1
    ib2=v2

   if _self.oface[_destElem,vjump] != -1:
    _self.jumpToElem(_self.oface[_destElem,vjump],_iiVert,_R2X,_R2Y)
   else:
    Bl1,Bl2 = _self.computeIntercept(_iiVert,_R2X,_R2Y,ib1,ib2)
    _self.conv[_iiVert,ib1] = Bl1
    _self.conv[_iiVert,ib2] = Bl2

 def computeIntercept(_self,_i,_R2X,_R2Y,_ib1,_ib2):
  R1X = _self.X[_i]
  R1Y = _self.Y[_i]

  B1X = _self.X[_ib1]
  B2X = _self.X[_ib2]
  B1Y = _self.Y[_ib1]
  B2Y = _self.Y[_ib2]

  a1 = B1X-B2X; b1 = R1X-_R2X; c1 = R1X-B2X
  a2 = B1Y-B2Y; b2 = R1Y-_R2Y; c2 = R1Y-B2Y

  det  = (a1*b2) - (a2*b1)
  detx = (c1*b2) - (c2*b1) 

  x1 = detx/det

  # return C++ equivalent Bl1 and Bl2 (see femSIM2d)
  return x1,1.0-x1

 def testElement(_self,_mele,_xp,_yp):
  v1 = _self.IEN[_mele][0]
  v2 = _self.IEN[_mele][1]
  v3 = _self.IEN[_mele][2]
  EPSlocal = 1e-04

  A = (1.0/2.0) * ( _self.X[v2]*_self.Y[v3] +
                    _self.X[v1]*_self.Y[v2] +
                    _self.Y[v1]*_self.X[v3] -
                    _self.X[v1]*_self.Y[v3] -
                    _self.Y[v2]*_self.X[v3] -
                    _self.Y[v1]*_self.X[v2] )

  A1 = (1.0/2.0) * ( _self.X[v2]*_self.Y[v3] +
                     _xp*_self.Y[v2] +
                     _yp*_self.X[v3] -
                     _xp*_self.Y[v3] -
                     _self.Y[v2]*_self.X[v3] -
                     _yp*_self.X[v2] )

  
  A2 = (1.0/2.0) * ( _xp*_self.Y[v3] +
                     _self.X[v1]*_yp +
                     _self.Y[v1]*_self.X[v3] -
                     _self.X[v1]*_self.Y[v3] -
                     _yp*_self.X[v3] - 
                     _self.Y[v1]*_xp )

  _self.l1 = A1/A
  _self.l2 = A2/A
  _self.l3 = 1.0 - _self.l2 - _self.l1

  return ( (_self.l1>=0.0-EPSlocal) and (_self.l1<=1.0+EPSlocal) and \
         (  _self.l2>=0.0-EPSlocal) and (_self.l2<=1.0+EPSlocal) and \
         (  _self.l3>=0.0-EPSlocal) and (_self.l3<=1.0+EPSlocal) )

 def setCentroid(_self,_vxd,_vyd):
  # interpolando no centroid
  zeros = np.zeros( (_self.numNodes-_self.numVerts),dtype='float')
  _vxd = np.append(_vxd,zeros)
  _vyd = np.append(_vyd,zeros)
  for e in range(0,_self.numElems):
   [v1,v2,v3,v4] = _self.IEN[e]
   _vxd[v4] = (_vxd[v1] + _vxd[v2] + _vxd[v3])/3.0
   _vyd[v4] = (_vyd[v1] + _vyd[v2] + _vyd[v3])/3.0
  return [_vxd,_vyd]

 def setBC(_self,_idv,_vbc,_vd):
  # impondo c.c. de velocidade
  # condicao de contorno (Dirichlet) para vxd,vyd 
  for i in _idv:
   _vd[i] = _vbc[i]
  return _vd

