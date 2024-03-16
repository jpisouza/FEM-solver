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

class Quad:
 def __init__(_self,_IEN,_X,_Y,_neighborElem,_oface,_velU,_velV, mesh_disp):

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
  _self.mesh_disp = mesh_disp

  _self.l1 = 0.0
  _self.l2 = 0.0
  _self.l3 = 0.0

  _self.conv  = lil_matrix((_self.numNodes,_self.numNodes), dtype='float32')

 def compute(_self,_dt):
  _self.getDepartElem(_dt)

 def getDepartElem(_self,_dt):
  for i in range(0,_self.numNodes):
   mele = _self.neighborElem[i][0]
   xp = _self.X[i] - (_self.velU[i]*_dt - _self.mesh_disp[i,0])
   yp = _self.Y[i] - (_self.velV[i]*_dt - _self.mesh_disp[i,1])
   
   _self.jumpToElem(mele,i,xp,yp)

 def jumpToElem(_self,_destElem,_iiVert,_R2X,_R2Y):
  v1 = _self.IEN[_destElem][0]
  v2 = _self.IEN[_destElem][1]
  v3 = _self.IEN[_destElem][2]
  v4 = _self.IEN[_destElem][3]
  v5 = _self.IEN[_destElem][4]
  v6 = _self.IEN[_destElem][5]
 
  if _self.testElement(_destElem,_R2X,_R2Y):
   N1 = (2*_self.l1-1.0)*_self.l1;
   N2 = (2*_self.l2-1.0)*_self.l2;
   N3 = (2*_self.l3-1.0)*_self.l3;
   N4 = 4*_self.l1*_self.l2;
   N5 = 4*_self.l2*_self.l3;
   N6 = 4*_self.l1*_self.l3;
   _self.conv[_iiVert,v1] = N1
   _self.conv[_iiVert,v2] = N2
   _self.conv[_iiVert,v3] = N3
   _self.conv[_iiVert,v4] = N4
   _self.conv[_iiVert,v5] = N5
   _self.conv[_iiVert,v6] = N6
  else:
   if (_self.l1<=_self.l2) and (_self.l1<=_self.l3):
    vjump=0
    ib1=v2
    ib2=v3
    ib3=v5
   if (_self.l2<=_self.l1) and (_self.l2<=_self.l3):
    vjump=1
    ib1=v3
    ib2=v1
    ib3=v6
   if (_self.l3<=_self.l1) and (_self.l3<=_self.l2): 
    vjump=2 
    ib1=v1
    ib2=v2
    ib3=v4

   if _self.oface[_destElem,vjump] != -1:
    _self.jumpToElem(_self.oface[_destElem,vjump],_iiVert,_R2X,_R2Y)
   else:
    Bl1,Bl2 = _self.computeIntercept(_iiVert,_R2X,_R2Y,ib1,ib2)
    BN1 = (2*Bl1-1.0)*Bl1
    BN2 = (2*Bl2-1.0)*Bl2
    BN3 = 4*Bl1*Bl2
    _self.conv[_iiVert,ib1] = BN1
    _self.conv[_iiVert,ib2] = BN2
    _self.conv[_iiVert,ib3] = BN3

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

 def setBC(_self,_idv,_vbc,_vd):
  # impondo c.c. de velocidade
  # condicao de contorno (Dirichlet) para vxd,vyd 
  for i in _idv:
   _vd[i] = _vbc[i]
  return _vd

class Linear3D:
 def __init__(_self,_IEN,_X,_Y,_Z,_neighborElem,_oface,_velU,_velV,_velW):

  _self.X = _X
  _self.Y = _Y
  _self.Z = _Z
  _self.IEN = _IEN
  _self.numNodes = len(_X)
  _self.numElems = _IEN.shape[0]
  _self.numVerts = _self.numNodes-_self.numElems
  _self.neighborElem = _neighborElem
  _self.oface = _oface
  _self.velU = _velU
  _self.velV = _velV
  _self.velW = _velW

  _self.l1 = 0.0
  _self.l2 = 0.0
  _self.l3 = 0.0
  _self.l4 = 0.0

  _self.conv  = lil_matrix((_self.numVerts,_self.numVerts), dtype='float32')

 def compute(_self,_dt):
  _self.getDepartElem(_dt)

 def getDepartElem(_self,_dt):
  for i in range(0,_self.numVerts):
   mele = _self.neighborElem[i][0]
   xp = _self.X[i] - _self.velU[i]*_dt
   yp = _self.Y[i] - _self.velV[i]*_dt
   zp = _self.Z[i] - _self.velW[i]*_dt
   
   _self.jumpToElem(mele,i,xp,yp,zp)

 # _R2X = xp, _R2Y = yp, _R2Z = zp (departure point)
 def jumpToElem(_self,_destElem,_iiVert,_R2X,_R2Y,_R2Z):
  v1 = _self.IEN[_destElem][0]
  v2 = _self.IEN[_destElem][1]
  v3 = _self.IEN[_destElem][2]
  v4 = _self.IEN[_destElem][3]
 
  if _self.testElement(_destElem,_R2X,_R2Y,_R2Z):
   _self.conv[_iiVert,v1] = _self.l1
   _self.conv[_iiVert,v2] = _self.l2
   _self.conv[_iiVert,v3] = _self.l3
   _self.conv[_iiVert,v4] = _self.l4
  else:
   if (_self.l1<=_self.l2) and (_self.l1<=_self.l3) and (_self.l1<=_self.l4):
    vjump=0
    ib1=v2
    ib2=v3
    ib3=v4
   if (_self.l2<=_self.l1) and (_self.l2<=_self.l3) and (_self.l2<=_self.l4):
    vjump=1
    ib1=v3
    ib2=v1
    ib3=v4
   if (_self.l3<=_self.l1) and (_self.l3<=_self.l2) and (_self.l3<=_self.l4): 
    vjump=2 
    ib1=v1
    ib2=v2
    ib3=v4
   if (_self.l4<=_self.l1) and (_self.l4<=_self.l2) and (_self.l4<=_self.l3): 
    vjump=3 
    ib1=v2
    ib2=v1
    ib3=v3

   if _self.oface[_destElem,vjump] != -1:
    _self.jumpToElem(_self.oface[_destElem,vjump],_iiVert,_R2X,_R2Y,_R2Z)
   else:
    Bl1,Bl2,Bl3 = _self.computeIntercept(_iiVert,_R2X,_R2Y,_R2Z,ib1,ib2,ib3)
    _self.conv[_iiVert,ib1] = Bl1
    _self.conv[_iiVert,ib2] = Bl2
    _self.conv[_iiVert,ib3] = Bl3

 # R1X = X[i], R1Y = Y[i], R1Z = Z[i] (arrival point)
 # _R2X = xp, _R2Y = yp, _R2Z = zp (departure point)
 # B1X, B2Y, B3Z etc. (boundary points - triangle)
 def computeIntercept(_self,_i,_R2X,_R2Y,_R2Z,_ib1,_ib2,_ib3):
  R1X = _self.X[_i]
  R1Y = _self.Y[_i]
  R1Z = _self.Z[_i]

  B1X = _self.X[_ib1]
  B2X = _self.X[_ib2]
  B3X = _self.X[_ib3]
  B1Y = _self.Y[_ib1]
  B2Y = _self.Y[_ib2]
  B3Y = _self.Y[_ib3]
  B1Z = _self.Z[_ib1]
  B2Z = _self.Z[_ib2]
  B3Z = _self.Z[_ib3]

  a1 = B1X-B3X; b1 = B2X-B3X; c1 = R1X-_R2X; d1 = R1X-B3X;
  a2 = B1Y-B3Y; b2 = B2Y-B3Y; c2 = R1Y-_R2Y; d2 = R1Y-B3Y; 
  a3 = B1Z-B3Z; b3 = B2Z-B3Z; c3 = R1Z-_R2Z; d3 = R1Z-B3Z;

  det  = (a1*b2*c3)+(a3*b1*c2)+(a2*b3*c1)-(a3*b2*c1)-(a1*b3*c2)-(a2*b1*c3)
  detx = (d1*b2*c3)+(d3*b1*c2)+(d2*b3*c1)-(d3*b2*c1)-(d1*b3*c2)-(d2*b1*c3)
  dety = (a1*d2*c3)+(a3*d1*c2)+(a2*d3*c1)-(a3*d2*c1)-(a1*d3*c2)-(a2*d1*c3)

  x1 = detx/det
  x2 = dety/det

  Bl1 = x1
  Bl2 = x2
  Bl3 = 1.0-Bl1-Bl2

  # return C++ equivalent Bl1, Bl2 and Bl3 (see femSIM3d)
  return Bl1,Bl2,Bl3

 def testElement(_self,_mele,_xp,_yp,_zp):
  v1 = _self.IEN[_mele][0]
  v2 = _self.IEN[_mele][1]
  v3 = _self.IEN[_mele][2]
  v4 = _self.IEN[_mele][3]
  EPSlocal = 1e-04

  V = (1.0/6.0) * (+1*( (_self.X[v2]*_self.Y[v3]*_self.Z[v4])
                       +(_self.Y[v2]*_self.Z[v3]*_self.X[v4])
                       +(_self.Z[v2]*_self.X[v3]*_self.Y[v4])
                       -(_self.Y[v2]*_self.X[v3]*_self.Z[v4])
                       -(_self.X[v2]*_self.Z[v3]*_self.Y[v4])
                       -(_self.Z[v2]*_self.Y[v3]*_self.X[v4]) )
         -_self.X[v1]*( +_self.Y[v3]*_self.Z[v4]
                        +_self.Y[v2]*_self.Z[v3]
                        +_self.Z[v2]*_self.Y[v4]
                        -_self.Y[v2]*_self.Z[v4]
                        -_self.Z[v3]*_self.Y[v4]
                        -_self.Z[v2]*_self.Y[v3] )
         +_self.Y[v1]*( +_self.X[v3]*_self.Z[v4]
                        +_self.X[v2]*_self.Z[v3]
                        +_self.Z[v2]*_self.X[v4]
                        -_self.X[v2]*_self.Z[v4]
                        -_self.Z[v3]*_self.X[v4]
                        -_self.Z[v2]*_self.X[v3] )
         -_self.Z[v1]*( +_self.X[v3]*_self.Y[v4]
                        +_self.X[v2]*_self.Y[v3]
                        +_self.Y[v2]*_self.X[v4]
                        -_self.X[v2]*_self.Y[v4]
                        -_self.Y[v3]*_self.X[v4]
                        -_self.Y[v2]*_self.X[v3] ) )

  V1 = (1.0/6.0) * (+1*( (_self.X[v2]*_self.Y[v3]*_self.Z[v4])
                        +(_self.Y[v2]*_self.Z[v3]*_self.X[v4])
                        +(_self.Z[v2]*_self.X[v3]*_self.Y[v4])
                        -(_self.Y[v2]*_self.X[v3]*_self.Z[v4])
                        -(_self.X[v2]*_self.Z[v3]*_self.Y[v4])
                        -(_self.Z[v2]*_self.Y[v3]*_self.X[v4]) )
                  -_xp*( +_self.Y[v3]*_self.Z[v4]
                         +_self.Y[v2]*_self.Z[v3]
                         +_self.Z[v2]*_self.Y[v4]
                         -_self.Y[v2]*_self.Z[v4]
                         -_self.Z[v3]*_self.Y[v4]
                         -_self.Z[v2]*_self.Y[v3] )
                  +_yp*( +_self.X[v3]*_self.Z[v4]
                         +_self.X[v2]*_self.Z[v3]
                         +_self.Z[v2]*_self.X[v4]
                         -_self.X[v2]*_self.Z[v4]
                         -_self.Z[v3]*_self.X[v4]
                         -_self.Z[v2]*_self.X[v3] )
                  -_zp*( +_self.X[v3]*_self.Y[v4]
                         +_self.X[v2]*_self.Y[v3]
                         +_self.Y[v2]*_self.X[v4]
                         -_self.X[v2]*_self.Y[v4]
                         -_self.Y[v3]*_self.X[v4]
                         -_self.Y[v2]*_self.X[v3] ) )

  V2 = (1.0/6.0) * (+1*( (_xp*_self.Y[v3]*_self.Z[v4])
                        +(_yp*_self.Z[v3]*_self.X[v4])
                        +(_zp*_self.X[v3]*_self.Y[v4])
                        -(_yp*_self.X[v3]*_self.Z[v4])
                        -(_xp*_self.Z[v3]*_self.Y[v4])
                        -(_zp*_self.Y[v3]*_self.X[v4]) )
             -_self.X[v1]*( +_self.Y[v3]*_self.Z[v4]
                         +_yp*_self.Z[v3]
                         +_zp*_self.Y[v4]
                         -_yp*_self.Z[v4]
                            -_self.Z[v3]*_self.Y[v4]
                         -_zp*_self.Y[v3] )
             +_self.Y[v1]*( +_self.X[v3]*_self.Z[v4]
                         +_xp*_self.Z[v3]
                         +_zp*_self.X[v4]
                         -_xp*_self.Z[v4]
                            -_self.Z[v3]*_self.X[v4]
                         -_zp*_self.X[v3] )
             -_self.Z[v1]*( +_self.X[v3]*_self.Y[v4]
                         +_xp*_self.Y[v3]
                         +_yp*_self.X[v4]
                         -_xp*_self.Y[v4]
                         -_self.Y[v3]*_self.X[v4]
                         -_yp*_self.X[v3] ) )

  V3 = (1.0/6.0) * (+1*( (_self.X[v2]*_yp*_self.Z[v4])
                        +(_self.Y[v2]*_zp*_self.X[v4])
                        +(_self.Z[v2]*_xp*_self.Y[v4])
                        -(_self.Y[v2]*_xp*_self.Z[v4])
                        -(_self.X[v2]*_zp*_self.Y[v4])
                        -(_self.Z[v2]*_yp*_self.X[v4]) )
       -_self.X[v1]*( +_yp*_self.Z[v4]
                         +_self.Y[v2]*_zp
                         +_self.Z[v2]*_self.Y[v4]
                         -_self.Y[v2]*_self.Z[v4]
                         -_zp*_self.Y[v4]
                         -_self.Z[v2]*_yp )
       +_self.Y[v1]*( +_xp*_self.Z[v4]
                         +_self.X[v2]*_zp
                         +_self.Z[v2]*_self.X[v4]
                         -_self.X[v2]*_self.Z[v4]
                         -_zp*_self.X[v4]
                         -_self.Z[v2]*_xp )
       -_self.Z[v1]*( +_xp*_self.Y[v4]
                         +_self.X[v2]*_yp
                         +_self.Y[v2]*_self.X[v4]
                         -_self.X[v2]*_self.Y[v4]
                         -_yp*_self.X[v4]
                         -_self.Y[v2]*_xp ) );

  _self.l1 = V1/V
  _self.l2 = V2/V
  _self.l3 = V3/V
  _self.l4 = 1.0 - _self.l3 - _self.l2 - _self.l1

  return ( (_self.l1>=0.0-EPSlocal) and (_self.l1<=1.0+EPSlocal) and \
           (_self.l2>=0.0-EPSlocal) and (_self.l2<=1.0+EPSlocal) and \
           (_self.l3>=0.0-EPSlocal) and (_self.l3<=1.0+EPSlocal) and \
           (_self.l4>=0.0-EPSlocal) and (_self.l4<=1.0+EPSlocal) )

 def setCentroid(_self,_vxd,_vyd,_vzd):
  # interpolando no centroid
  zeros = np.zeros( (_self.numNodes-_self.numVerts),dtype='float')
  _vxd = np.append(_vxd,zeros)
  _vyd = np.append(_vyd,zeros)
  _vzd = np.append(_vzd,zeros)
  for e in range(0,_self.numElems):
   [v1,v2,v3,v4,v5] = _self.IEN[e]
   _vxd[v5] = (_vxd[v1] + _vxd[v2] + _vxd[v3] + _vxd[v4])/4.0
   _vyd[v5] = (_vyd[v1] + _vyd[v2] + _vyd[v3] + _vyd[v4])/4.0
   _vzd[v5] = (_vzd[v1] + _vzd[v2] + _vzd[v3] + _vzd[v4])/4.0
  return [_vxd,_vyd,_vzd]

 def setBC(_self,_idv,_vbc,_vd):
  # impondo c.c. de velocidade
  # condicao de contorno (Dirichlet) para vxd,vyd 
  for i in _idv:
   _vd[i] = _vbc[i]
  return _vd

