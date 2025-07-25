import numpy as np
# from numba import int32, float32    
# from numba.experimental import jitclass

class Linear:
 def __init__(_self,_X,_Y):

  _self.NUMRULE = 4 # pontos de Gauss
  _self.NUMGLEC = 3 # nos no triangulo

  _self.X = _X
  _self.Y = _Y
  _self.mass = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.area = 0.0

  # L1, L2 e L3 de todos os pontos de Gauss
  _self.gqPoints = [ [0.33333333333333, 0.33333333333333, 0.33333333333333], 
                     [0.60000000000000, 0.20000000000000, 0.20000000000000], 
                     [0.20000000000000, 0.60000000000000, 0.20000000000000], 
                     [0.20000000000000, 0.20000000000000, 0.60000000000000] ]

  _self.gqWeights = [ -0.562500000000000, 
                       0.520833333333333, 
                       0.520833333333333, 
                       0.520833333333333 ]

  # N1, N2 e N3
  _self.phiJ = [ [0.33333333333333, 0.33333333333333, 0.33333333333333], 
                 [0.60000000000000, 0.20000000000000, 0.20000000000000], 
                 [0.20000000000000, 0.60000000000000, 0.20000000000000], 
                 [0.20000000000000, 0.20000000000000, 0.60000000000000] ]

  # dN1/dL1, dN2/dL1 e dN3/dL1
  _self.dphiJdl1 = [ [1.0, 0.0,-1.0], 
                     [1.0, 0.0,-1.0], 
                     [1.0, 0.0,-1.0], 
                     [1.0, 0.0,-1.0] ]

  # dN1/dL2, dN2/dL2 e dN3/dL2
  _self.dphiJdl2 = [ [0.0, 1.0,-1.0], 
                     [0.0, 1.0,-1.0], 
                     [0.0, 1.0,-1.0], 
                     [0.0, 1.0,-1.0] ]
 
 def getM(_self,v):
  # calcula 2*area do triangulo da malha
  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  # Mudanca de variavel: pontos de Gauss
  localx = np.zeros((_self.NUMRULE), dtype=np.float)
  localy = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE):
   valx = 0.0;
   valy = 0.0;
   for i in range(0,_self.NUMGLEC):
    valx += _self.X[v[i]] * _self.phiJ[k][i]
    valy += _self.Y[v[i]] * _self.phiJ[k][i]
   localx[k] = valx;
   localy[k] = valy;

  # Mudanca de variavel: derivadas pontos de Gauss
  dxdl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dxdl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  jacob = np.zeros((_self.NUMRULE),dtype='float')
  for k in range(0,_self.NUMRULE): 
   valxl1 = 0.0;
   valxl2 = 0.0;
   valyl1 = 0.0;
   valyl2 = 0.0;
   for i in range(0,_self.NUMGLEC): 
    valxl1 += _self.X[v[i]]*_self.dphiJdl1[k][i]
    valxl2 += _self.X[v[i]]*_self.dphiJdl2[k][i]
    valyl1 += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    valyl2 += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   dxdl1[k] = valxl1
   dxdl2[k] = valxl2
   dydl1[k] = valyl1
   dydl2[k] = valyl2
   # compute det(jacobian)
   jacob[k] = abs(valxl1*valyl2 - valxl2*valyl1)

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEC), dtype=np.float)
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEC), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEC):
    dphiJdx[k][i] = ( _self.dphiJdl1[k][i]*dydl2[k]-_self.dphiJdl2[k][i]\
                  *dydl1[k] )/jacobian;
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+_self.dphiJdl2[k][i]\
                  *dxdl1[k] )/jacobian;

  _self.mass = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEC):
    for j in range(0,_self.NUMGLEC):
     _self.mass[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]\
                         *jacob[k]*_self.gqWeights[k];
     _self.kxx[i][j]  += dphiJdx[k][i]*dphiJdx[k][j]\
                         *jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j]  += dphiJdy[k][i]*dphiJdy[k][j]\
                         *jacob[k]*_self.gqWeights[k];
     _self.gx[i][j]   += _self.gqPoints[k][j]*dphiJdx[k][i]\
                         *jacob[k]*_self.gqWeights[k];
     _self.gy[i][j]   += _self.gqPoints[k][j]*dphiJdy[k][i]\
                         *jacob[k]*_self.gqWeights[k];
     _self.dx[j][i]   += _self.gqPoints[k][j]*dphiJdx[k][i]\
                         *jacob[k]*_self.gqWeights[k];
     _self.dy[j][i]   += _self.gqPoints[k][j]*dphiJdy[k][i]\
                         *jacob[k]*_self.gqWeights[k];

   # compute area 
   _self.area += (1.0/2.0)*jacob[k]*_self.gqWeights[k]


 def getK(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = np.zeros((_self.NUMRULE), dtype=np.float)
  localy = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEC):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dxdl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEC): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEC), dtype=np.float)
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEC), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEC):
    dphiJdx[k][i] = ( _self.dphiJdl1[k][i]*dydl2[k]-_self.dphiJdl2[k][i]\
                  *dydl1[k] )/jacobian;
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+_self.dphiJdl2[k][i]\
                  *dxdl1[k] )/jacobian;

  _self.mass = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEC,_self.NUMGLEC), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEC):
    for j in range(0,_self.NUMGLEC):
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]\
                     *jacobian*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]\
                     *jacobian*_self.gqWeights[k];


 def getKAnalytic(_self,v):
  A=0.5*( ((_self.X[v[1]]-_self.X[v[0]])*(_self.Y[v[2]]-_self.Y[v[0]]))\
         -((_self.X[v[2]]-_self.X[v[0]])*(_self.Y[v[1]]-_self.Y[v[0]])) )
 
  _self.mass = ([ [A*(1.0/6.0),  A*(1.0/12.0), A*(1.0/12.0)],
                  [A*(1.0/12.0), A*(1.0/6.0),  A*(1.0/12.0)],
                  [A*(1.0/12.0), A*(1.0/12.0), A*(1.0/6.0)] ])
 
  b1 = _self.Y[v[1]]-_self.Y[v[2]]
  b2 = _self.Y[v[2]]-_self.Y[v[0]]
  b3 = _self.Y[v[0]]-_self.Y[v[1]]
 
  c1 = _self.X[v[2]]-_self.X[v[1]]
  c2 = _self.X[v[0]]-_self.X[v[2]]
  c3 = _self.X[v[1]]-_self.X[v[0]]
 
  B = np.array([ [(1.0/2*A)*b1, (1.0/2*A)*b2, (1.0/2*A)*b3],
                 [(1.0/2*A)*c1, (1.0/2*A)*c2, (1.0/2*A)*c3] ])

  BT = np.array([ [(1.0/2*A)*b1, (1.0/2*A)*c1],
                  [(1.0/2*A)*b2, (1.0/2*A)*c2],
                  [(1.0/2*A)*b3, (1.0/2*A)*c3] ])

  for i in range(0,_self.NUMGLEC):
   for j in range(0,_self.NUMGLEC-1):
    _self.k[i][j] += BT[i][j]*B[j][i]

 def getMAnalytic(_self,v):
  A=0.5*( ((_self.X[v[1]]-_self.X[v[0]])*(_self.Y[v[2]]-_self.Y[v[0]]))\
         -((_self.X[v[2]]-_self.X[v[0]])*(_self.Y[v[1]]-_self.Y[v[0]])) )
 
  _self.mass = (A/12.0)*np.array([ [2.0, 1.0, 1.0],
                                   [1.0, 2.0, 1.0],
                                   [1.0, 1.0, 2.0] ])
 
  b1 = _self.Y[v[1]]-_self.Y[v[2]]
  b2 = _self.Y[v[2]]-_self.Y[v[0]]
  b3 = _self.Y[v[0]]-_self.Y[v[1]]
 
  c1 = _self.X[v[2]]-_self.X[v[1]]
  c2 = _self.X[v[0]]-_self.X[v[2]]
  c3 = _self.X[v[1]]-_self.X[v[0]]
 
  B = (1.0/(2*A))*np.array([ [b1, b2, b3],
                             [c1, c2, c3] ])

  BT = (1.0/(2*A))*np.array([ [b1, c1],
                              [b2, c2],
                              [b3, c3] ])

  for i in range(0,_self.NUMGLEC):
   for j in range(0,_self.NUMGLEC-1):
    _self.k[i][j] += BT[i][j]*B[j][i]

  _self.kxx = (1.0/(4*A))*np.array( [[b1*b1,b1*b2,b1*b3],
                                     [b2*b1,b2*b2,b2*b3],
                                     [b3*b1,b3*b2,b3*b3]])
  _self.kyy = (1.0/(4*A))*np.array( [[c1*c1,c1*c2,c1*c3],
                                     [c2*c1,c2*c2,c2*c3],
                                     [c3*c1,c3*c2,c3*c3]])
  _self.kxy = (1.0/(4*A))*np.array( [[b1*c1,b1*c2,b1*c3],
                                     [b2*c1,b2*c2,b2*c3],
                                     [b3*c1,b3*c2,b3*c3]])

  _self.gx = (1.0/6.0)*np.array( [[b1,b2,b3],
                                  [b1,b2,b3],
                                  [b1,b2,b3]])
  _self.gy = (1.0/6.0)*np.array( [[c1,c2,c3],
                                  [c1,c2,c3],
                                  [c1,c2,c3]])


class Mini:
 def __init__(_self,_X,_Y):

  _self.NUMRULE = 12
  _self.NUMGLEU = 4 
  _self.NUMGLEP = 3 

  _self.X = _X
  _self.Y = _Y
  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.kxxp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.kxyp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.kyxp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.kyyp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)

  _self.gqPoints=[ [0.873821971016996, 0.063089014491502, 0.063089014491502],
                   [0.063089014491502, 0.873821971016996, 0.063089014491502],
                   [0.063089014491502, 0.063089014491502, 0.873821971016996],
                   [0.501426509658179, 0.249286745170910, 0.249286745170911],
                   [0.249286745170910, 0.501426509658179, 0.249286745170911],
                   [0.249286745170910, 0.249286745170910, 0.50142650965818 ],
                   [0.636502499121399, 0.310352451033785, 0.053145049844816],
                   [0.636502499121399, 0.053145049844816, 0.310352451033785],
                   [0.310352451033785, 0.636502499121399, 0.053145049844816],
                   [0.310352451033785, 0.053145049844816, 0.636502499121399],
                   [0.053145049844816, 0.636502499121399, 0.310352451033785],
                   [0.053145049844816, 0.310352451033785, 0.636502499121399] ]

  _self.gqWeights = [ 0.050844906370207,
                      0.050844906370207,
                      0.050844906370207,
                      0.116786275726379,
                      0.116786275726379,
                      0.116786275726379,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374 ]

  _self.phiJ = [ 
[ 0.842519908360035, 0.031786951834541, 0.031786951834541, 0.093906187970883],
[ 0.031786951834541, 0.842519908360035, 0.031786951834541, 0.093906187970883],
[ 0.031786951834541, 0.031786951834541, 0.842519908360035, 0.093906187970883],
[ 0.220981204105529,-0.03115856038174 ,-0.031158560381739, 0.841335916657949],
[-0.03115856038174 , 0.220981204105529,-0.031158560381739, 0.841335916657949],
[-0.031158560381739,-0.031158560381739, 0.220981204105531, 0.841335916657947],
[ 0.542017987859968, 0.215867939772354,-0.041339461416615, 0.283453533784293],
[ 0.542017987859968,-0.041339461416615, 0.215867939772354, 0.283453533784293],
[ 0.215867939772354, 0.542017987859968,-0.041339461416615, 0.283453533784293],
[ 0.215867939772354,-0.041339461416615, 0.542017987859968, 0.283453533784293],
[-0.041339461416615, 0.542017987859968, 0.215867939772354, 0.283453533784293],
[-0.041339461416615, 0.215867939772354, 0.542017987859968, 0.283453533784293] ]

  _self.dphiJdl1 = [ 
[ 1.460335089186776,  4.603350891867764e-01,-0.539664910813224,
  -1.381005267560329e+00],
[ 1.               ,  3.885780586188048e-16,-1.               ,
  -1.332267629550188e-15],
[ 0.539664910813224, -4.603350891867763e-01,-1.460335089186776,
  1.381005267560329e+00],
[ 1.565695910954718,  5.656959109547176e-01,-0.434304089045282,
  -1.697087732864153e+00],
[ 0.999999999999996, -4.440892098500626e-15,-1.000000000000004,
  1.332267629550188e-14],
[ 0.434304089045278, -5.656959109547223e-01,-1.565695910954722,
  1.697087732864166e+00],
[ 2.62941772790624 ,  1.629417727906240e+00, 0.62941772790624 ,
  -4.888253183718719e+00],
[ 1.155999345062549,  1.559993450625485e-01,-0.844000654937452,
  -4.679980351876453e-01],
[ 2.473418382843691,  1.473418382843691e+00, 0.473418382843691,
  -4.420255148531075e+00],
[ 0.844000654937452, -1.559993450625484e-01,-1.155999345062548,
  4.679980351876453e-01],
[-0.473418382843691, -1.473418382843691e+00,-2.473418382843691,
  4.420255148531074e+00],
[-0.62941772790624 , -1.629417727906240e+00,-2.62941772790624 ,
  4.888253183718721e+00] ]

  _self.dphiJdl2 = [
[ 4.440892098500626e-16, 1.               ,-1.               ,
 -1.332267629550188e-15],
[ 4.603350891867763e-01, 1.460335089186776,-0.539664910813224,
 -1.381005267560329e+00],
[-4.603350891867763e-01, 0.539664910813224,-1.460335089186776,
 1.381005267560329e+00],
[-4.440892098500626e-15, 0.999999999999996,-1.000000000000004,
 1.332267629550188e-14],
[ 5.656959109547176e-01, 1.565695910954718,-0.434304089045282,
 -1.697087732864153e+00],
[-5.656959109547223e-01, 0.434304089045278,-1.565695910954722,
 1.697087732864166e+00],
[ 1.473418382843691e+00, 2.473418382843691, 0.473418382843691,
 -4.420255148531074e+00],
[-1.473418382843691e+00,-0.473418382843691,-2.473418382843691,
 4.420255148531074e+00],
[ 1.629417727906240e+00, 2.62941772790624 , 0.62941772790624 ,
 -4.888253183718720e+00],
[-1.629417727906240e+00,-0.62941772790624 ,-2.62941772790624 ,
 4.888253183718721e+00],
[ 1.559993450625485e-01, 1.155999345062549,-0.844000654937452,
 -4.679980351876454e-01],
[-1.559993450625484e-01, 0.844000654937452,-1.155999345062548,
 4.679980351876453e-01] ]

  _self.dgqPointsdl1 = [ [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1] ]

  _self.dgqPointsdl2 = [ [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1] ]

 def getM(_self,v):

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  jacob = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   # compute det(jacobian)
   jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];

  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*\
                         jacob[k]*_self.gqWeights[k];
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.gvx[i][j] += _self.phiJ[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.gvy[i][j] += _self.phiJ[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];

    for j in range(0,_self.NUMGLEP):
     _self.gx[i][j] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.gy[i][j] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dx[j][i] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dy[j][i] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];


 def getMass_porous(_self,v,porous_nodes):

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   # compute det(jacobian)
   jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];

  _self.mass_porous = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.GQ_porous_values = _self.phiJ@porous_nodes[v]
  
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass_porous[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*_self.GQ_porous_values[k]*\
                         jacob[k]*_self.gqWeights[k];
                         
 def getMass_Forchheimer(_self,v,porous_nodes,vx,vy):

    localx = [0.0]*_self.NUMRULE
    localy = [0.0]*_self.NUMRULE
    for k in range(0,_self.NUMRULE):
     for i in range(0,_self.NUMGLEU):
      localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
      localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]
    
    dxdl1 = [0.0]*_self.NUMRULE
    dxdl2 = [0.0]*_self.NUMRULE
    dydl1 = [0.0]*_self.NUMRULE
    dydl2 = [0.0]*_self.NUMRULE
    jacob = [0.0]*_self.NUMRULE # area for each Gauss point
    for k in range(0,_self.NUMRULE): 
     for i in range(0,_self.NUMGLEU): 
      dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
      dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
      dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
      dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
     # compute det(jacobian)
     jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]
    
    dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
    dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
    for k in range(0,_self.NUMRULE): 
     for i in range(0,_self.NUMGLEU):
      dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
     	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
      dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
     	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];
    
    _self.mass_forchheimer = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
    _self.GQ_porous_values = _self.phiJ@porous_nodes[v]
    # _self.GQ_velocity_mod = _self.phiJ@vx[v]
    _self.GQ_velocity_mod = np.zeros((len(vx)),dtype='float')
    
    for k in range(0,_self.NUMRULE): 
     _self.GQ_velocity_mod[k] = np.sqrt(((_self.phiJ@vx[v])[k])**2 + ((_self.phiJ@vy[v])[k])**2)
     for i in range(0,_self.NUMGLEU):
      for j in range(0,_self.NUMGLEU):
       _self.mass_forchheimer[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*_self.GQ_porous_values[k]*_self.GQ_velocity_mod[k]*\
                           jacob[k]*_self.gqWeights[k];
                         
 def getMSlip(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]

  # gqPoints
  localxlin = [0.0]*_self.NUMRULE
  localylin = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    localxlin[k] += _self.X[v[i]] * _self.gqPoints[k][i]
    localylin[k] += _self.Y[v[i]] * _self.gqPoints[k][i]

  dxdl1lin = [0.0]*_self.NUMRULE
  dxdl2lin = [0.0]*_self.NUMRULE
  dydl1lin = [0.0]*_self.NUMRULE
  dydl2lin = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEP): 
    dxdl1lin[k] += _self.X[v[i]]*_self.dgqPointsdl1[k][i]
    dxdl2lin[k] += _self.X[v[i]]*_self.dgqPointsdl2[k][i]
    dydl1lin[k] += _self.Y[v[i]]*_self.dgqPointsdl1[k][i]
    dydl2lin[k] += _self.Y[v[i]]*_self.dgqPointsdl2[k][i]

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dgqPointsdx = [[0.0]*_self.NUMGLEP for i in range(_self.NUMRULE)]
  dgqPointsdy = [[0.0]*_self.NUMGLEP for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacobian;
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacobian;
   for i in range(0,_self.NUMGLEP):
    dgqPointsdx[k][i] = (  _self.dgqPointsdl1[k][i]*dydl2lin[k]-
	                   _self.dgqPointsdl2[k][i]*dydl1lin[k] )/jacobian;
    dgqPointsdy[k][i] = ( -_self.dgqPointsdl1[k][i]*dxdl2lin[k]+
	                   _self.dgqPointsdl2[k][i]*dxdl1lin[k] )/jacobian;

  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.kxxp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.kxyp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.kyxp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.kyyp = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*\
                         jacobian*_self.gqWeights[k];
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]*\
                        jacobian*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]*\
                        jacobian*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]*\
                        jacobian*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]*\
                        jacobian*_self.gqWeights[k];
     _self.gvx[i][j] += _self.phiJ[k][i]*dphiJdx[k][j]*\
                        jacobian*_self.gqWeights[k];
     _self.gvy[i][j] += _self.phiJ[k][i]*dphiJdy[k][j]*\
                        jacobian*_self.gqWeights[k];

    for j in range(0,_self.NUMGLEP):
     _self.gx[i][j] += _self.dgqPointsdx[k][j]*_self.phiJ[k][i]*\
                       jacobian*_self.gqWeights[k];
     _self.gy[i][j] += _self.dgqPointsdy[k][j]*_self.phiJ[k][i]*\
                       jacobian*_self.gqWeights[k];
     _self.dx[j][i] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacobian*_self.gqWeights[k];
     _self.dy[j][i] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacobian*_self.gqWeights[k];
     _self.kxxp[i][j] += dphiJdx[k][i]*dgqPointsdx[k][j]*\
                         jacobian*_self.gqWeights[k];
     _self.kxyp[i][j] += dphiJdx[k][i]*dgqPointsdy[k][j]*\
                         jacobian*_self.gqWeights[k];
     _self.kyxp[i][j] += dphiJdy[k][i]*dgqPointsdx[k][j]*\
                         jacobian*_self.gqWeights[k];
     _self.kyyp[i][j] += dphiJdy[k][i]*dgqPointsdy[k][j]*\
                         jacobian*_self.gqWeights[k];

 def getK(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = np.zeros((_self.NUMRULE), dtype=np.float)
  localy = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE):
   valx = 0.0;
   valy = 0.0;
   for i in range(0,_self.NUMGLEU):
    valx += _self.X[v[i]] * _self.phiJ[k][i]
    valy += _self.Y[v[i]] * _self.phiJ[k][i]
   localx[k] = valx;
   localy[k] = valy;

  dxdl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dxdl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   valxl1 = 0.0;
   valxl2 = 0.0;
   valyl1 = 0.0;
   valyl2 = 0.0;
   for i in range(0,_self.NUMGLEU): 
    valxl1 += _self.X[v[i]]*_self.dphiJdl1[k][i]
    valxl2 += _self.X[v[i]]*_self.dphiJdl2[k][i]
    valyl1 += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    valyl2 += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   dxdl1[k] = valxl1
   dxdl2[k] = valxl2
   dydl1[k] = valyl1
   dydl2[k] = valyl2

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = ( _self.dphiJdl1[k][i]*dydl2[k]-_self.dphiJdl2[k][i]\
                  *dydl1[k] )/jacobian;
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+_self.dphiJdl2[k][i]\
                  *dxdl1[k] )/jacobian;

  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]\
                     *jacobian*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]\
                     *jacobian*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]\
                     *jacobian*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]\
                     *jacobian*_self.gqWeights[k];

 def getMAnalytic(_self,v):
  A=0.5*( ((_self.X[v[1]]-_self.X[v[0]])*(_self.Y[v[2]]-_self.Y[v[0]]))\
         -((_self.X[v[2]]-_self.X[v[0]])*(_self.Y[v[1]]-_self.Y[v[0]])) )
 
  _self.mass = (A/12.0)*np.array([ [2.0, 1.0, 1.0],
                                   [1.0, 2.0, 1.0],
                                   [1.0, 1.0, 2.0] ])
 
  b1 = _self.Y[v[1]]-_self.Y[v[2]]
  b2 = _self.Y[v[2]]-_self.Y[v[0]]
  b3 = _self.Y[v[0]]-_self.Y[v[1]]
 
  c1 = _self.X[v[2]]-_self.X[v[1]]
  c2 = _self.X[v[0]]-_self.X[v[2]]
  c3 = _self.X[v[1]]-_self.X[v[0]]
 
  B = (1.0/(2*A))*np.array([ [b1, b2, b3],
                             [c1, c2, c3] ])

  BT = (1.0/(2*A))*np.array([ [b1, c1],
                              [b2, c2],
                              [b3, c3] ])

  klin = np.zeros((_self.NUMGLEP,_self.NUMGLEP), dtype=np.float)
  for i in range(0,_self.NUMGLEP):
   for j in range(0,_self.NUMGLEP-1):
    klin[i][j] += BT[i][j]*B[j][i]

  kxlin = (1.0/(4*A))*np.array( [[b1*b1,b1*b2,b1*b3],
                                 [b2*b1,b2*b2,b2*b3],
                                 [b3*b1,b3*b2,b3*b3]])
  kylin = (1.0/(4*A))*np.array( [[c1*c1,c1*c2,c1*c3],
                                 [c2*c1,c2*c2,c2*c3],
                                 [c3*c1,c3*c2,c3*c3]])
  kxylin = (1.0/(4*A))*np.array( [[b1*c1,b1*c2,b1*c3],
                                  [b2*c1,b2*c2,b2*c3],
                                  [b3*c1,b3*c2,b3*c3]])

  gxlin = (1.0/6.0)*np.array( [[b1,b2,b3],
                               [b1,b2,b3],
                               [b1,b2,b3]])
  gylin = (1.0/6.0)*np.array( [[c1,c2,c3],
                               [c1,c2,c3],
                               [c1,c2,c3]])

  zz   = (1.0/(4.0*A)) * (b2*b2 + b3*b3 + b2*b3 + c2*c2 + c3*c3 + c2*c3)
  zzx  = (1.0/(4.0*A)) * (b2*b2 + b3*b3 + b2*b3)
  zzy  = (1.0/(4.0*A)) * (c2*c2 + c3*c3 + c2*c3)
  zzxy = (1.0/(4.0*A)) * (b2*c2 + b3*c3 + 0.5*b2*c3 + 0.5*b3*c2)

  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.mass = (A/840.0)*np.array( [[83,13,13, 45],
                                    [13,83,13, 45],
                                    [13,13,83, 45],
                                    [45,45,45,243]] )

  
  # k = kxx + kyy mini
  _self.k = np.zeros( (4,4),dtype='float' )
  _self.k[0:3,0:3]  = A*klin + (9/10)*zz
  _self.k[0:3,3] = -(27/10)*zz
  _self.k[3,0:3] = -(27/10)*zz
  _self.k[3,3]   = (81/10)*zz

  # kxx mini
  _self.kxx = np.zeros( (4,4),dtype='float' )
  _self.kxx[0:3,0:3]  = kxlin + (9/10)*zzx
  _self.kxx[0:3,3] = -(27/10)*zzx
  _self.kxx[3,0:3] = -(27/10)*zzx
  _self.kxx[3,3]   = (81/10)*zzx

  # kyy mini
  _self.kyy = np.zeros( (4,4),dtype='float' )
  _self.kyy[0:3,0:3]  = kylin + (9/10)*zzy
  _self.kyy[0:3,3] = -(27/10)*zzy
  _self.kyy[3,0:3] = -(27/10)*zzy
  _self.kyy[3,3]   = (81/10)*zzy

  # kxy mini
  _self.kxy = np.zeros( (4,4),dtype='float' )
  _self.kxy[0:3,0:3]  = kxylin + (9/10)*zzxy
  _self.kxy[0:3,3] = -(27/10)*zzxy
  _self.kxy[3,0:3] = -(27/10)*zzxy
  _self.kxy[3,3]   = (81/10)*zzxy

  # kyx mini
  _self.kyx = np.zeros( (4,4),dtype='float' )
  _self.kyx = _self.kxy.transpose()

  # gx mini
  _self.gx = np.zeros( (4,3),dtype='float' )
  _self.gx[0:3,0:3] = (9/20)*gxlin + gxlin.transpose()
  _self.gx[3,:] = [-(9/40)*b1,-(9/40)*b2,-(9/40)*b3]

  # gy mini
  _self.gy = np.zeros( (4,3),dtype='float' )
  _self.gy[0:3,0:3] = (9/20)*gylin + gylin.transpose()
  _self.gy[3,:] = [-(9/40)*c1,-(9/40)*c2,-(9/40)*c3]

  # dx,dy mini (dx = gx mini^T, dy = gy mini^T)
  _self.dx = _self.gx.transpose()
  _self.dy = _self.gy.transpose()

  # uncoupled (projection method):
  # - gx dx=gx^T (okay)
  # - gx slip dx=gx (okay)
  # - gx slip dx slip (no)
  #
  # coupled:
  # - gx dx=gx^T (okay)
  # - gx slip dx=gx (no)
  # - gx slip dx slip (no)

  ## gx slip mini (must be done after dx, since dx is NOT gx slip mini^T)
  #_self.gx = np.zeros( (4,3),dtype='float' )
  #_self.gx[0:3,0] = (11.0/120.0)*b1
  #_self.gx[0:3,1] = (11.0/120.0)*b2
  #_self.gx[0:3,2] = (11.0/120.0)*b3
  #_self.gx[3,:] = [(9/40)*b1,(9/40)*b2,(9/40)*b3]

  ## gy slip mini (must be done after dy, since dy is NOT gy slip mini^T)
  #_self.gy = np.zeros( (4,3),dtype='float' )
  #_self.gy[0:3,0] = (11.0/120.0)*c1
  #_self.gy[0:3,1] = (11.0/120.0)*c2
  #_self.gy[0:3,2] = (11.0/120.0)*c3
  #_self.gy[3,:] = [(9/40)*c1,(9/40)*c2,(9/40)*c3]

  # gvx mini
  _self.gvx = np.zeros( (4,4),dtype='float' )
  _self.gvx[0:3,0:3] = (11/20)*gxlin + (9/20)*gxlin.transpose()
  _self.gvx[0,3]     = -(9/40)*b1
  _self.gvx[1,3]     = -(9/40)*b2
  _self.gvx[2,3]     = -(9/40)*b3
  _self.gvx[3,0:3]   = [(9/40)*b1,(9/40)*b2,(9/40)*b3]
  _self.gvx[3,3]     = 0.0

  # gvy mini
  _self.gvy = np.zeros( (4,4),dtype='float' )
  _self.gvy[0:3,0:3] = (11/20)*gylin + (9/20)*gylin.transpose()
  _self.gvy[0,3]     = -(9/40)*c1
  _self.gvy[1,3]     = -(9/40)*c2
  _self.gvy[2,3]     = -(9/40)*c3
  _self.gvy[3,0:3]   = [(9/40)*c1,(9/40)*c2,(9/40)*c3]
  _self.gvy[3,3]     = 0.0
  
# spec = [
#     ('X', float32[:]),               # a simple scalar field
#     ('Y', float32[:]),          # an array field
# ]
class Quad:
 def __init__(_self,_X,_Y):

  _self.NUMRULE = 12
  _self.NUMGLEU = 6 
  _self.NUMGLEP = 3 

  _self.X = _X
  _self.Y = _Y
  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.gvx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.area = 0.0

  _self.gqPoints=[ [0.249286745171, 0.249286745171, 0.501426509658 ],
                   [0.249286745171, 0.501426509658, 0.249286745171 ],
                   [0.501426509658, 0.249286745171, 0.249286745171 ],
                   [0.063089014492, 0.063089014492, 0.873821971017 ],
                   [0.063089014492, 0.873821971017, 0.063089014491 ],
                   [0.873821971017, 0.063089014492, 0.063089014491 ],
                   [0.310352451034, 0.636502499121, 0.053145049845 ],
                   [0.636502499121, 0.053145049845, 0.310352451034 ],
                   [0.053145049845, 0.310352451034, 0.636502499121 ],
                   [0.636502499121, 0.310352451034, 0.053145049845 ],
                   [0.310352451034, 0.053145049845, 0.636502499121 ],
                   [0.053145049845, 0.636502499121, 0.310352451034 ] ]

  _self.gqWeights = [ 0.116786275726,
                      0.116786275726,
                      0.116786275726,
                      0.050844906370,
                      0.050844906370,
                      0.050844906370,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618 ]

  _self.phiJ = [ 
[-0.124998982535, -0.124998982535, 0.001430579518, 0.248575525272, 0.499995930140, 0.499995930140 ],
[-0.124998982535, 0.001430579518, -0.124998982535, 0.499995930140, 0.499995930140, 0.248575525272 ],
[0.001430579518, -0.124998982535, -0.124998982535, 0.499995930140, 0.248575525272, 0.499995930140 ],
[-0.055128566992, -0.055128566992, 0.653307703047, 0.015920894998, 0.220514267970, 0.220514267970 ],
[-0.055128566992, 0.653307703047, -0.055128566992, 0.220514267970, 0.220514267970, 0.015920894998 ],
[0.653307703047, -0.055128566992, -0.055128566992, 0.220514267970, 0.015920894998, 0.220514267970 ],
[-0.117715163308, 0.173768363654, -0.047496257199, 0.790160442766, 0.135307828169, 0.065974785919 ],
[0.173768363654, -0.047496257199, -0.117715163308, 0.135307828169, 0.065974785919, 0.790160442766 ],
[-0.047496257199, -0.117715163308, 0.173768363654, 0.065974785919, 0.790160442766, 0.135307828169 ],
[0.173768363654, -0.117715163308, -0.047496257199, 0.790160442766, 0.065974785919, 0.135307828169 ],
[-0.117715163308, -0.047496257199, 0.173768363654, 0.065974785919, 0.135307828169, 0.790160442766 ],
[-0.047496257199, 0.173768363654, -0.117715163308, 0.135307828169, 0.790160442766, 0.065974785919 ] ]



  _self.dphiJdl1 = [ 
[-0.002853019316, 0.000000000000, -1.005706038632, 0.997146980684, -0.997146980684, 1.008559057948 ],
[-0.002853019316, 0.000000000000, 0.002853019316, 2.005706038632, -2.005706038632, 0.000000000000  ],
[1.005706038632, 0.000000000000, 0.002853019316, 0.997146980684,  -0.997146980684, -1.008559057948 ],
[-0.747643942034, 0.000000000000, -2.495287884068, 0.252356057966, -0.252356057966, 3.242931826102 ],
[-0.747643942034, 0.000000000000, 0.747643942034, 3.495287884068,  -3.495287884068, -0.000000000000],
[2.495287884068, 0.000000000000, 0.747643942034, 0.252356057966, -0.252356057966, -3.242931826102  ],
[0.241409804136, 0.000000000000, 0.787419800620, 2.546009996484, -2.546009996484, -1.028829604756  ],
[1.546009996484, 0.000000000000, -0.241409804137, 0.212580199379, -0.212580199379, -1.304600192347 ],
[-0.787419800621, 0.000000000000, -1.546009996485, 1.241409804136, -1.241409804136, 2.333429797106 ],
[1.546009996484, 0.000000000000, 0.787419800620, 1.241409804136, -1.241409804136, -2.333429797104  ],
[0.241409804136, 0.000000000000, -1.546009996485, 0.212580199379, -0.212580199379, 1.304600192349  ],
[-0.787419800621, 0.000000000000, -0.241409804137, 2.546009996484, -2.546009996484, 1.028829604758 ] ]


  _self.dphiJdl2 = [
[0.000000000000, -0.002853019316, -1.005706038632, 0.997146980684, 1.008559057948, -0.997146980684 ],
[0.000000000000, 1.005706038632, 0.002853019316, 0.997146980684, -1.008559057948, -0.997146980684  ],
[0.000000000000, -0.002853019316, 0.002853019316, 2.005706038632, -0.000000000000, -2.005706038632 ],
[0.000000000000, -0.747643942034, -2.495287884068, 0.252356057966, 3.242931826102, -0.252356057966 ],
[0.000000000000, 2.495287884068, 0.747643942034, 0.252356057966, -3.242931826102, -0.252356057966  ],
[0.000000000000, -0.747643942034, 0.747643942034, 3.495287884068, -0.000000000000, -3.495287884068 ],
[0.000000000000, 1.546009996484, 0.787419800620, 1.241409804136, -2.333429797104, -1.241409804136  ],
[0.000000000000, -0.787419800621, -0.241409804137, 2.546009996484, 1.028829604758, -2.546009996484 ],
[0.000000000000, 0.241409804136, -1.546009996485, 0.212580199379, 1.304600192349, -0.212580199379  ],
[0.000000000000, 0.241409804136, 0.787419800620, 2.546009996484, -1.028829604756, -2.546009996484  ],
[0.000000000000, -0.787419800621, -1.546009996485, 1.241409804136, 2.333429797106, -1.241409804136 ],
[0.000000000000, 1.546009996484, -0.241409804137, 0.212580199379, -1.304600192347, -0.212580199379 ] ]



  _self.dgqPointsdl1 = [ [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1] ]

  _self.dgqPointsdl2 = [ [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1] ]


 def getM(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   # compute det(jacobian)
   jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];

  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.gvx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*\
                         jacob[k]*_self.gqWeights[k];
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.gvx[i][j] += _self.phiJ[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.gvy[i][j] += _self.phiJ[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];

    for j in range(0,_self.NUMGLEP):
     _self.gx[i][j] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.gy[i][j] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dx[j][i] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dy[j][i] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];

   # compute area 
   _self.area += (1.0/2.0)*jacob[k]*_self.gqWeights[k]
   
 def getMSolid(_self,v,D):

   localx = [0.0]*_self.NUMRULE
   localy = [0.0]*_self.NUMRULE
   for k in range(0,_self.NUMRULE):
    for i in range(0,_self.NUMGLEU):
     localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
     localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

   dxdl1 = [0.0]*_self.NUMRULE
   dxdl2 = [0.0]*_self.NUMRULE
   dydl1 = [0.0]*_self.NUMRULE
   dydl2 = [0.0]*_self.NUMRULE
   jacob = [0.0]*_self.NUMRULE # area for each Gauss point
   for k in range(0,_self.NUMRULE): 
    for i in range(0,_self.NUMGLEU): 
     dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
     dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
     dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
     dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
    # compute det(jacobian)
    jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]

   dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
   dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
   for k in range(0,_self.NUMRULE): 
    for i in range(0,_self.NUMGLEU):
     dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
 	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
     dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
 	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];

   _self.mass_solid = np.zeros((2*_self.NUMGLEU,2*_self.NUMGLEU), dtype=np.float)
   _self.k_solid  = np.zeros((2*_self.NUMGLEU,2*_self.NUMGLEU), dtype=np.float)

   for k in range(0,_self.NUMRULE): 
    B = np.array([[dphiJdx[k][0], 0, dphiJdx[k][1], 0, dphiJdx[k][2], 0, dphiJdx[k][3], 0, dphiJdx[k][4], 0, dphiJdx[k][5], 0],
            [0, dphiJdy[k][0], 0, dphiJdy[k][1], 0, dphiJdy[k][2], 0, dphiJdy[k][3], 0, dphiJdy[k][4], 0, dphiJdy[k][5]],
            [dphiJdy[k][0], dphiJdx[k][0], dphiJdy[k][1], dphiJdx[k][1], dphiJdy[k][2], dphiJdx[k][2], dphiJdy[k][3], dphiJdx[k][3], dphiJdy[k][4], dphiJdx[k][4], dphiJdy[k][5], dphiJdx[k][5]]]);
    
    _self.k_solid = _self.k_solid + 0.5*np.transpose(B)@D@B*jacob[k]*_self.gqWeights[k]
    
    N = np.array([[_self.phiJ[k][0], 0, _self.phiJ[k][1], 0, _self.phiJ[k][2], 0, _self.phiJ[k][3], 0, _self.phiJ[k][4], 0, _self.phiJ[k][5], 0],
           [0, _self.phiJ[k][0], 0, _self.phiJ[k][1], 0, _self.phiJ[k][2], 0, _self.phiJ[k][3], 0, _self.phiJ[k][4], 0, _self.phiJ[k][5]]])
    
    _self.mass_solid = _self.mass_solid + 0.5*np.transpose(N)@N*jacob[k]*_self.gqWeights[k]

    # compute area 
    _self.area += (1.0/2.0)*jacob[k]*_self.gqWeights[k]
 
 def getMSolidHE(_self,v,D,ux,uy,S):

    localx = [0.0]*_self.NUMRULE
    localy = [0.0]*_self.NUMRULE
    for k in range(0,_self.NUMRULE):
     for i in range(0,_self.NUMGLEU):
      localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
      localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

    dxdl1 = [0.0]*_self.NUMRULE
    dxdl2 = [0.0]*_self.NUMRULE
    dydl1 = [0.0]*_self.NUMRULE
    dydl2 = [0.0]*_self.NUMRULE
    jacob = [0.0]*_self.NUMRULE # area for each Gauss point
    sigma_x = [0.0]*_self.NUMRULE
    sigma_y = [0.0]*_self.NUMRULE
    sigma_xy = [0.0]*_self.NUMRULE
    duxdx = [0.0]*_self.NUMRULE
    duxdy = [0.0]*_self.NUMRULE
    duydx = [0.0]*_self.NUMRULE
    duydy = [0.0]*_self.NUMRULE
    
    for k in range(0,_self.NUMRULE): 
     for i in range(0,_self.NUMGLEU): 
      dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
      dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
      dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
      dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
     # compute det(jacobian)
     jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]

    dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
    dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
    for k in range(0,_self.NUMRULE): 
     for i in range(0,_self.NUMGLEU):
      dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
  	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
      dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
  	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];
      
      # sigma_x[k] = sigma_x[k] + _self.phiJ[k][i]*S[v[i],0]
      # sigma_y[k] = sigma_x[k] + _self.phiJ[k][i]*S[v[i],1]
      # sigma_xy[k] = sigma_x[k] + _self.phiJ[k][i]*S[v[i],2]
      
      duxdx[k] = duxdx[k] + dphiJdx[k][i]*ux[v[i]]
      duxdy[k] = duxdy[k] + dphiJdy[k][i]*ux[v[i]]
      duydx[k] = duydx[k] + dphiJdx[k][i]*uy[v[i]]
      duydy[k] = duydy[k] + dphiJdy[k][i]*uy[v[i]]
     
     E_x = duxdx[k] +(1.0/2.0)*(duxdx[k]**2 + duydx[k]**2)
     E_y = duydy[k] + (1.0/2.0)*(duxdy[k]**2 + duydy[k]**2)
     E_xy = duxdy[k] + duydx[k] + duxdy[k]*duxdx[k] + duydx[k]*duydy[k]
     
     sigma = D@np.array([E_x,E_y,E_xy])
     sigma_x[k] = sigma[0]
     sigma_y[k] = sigma[1]
     sigma_xy[k] = sigma[2]

    _self.mass_solid = np.zeros((2*_self.NUMGLEU,2*_self.NUMGLEU), dtype=np.float)
    _self.mass_stress = np.zeros((2*_self.NUMGLEU,3*_self.NUMGLEU), dtype=np.float)
    _self.res_stress = np.zeros((2*_self.NUMGLEU), dtype=np.float)
    _self.kN  = np.zeros((2*_self.NUMGLEU,2*_self.NUMGLEU), dtype=np.float)
    _self.kG  = np.zeros((2*_self.NUMGLEU,2*_self.NUMGLEU), dtype=np.float)
    _self.k_solid  = np.zeros((2*_self.NUMGLEU,2*_self.NUMGLEU), dtype=np.float)

    for k in range(0,_self.NUMRULE): 
     F11 = 1.0 + np.array([dphiJdx[k][0], dphiJdx[k][1], dphiJdx[k][2], dphiJdx[k][3], dphiJdx[k][4], dphiJdx[k][5]])@ux[v]
     F12 = np.array([dphiJdy[k][0], dphiJdy[k][1], dphiJdy[k][2], dphiJdy[k][3], dphiJdy[k][4], dphiJdy[k][5]])@ux[v]
     F21 = np.array([dphiJdx[k][0], dphiJdx[k][1], dphiJdx[k][2], dphiJdx[k][3], dphiJdx[k][4], dphiJdx[k][5]])@uy[v]
     F22 = 1.0 + np.array([dphiJdy[k][0], dphiJdy[k][1], dphiJdy[k][2], dphiJdy[k][3], dphiJdy[k][4], dphiJdy[k][5]])@uy[v]
     
     BN = np.array([[F11*dphiJdx[k][0], F21*dphiJdx[k][0], F11*dphiJdx[k][1], F21*dphiJdx[k][1], F11*dphiJdx[k][2], F21*dphiJdx[k][2], F11*dphiJdx[k][3], F21*dphiJdx[k][3], F11*dphiJdx[k][4], F21*dphiJdx[k][4], F11*dphiJdx[k][5], F21*dphiJdx[k][5]],
             [F12*dphiJdy[k][0], F22*dphiJdy[k][0], F12*dphiJdy[k][1], F22*dphiJdy[k][1], F12*dphiJdy[k][2], F22*dphiJdy[k][2], F12*dphiJdy[k][3], F22*dphiJdy[k][3], F12*dphiJdy[k][4], F22*dphiJdy[k][4], F12*dphiJdy[k][5], F22*dphiJdy[k][5]],
             [F11*dphiJdy[k][0] + F12*dphiJdx[k][0], F21*dphiJdy[k][0] + F22*dphiJdx[k][0], F11*dphiJdy[k][1] + F12*dphiJdx[k][1], F21*dphiJdy[k][1] + F22*dphiJdx[k][1], F11*dphiJdy[k][2] + F12*dphiJdx[k][2], F21*dphiJdy[k][2] + F22*dphiJdx[k][2], F11*dphiJdy[k][3] + F12*dphiJdx[k][3], F21*dphiJdy[k][3] + F22*dphiJdx[k][3], F11*dphiJdy[k][4] + F12*dphiJdx[k][4], F21*dphiJdy[k][4] + F22*dphiJdx[k][4], F11*dphiJdy[k][5] + F12*dphiJdx[k][5], F21*dphiJdy[k][5] + F22*dphiJdx[k][5]]]);
     
     
     _self.kN = _self.kN + 0.5*np.transpose(BN)@D@BN*jacob[k]*_self.gqWeights[k]
     
     SIGMA = np.array([[sigma_x[k], sigma_xy[k], 0, 0], [sigma_xy[k], sigma_y[k], 0, 0], [0 , 0 , sigma_x[k], sigma_xy[k]], [0 , 0 , sigma_xy[k], sigma_y[k]]])
     
     BG = np.array([[dphiJdx[k][0], 0,  dphiJdx[k][1], 0, dphiJdx[k][2], 0, dphiJdx[k][3], 0, dphiJdx[k][4], 0, dphiJdx[k][5], 0],
                    [dphiJdy[k][0], 0,  dphiJdy[k][1], 0, dphiJdy[k][2], 0, dphiJdy[k][3], 0, dphiJdy[k][4], 0, dphiJdy[k][5], 0],
                    [0, dphiJdx[k][0], 0,  dphiJdx[k][1], 0, dphiJdx[k][2], 0, dphiJdx[k][3], 0, dphiJdx[k][4], 0, dphiJdx[k][5]],
                    [0, dphiJdy[k][0], 0,  dphiJdy[k][1], 0, dphiJdy[k][2], 0, dphiJdy[k][3], 0, dphiJdy[k][4], 0, dphiJdy[k][5]]])
     
     
     
     _self.kG = _self.kG + 0.5*np.transpose(BG)@SIGMA@BG*jacob[k]*_self.gqWeights[k]
     
     _self.k_solid = _self.kN + _self.kG
     
     N = np.array([[_self.phiJ[k][0], 0, _self.phiJ[k][1], 0, _self.phiJ[k][2], 0, _self.phiJ[k][3], 0, _self.phiJ[k][4], 0, _self.phiJ[k][5], 0],
            [0, _self.phiJ[k][0], 0, _self.phiJ[k][1], 0, _self.phiJ[k][2], 0, _self.phiJ[k][3], 0, _self.phiJ[k][4], 0, _self.phiJ[k][5]]])
     
     _self.mass_solid = _self.mass_solid + 0.5*np.transpose(N)@N*jacob[k]*_self.gqWeights[k]
     
     N_S = np.array([[_self.phiJ[k][0], 0, 0, _self.phiJ[k][1], 0, 0, _self.phiJ[k][2], 0, 0, _self.phiJ[k][3], 0, 0, _self.phiJ[k][4], 0, 0, _self.phiJ[k][5], 0, 0],
                     [0, _self.phiJ[k][0], 0, 0, _self.phiJ[k][1], 0, 0, _self.phiJ[k][2], 0, 0, _self.phiJ[k][3], 0, 0, _self.phiJ[k][4], 0, 0, _self.phiJ[k][5], 0],
                     [0, 0, _self.phiJ[k][0], 0, 0, _self.phiJ[k][1], 0, 0, _self.phiJ[k][2], 0, 0, _self.phiJ[k][3], 0, 0, _self.phiJ[k][4], 0, 0, _self.phiJ[k][5]]])
     
     stress = np.array([sigma_x[k],sigma_y[k],sigma_xy[k]])
     _self.mass_stress = _self.mass_stress + 0.5*np.transpose(BN)@N_S*jacob[k]*_self.gqWeights[k]
     _self.res_stress = _self.res_stress + 0.5*np.transpose(BN)@stress*jacob[k]*_self.gqWeights[k]
     # compute area 
     _self.area += (1.0/2.0)*jacob[k]*_self.gqWeights[k]
     # if k == 0:
     #     print(N)
     #     print('\n')
     #     print(BN)
     #     print('\n')
     #     print(BG)
     #     print('\n')
     #     print(SIGMA)


     
 def getMass_porous(_self,v,porous_nodes):

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   # compute det(jacobian)
   jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];

  _self.mass_porous = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.GQ_porous_values = _self.phiJ@porous_nodes[v]
  
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass_porous[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*_self.GQ_porous_values[k]*\
                         jacob[k]*_self.gqWeights[k];

 def getMass_Forchheimer(_self,v,porous_nodes,vx,vy):

    localx = [0.0]*_self.NUMRULE
    localy = [0.0]*_self.NUMRULE
    for k in range(0,_self.NUMRULE):
     for i in range(0,_self.NUMGLEU):
      localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
      localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]
    
    dxdl1 = [0.0]*_self.NUMRULE
    dxdl2 = [0.0]*_self.NUMRULE
    dydl1 = [0.0]*_self.NUMRULE
    dydl2 = [0.0]*_self.NUMRULE
    jacob = [0.0]*_self.NUMRULE # area for each Gauss point
    for k in range(0,_self.NUMRULE): 
     for i in range(0,_self.NUMGLEU): 
      dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
      dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
      dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
      dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
     # compute det(jacobian)
     jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]
    
    dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
    dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
    for k in range(0,_self.NUMRULE): 
     for i in range(0,_self.NUMGLEU):
      dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
     	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
      dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
     	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];
    
    _self.mass_forchheimer = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
    _self.GQ_porous_values = _self.phiJ@porous_nodes[v]
    # _self.GQ_velocity_mod = _self.phiJ@vel[v]
    _self.GQ_velocity_mod = np.zeros((len(vx)),dtype='float')
    
    for k in range(0,_self.NUMRULE): 
     _self.GQ_velocity_mod[k] = np.sqrt(((_self.phiJ@vx[v])[k])**2 + ((_self.phiJ@vy[v])[k])**2)
     for i in range(0,_self.NUMGLEU):
      for j in range(0,_self.NUMGLEU):
       _self.mass_forchheimer[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*_self.GQ_porous_values[k]*_self.GQ_velocity_mod[k]*\
                           jacob[k]*_self.gqWeights[k];



 def getMSlip(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   # compute det(jacobian)
   jacob[k] = abs(dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k])

  # gqPoints
  localxlin = [0.0]*_self.NUMRULE
  localylin = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    localxlin[k] += _self.X[v[i]] * _self.gqPoints[k][i]
    localylin[k] += _self.Y[v[i]] * _self.gqPoints[k][i]

  dxdl1lin = [0.0]*_self.NUMRULE
  dxdl2lin = [0.0]*_self.NUMRULE
  dydl1lin = [0.0]*_self.NUMRULE
  dydl2lin = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEP): 
    dxdl1lin[k] += _self.X[v[i]]*_self.dgqPointsdl1[k][i]
    dxdl2lin[k] += _self.X[v[i]]*_self.dgqPointsdl2[k][i]
    dydl1lin[k] += _self.Y[v[i]]*_self.dgqPointsdl1[k][i]
    dydl2lin[k] += _self.Y[v[i]]*_self.dgqPointsdl2[k][i]

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dgqPointsdx = [[0.0]*_self.NUMGLEP for i in range(_self.NUMRULE)]
  dgqPointsdy = [[0.0]*_self.NUMGLEP for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];
   for i in range(0,_self.NUMGLEP):
    dgqPointsdx[k][i] = (  _self.dgqPointsdl1[k][i]*dydl2lin[k]-
	                   _self.dgqPointsdl2[k][i]*dydl1lin[k] )/jacob[k];
    dgqPointsdy[k][i] = ( -_self.dgqPointsdl1[k][i]*dxdl2lin[k]+
	                   _self.dgqPointsdl2[k][i]*dxdl1lin[k] )/jacob[k];

  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gvy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*\
                         jacob[k]*_self.gqWeights[k];
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.gvx[i][j] += _self.phiJ[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.gvy[i][j] += _self.phiJ[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];

    for j in range(0,_self.NUMGLEP):
     _self.gx[i][j] += dgqPointsdx[k][j]*_self.phiJ[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.gy[i][j] += dgqPointsdy[k][j]*_self.phiJ[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dx[j][i] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dy[j][i] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];

 def getK(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = np.zeros((_self.NUMRULE), dtype=np.float)
  localy = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE):
   valx = 0.0;
   valy = 0.0;
   for i in range(0,_self.NUMGLEU):
    valx += _self.X[v[i]] * _self.phiJ[k][i]
    valy += _self.Y[v[i]] * _self.phiJ[k][i]
   localx[k] = valx;
   localy[k] = valy;

  dxdl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dxdl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   valxl1 = 0.0;
   valxl2 = 0.0;
   valyl1 = 0.0;
   valyl2 = 0.0;
   for i in range(0,_self.NUMGLEU): 
    valxl1 += _self.X[v[i]]*_self.dphiJdl1[k][i]
    valxl2 += _self.X[v[i]]*_self.dphiJdl2[k][i]
    valyl1 += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    valyl2 += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   dxdl1[k] = valxl1
   dxdl2[k] = valxl2
   dydl1[k] = valyl1
   dydl2[k] = valyl2
   # compute det(jacobian)
   jacob[k] = dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k]

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = ( _self.dphiJdl1[k][i]*dydl2[k]-_self.dphiJdl2[k][i]\
                  *dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+_self.dphiJdl2[k][i]\
                  *dxdl1[k] )/jacob[k];

  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]\
                     *jacob[k]*_self.gqWeights[k];



class QuadBubble:
 def __init__(_self,_X,_Y):

  _self.NUMRULE = 12
  _self.NUMGLEU = 7 
  _self.NUMGLEP = 3 

  _self.X = _X
  _self.Y = _Y
  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.area = 0.0

  _self.gqPoints=[ [0.249286745171, 0.249286745171, 0.501426509658 ],
                   [0.249286745171, 0.501426509658, 0.249286745171 ],
                   [0.501426509658, 0.249286745171, 0.249286745171 ],
                   [0.063089014492, 0.063089014492, 0.873821971017 ],
                   [0.063089014492, 0.873821971017, 0.063089014491 ],
                   [0.873821971017, 0.063089014492, 0.063089014491 ],
                   [0.310352451034, 0.636502499121, 0.053145049845 ],
                   [0.636502499121, 0.053145049845, 0.310352451034 ],
                   [0.053145049845, 0.310352451034, 0.636502499121 ],
                   [0.636502499121, 0.310352451034, 0.053145049845 ],
                   [0.310352451034, 0.053145049845, 0.636502499121 ],
                   [0.053145049845, 0.636502499121, 0.310352451034 ] ]

  _self.gqWeights = [ 0.116786275726,
                      0.116786275726,
                      0.116786275726,
                      0.050844906370,
                      0.050844906370,
                      0.050844906370,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618 ]

  _self.phiJ = [ 
[-0.031517214017514, -0.031517214017514, 0.094912348035192, -0.125351548798530, 0.126068856070057, 0.126068856070057, 0.841335916658253 ],
[-0.031517214017514, 0.094912348035192, -0.031517214017514, 0.126068856070057, 0.126068856070057, -0.125351548798530, 0.841335916658253 ],
[0.094912348035192, -0.031517214017514, -0.031517214017514, 0.126068856070057, -0.125351548798530, 0.126068856070057, 0.841335916658253 ],
[-0.044694546106830, -0.044694546106830, 0.663741723932723, -0.025815188544578, 0.178778184427318, 0.178778184427318, 0.093906187970878 ],
[-0.044694546106830, 0.663741723932723, -0.044694546106829, 0.178778184427318, 0.178778184427318, -0.025815188544578, 0.093906187970878 ],
[0.663741723932723, -0.044694546106830, -0.044694546106829, 0.178778184427318, -0.025815188544578, 0.178778184427318, 0.093906187970878 ],
[-0.086220326221122, 0.205263200740812, -0.016001420111690, 0.664181094416856, 0.009328479819991, -0.060004562430140, 0.283453533785293 ],
[0.205263200740714, -0.016001420111631, -0.086220326221172, 0.009328479819875, -0.060004562429953, 0.664181094417758, 0.283453533784409 ],
[-0.016001420111641, -0.086220326221231, 0.205263200741013, -0.060004562429953, 0.664181094417539, 0.009328479819959, 0.283453533784315 ],
[0.205263200740812, -0.086220326221122, -0.016001420111690, 0.664181094416856, -0.060004562430140, 0.009328479819991, 0.283453533785293 ],
[-0.086220326221231, -0.016001420111641, 0.205263200741013, -0.060004562429953, 0.009328479819959, 0.664181094417539, 0.283453533784315 ],
[-0.016001420111631, 0.205263200740714, -0.086220326221172, 0.009328479819875, 0.664181094417758, -0.060004562429953, 0.283453533784409 ] ]


  _self.dphiJdl1 = [ 
[0.185712284335440, 0.188565303651440, -0.817140734980560, 0.242885766078239, -1.751408195289760, 0.254297843342239, 1.697087732862961    ],
[-0.002853019316000, 0.000000000000000, 0.002853019316000, 2.005706038632000, -2.005706038632000, -0.000000000000000, 0.000000000000000   ],
[0.817140734980560, -0.188565303651440, -0.185712284335440, 1.751408195289760, -0.242885766078240, -0.254297843342239, -1.697087732862961 ],
[-0.594198912305078, 0.153445029728922, -2.341842854339078, -0.361424060949687, -0.866136176881687, 2.629151707186313, 1.381005267560296  ],
[-0.747643942034000, -0.000000000000000, 0.747643942034000, 3.495287884068000, -3.495287884068000, 0.000000000000000, -0.000000000000001  ],
[2.341842854339078, -0.153445029728922, 0.594198912305078, 0.866136176881687, 0.361424060949687, -2.629151707186313, -1.381005267560296   ],
[-0.249729656811648, -0.491139460947648, 0.296280339672351, 4.510567840274594, -0.581452152693406, 0.935728239034594, -4.420255148528836  ],
[1.494010214796629, -0.051999781687371, -0.293409585824171, 0.420579326128683, -0.004581072629717, -1.096601065597717, -0.467998035186336 ],
[-0.244280557985181, 0.543139242635619, -1.002870753849181, -0.931147166406477, -3.413966774678477, 0.160872826563123, 4.888253183720574  ],
[1.002870753848753, -0.543139242635247, 0.244280557984753, 3.413966774676987, 0.931147166404987, -0.160872826563013, -4.888253183717221   ],
[0.293409585823434, 0.051999781687434, -1.494010214797366, 0.004581072629462, -0.420579326128938, 1.096601065599062, 0.467998035186910    ],
[-0.296280339672388, 0.491139460948412, 0.249729656811612, 0.581452152690350, -4.510567840277649, -0.935728239036049, 4.420255148535712   ] ]

  _self.dphiJdl2 = [
[0.188565303651440, 0.185712284335440, -0.817140734980560, 0.242885766078239, 0.254297843342239, -1.751408195289760, 1.697087732862961    ],
[-0.188565303651440, 0.817140734980560, -0.185712284335440, 1.751408195289760, -0.254297843342239, -0.242885766078240, -1.697087732862961 ],
[-0.000000000000000, -0.002853019316000, 0.002853019316000, 2.005706038632000, -0.000000000000000, -2.005706038632000, -0.000000000000001 ],
[0.153445029728922, -0.594198912305078, -2.341842854339078, -0.361424060949687, 2.629151707186313, -0.866136176881687, 1.381005267560296  ],  
[-0.153445029728922, 2.341842854339078, 0.594198912305078, 0.866136176881687, -2.629151707186313, 0.361424060949687, -1.381005267560296   ],
[-0.000000000000000, -0.747643942034000, 0.747643942034000, 3.495287884068000, -0.000000000000001, -3.495287884068000, -0.000000000000001 ],
[-0.543139242635247, 1.002870753848753, 0.244280557984753, 3.413966774676987, -0.160872826563013, 0.931147166404987, -4.888253183717221   ],
[0.491139460948412, -0.296280339672388, 0.249729656811612, 0.581452152690350, -0.935728239036049, -4.510567840277649, 4.420255148535712   ],
[0.051999781687434, 0.293409585823434, -1.494010214797366, 0.004581072629462, 1.096601065599062, -0.420579326128938, 0.467998035186910    ],
[-0.491139460947648, -0.249729656811648, 0.296280339672351, 4.510567840274594, 0.935728239034594, -0.581452152693406, -4.420255148528836  ],
[0.543139242635619, -0.244280557985181, -1.002870753849181, -0.931147166406477, 0.160872826563123, -3.413966774678477, 4.888253183720574  ],
[-0.051999781687371, 1.494010214796629, -0.293409585824171, 0.420579326128683, -1.096601065597717, -0.004581072629717, -0.467998035186336 ] ]


  _self.dgqPointsdl1 = [ [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1] ]

  _self.dgqPointsdl2 = [ [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1] ]

 def getM(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   # compute det(jacobian)
   jacob[k] = abs(dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k])

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];

  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*\
                         jacob[k]*_self.gqWeights[k];
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];

    for j in range(0,_self.NUMGLEP):
     _self.gx[i][j] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.gy[i][j] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dx[j][i] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dy[j][i] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];

   # compute area 
   _self.area += (1.0/2.0)*jacob[k]*_self.gqWeights[k]

 def getK(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = np.zeros((_self.NUMRULE), dtype=np.float)
  localy = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE):
   valx = 0.0;
   valy = 0.0;
   for i in range(0,_self.NUMGLEU):
    valx += _self.X[v[i]] * _self.phiJ[k][i]
    valy += _self.Y[v[i]] * _self.phiJ[k][i]
   localx[k] = valx;
   localy[k] = valy;

  dxdl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dxdl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   valxl1 = 0.0;
   valxl2 = 0.0;
   valyl1 = 0.0;
   valyl2 = 0.0;
   for i in range(0,_self.NUMGLEU): 
    valxl1 += _self.X[v[i]]*_self.dphiJdl1[k][i]
    valxl2 += _self.X[v[i]]*_self.dphiJdl2[k][i]
    valyl1 += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    valyl2 += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   dxdl1[k] = valxl1
   dxdl2[k] = valxl2
   dydl1[k] = valyl1
   dydl2[k] = valyl2
   # compute det(jacobian)
   jacob[k] = abs(dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k])

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = ( _self.dphiJdl1[k][i]*dydl2[k]-_self.dphiJdl2[k][i]\
                  *dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+_self.dphiJdl2[k][i]\
                  *dxdl1[k] )/jacob[k];

  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]\
                     *jacob[k]*_self.gqWeights[k];

class Cubic:
 def __init__(_self,_X,_Y):

  _self.NUMRULE = 12
  _self.NUMGLEU = 10 
  _self.NUMGLEP = 3 

  _self.X = _X
  _self.Y = _Y
  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.area = 0.0

  _self.gqPoints=[ [0.249286745171, 0.249286745171, 0.501426509658 ],
                   [0.249286745171, 0.501426509658, 0.249286745171 ],
                   [0.501426509658, 0.249286745171, 0.249286745171 ],
                   [0.063089014492, 0.063089014492, 0.873821971017 ],
                   [0.063089014492, 0.873821971017, 0.063089014491 ],
                   [0.873821971017, 0.063089014492, 0.063089014491 ],
                   [0.310352451034, 0.636502499121, 0.053145049845 ],
                   [0.636502499121, 0.053145049845, 0.310352451034 ],
                   [0.053145049845, 0.310352451034, 0.636502499121 ],
                   [0.636502499121, 0.310352451034, 0.053145049845 ],
                   [0.310352451034, 0.053145049845, 0.636502499121 ],
                   [0.053145049845, 0.636502499121, 0.310352451034 ] ]

  _self.gqWeights = [ 0.116786275726,
                      0.116786275726,
                      0.116786275726,
                      0.050844906370,
                      0.050844906370,
                      0.050844906370,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618,
                      0.082851075618 ]

  _self.phiJ = [ 
[0.039351685817, 0.039351685817, -0.062673722052, -0.070510246199, -0.070510246199, -0.141827463079, 0.283654926158, 0.283654926158, -0.141827463079, 0.841335916658 ],
[0.039351685817, -0.062673722052, 0.039351685817, -0.141827463079, 0.283654926158, 0.283654926158, -0.141827463079, -0.070510246199, -0.070510246199, 0.841335916658 ],
[-0.062673722052, 0.039351685817, 0.039351685817, 0.283654926158, -0.141827463079, -0.070510246199, -0.070510246199, -0.141827463079, 0.283654926158, 0.841335916658 ],
[0.046307995391, 0.046307995391, 0.440268993399, -0.014521043556, -0.014521043556, -0.201125457481, 0.402250914961, 0.402250914961, -0.201125457481, 0.093906187971  ],
[0.046307995391, 0.440268993399, 0.046307995391, -0.201125457481, 0.402250914961, 0.402250914961, -0.201125457481, -0.014521043556, -0.014521043556, 0.093906187971  ],
[0.440268993399, 0.046307995391, 0.046307995391, 0.402250914961, -0.201125457481, -0.014521043556, -0.014521043556, -0.201125457481, 0.402250914961, 0.093906187971  ],
[0.011435826065, -0.026193226600, 0.041110728467, -0.061285221448, 0.808488952667, 0.138446419693, -0.127951879896, -0.062388096818, -0.005117035916, 0.283453533785 ],
[-0.026193226600, 0.041110728466, 0.011435826065, 0.138446419693, -0.127951879895, -0.062388096818, -0.005117035916, -0.061285221448, 0.808488952668, 0.283453533784 ],
[0.041110728466, 0.011435826065, -0.026193226600, -0.062388096818, -0.005117035916, -0.061285221448, 0.808488952668, 0.138446419693, -0.127951879895, 0.283453533784 ],
[-0.026193226600, 0.011435826065, 0.041110728467, 0.808488952667, -0.061285221448, -0.005117035916, -0.062388096818, -0.127951879896, 0.138446419693, 0.283453533785 ],
[0.011435826065, 0.041110728466, -0.026193226600, -0.005117035916, -0.062388096818, -0.127951879895, 0.138446419693, 0.808488952668, -0.061285221448, 0.283453533784 ],
[0.041110728466, -0.026193226600, 0.011435826065, -0.127951879895, 0.138446419693, 0.808488952668, -0.061285221448, -0.005117035916, -0.062388096818, 0.283453533784 ] ]


  _self.dphiJdl1 = [ 
[-0.404638308747, 0.000000000000, 0.118553234987, 0.556094442315, -0.282847955477, 0.282847955477, -2.253182175178, -1.115316116704, 1.401401190464, 1.697087732863 ],
[-0.404638308747, 0.000000000000, 0.404638308747, 1.118553234987, 1.137866058474, -1.137866058474, -1.118553234987, -0.838942397792, 0.838942397792, 0.000000000000 ],
[-0.118553234987, 0.000000000000, 0.404638308747, 2.253182175178, -0.282847955477, 0.282847955477, -0.556094442315, -1.401401190464, 1.115316116704, -1.697087732863],
[0.485931890195, 0.000000000000, -3.443727560779, -0.176434523975, -0.230167544593, 0.230167544593, -1.204570743585, 5.171355686771, -2.213560016186, 1.381005267560],
[0.485931890195, 0.000000000000, -0.485931890195, -2.443727560779, 6.375926430356, -6.375926430356, 2.443727560779, -0.053733020618, 0.053733020618, -0.000000000000],
[3.443727560779, 0.000000000000, -0.485931890195, 1.204570743585, -0.230167544593, 0.230167544593, 0.176434523975, 2.213560016186, -5.171355686771, -1.381005267560 ],
[-0.492870367158, 0.000000000000, -0.559823901756, 2.469321742625, 2.605067077684, -2.605067077684, 1.950933405904, 0.750232850759, 0.302461418155, -4.420255148529 ],
[0.740805831639, 0.000000000000, 0.492870367158, 0.674175115836, -0.201023373941, 0.201023373941, -0.206177080649, -2.565606080133, 1.331929881335, -0.467998035186 ],
[0.559823901757, 0.000000000000, -0.740805831641, -0.951256224702, -0.096284337505, 0.096284337505, -3.936996959018, 1.930891961850, -1.749910031967, 4.888253183721],
[0.740805831639, 0.000000000000, -0.559823901756, 3.936996959017, -0.096284337505, 0.096284337505, 0.951256224701, 1.749910031962, -1.930891961845, -4.888253183717 ],
[-0.492870367158, 0.000000000000, -0.740805831641, 0.206177080649, -0.201023373941, 0.201023373941, -0.674175115836, -1.331929881332, 2.565606080131, 0.467998035187],
[0.559823901757, 0.000000000000, 0.492870367158, -1.950933405907, 2.605067077684, -2.605067077684, -2.469321742629, -0.302461418154, -0.750232850762, 4.420255148536] ]

  _self.dphiJdl2 = [
[0.000000000000, -0.404638308747, 0.118553234987, -0.282847955477, 0.556094442315, 1.401401190464, -1.115316116704, -2.253182175178, 0.282847955477, 1.697087732863  ],
[0.000000000000, -0.118553234987, 0.404638308747, -0.282847955477, 2.253182175178, 1.115316116704, -1.401401190464, -0.556094442315, 0.282847955477, -1.697087732863 ],
[0.000000000000, -0.404638308747, 0.404638308747, 1.137866058474, 1.118553234987, 0.838942397792, -0.838942397792, -1.118553234987, -1.137866058474, -0.000000000000 ],
[0.000000000000, 0.485931890195, -3.443727560779, -0.230167544593, -0.176434523975, -2.213560016186, 5.171355686771, -1.204570743585, 0.230167544593, 1.381005267560 ],
[0.000000000000, 3.443727560779, -0.485931890195, -0.230167544593, 1.204570743585, -5.171355686771, 2.213560016186, 0.176434523975, 0.230167544593, -1.381005267560  ],
[0.000000000000, 0.485931890195, -0.485931890195, 6.375926430356, -2.443727560779, 0.053733020618, -0.053733020618, 2.443727560779, -6.375926430356, -0.000000000000 ],
[0.000000000000, 0.740805831639, -0.559823901756, -0.096284337505, 3.936996959017, -1.930891961845, 1.749910031962, 0.951256224701, 0.096284337505, -4.888253183717  ],
[0.000000000000, 0.559823901757, 0.492870367158, 2.605067077684, -1.950933405907, -0.750232850762, -0.302461418154, -2.469321742629, -2.605067077684, 4.420255148536 ],
[0.000000000000, -0.492870367158, -0.740805831641, -0.201023373941, 0.206177080649, 2.565606080131, -1.331929881332, -0.674175115836, 0.201023373941, 0.467998035187 ],
[0.000000000000, -0.492870367158, -0.559823901756, 2.605067077684, 2.469321742625, 0.302461418155, 0.750232850759, 1.950933405904, -2.605067077684, -4.420255148529  ],
[0.000000000000, 0.559823901757, -0.740805831641, -0.096284337505, -0.951256224702, -1.749910031967, 1.930891961850, -3.936996959018, 0.096284337505, 4.888253183721 ],
[0.000000000000, 0.740805831639, 0.492870367158, -0.201023373941, 0.674175115836, 1.331929881335, -2.565606080133, -0.206177080649, 0.201023373941, -0.467998035186  ] ]



  _self.dgqPointsdl1 = [ [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1] ]

  _self.dgqPointsdl2 = [ [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1],
                         [0, 1, -1] ]

 def getM(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = [0.0]*_self.NUMRULE
  localy = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  dxdl1 = [0.0]*_self.NUMRULE
  dxdl2 = [0.0]*_self.NUMRULE
  dydl1 = [0.0]*_self.NUMRULE
  dydl2 = [0.0]*_self.NUMRULE
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU): 
    dxdl1[k] += _self.X[v[i]]*_self.dphiJdl1[k][i]
    dxdl2[k] += _self.X[v[i]]*_self.dphiJdl2[k][i]
    dydl1[k] += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    dydl2[k] += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   # compute det(jacobian)
   jacob[k] = abs(dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k])

  dphiJdx = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  dphiJdy = [[0.0]*_self.NUMGLEU for i in range(_self.NUMRULE)]
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = (  _self.dphiJdl1[k][i]*dydl2[k]-
	                   _self.dphiJdl2[k][i]*dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+
	                   _self.dphiJdl2[k][i]*dxdl1[k] )/jacob[k];

  _self.mass = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.k    = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.gx   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.gy   = np.zeros((_self.NUMGLEU,_self.NUMGLEP), dtype=np.float)
  _self.dx   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  _self.dy   = np.zeros((_self.NUMGLEP,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.mass[i][j] += _self.phiJ[k][i]*_self.phiJ[k][j]*\
                         jacob[k]*_self.gqWeights[k];
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]*\
                        jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]*\
                        jacob[k]*_self.gqWeights[k];

    for j in range(0,_self.NUMGLEP):
     _self.gx[i][j] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.gy[i][j] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dx[j][i] += _self.gqPoints[k][j]*dphiJdx[k][i]*\
                       jacob[k]*_self.gqWeights[k];
     _self.dy[j][i] += _self.gqPoints[k][j]*dphiJdy[k][i]*\
                       jacob[k]*_self.gqWeights[k];

   # compute area 
   _self.area += (1.0/2.0)*jacob[k]*_self.gqWeights[k]

 def getK(_self,v):

  jacobian = _self.X[v[2]]*( _self.Y[v[0]]-_self.Y[v[1]]) \
           + _self.X[v[0]]*( _self.Y[v[1]]-_self.Y[v[2]]) \
           + _self.X[v[1]]*(-_self.Y[v[0]]+_self.Y[v[2]])

  localx = np.zeros((_self.NUMRULE), dtype=np.float)
  localy = np.zeros((_self.NUMRULE), dtype=np.float)
  for k in range(0,_self.NUMRULE):
   valx = 0.0;
   valy = 0.0;
   for i in range(0,_self.NUMGLEU):
    valx += _self.X[v[i]] * _self.phiJ[k][i]
    valy += _self.Y[v[i]] * _self.phiJ[k][i]
   localx[k] = valx;
   localy[k] = valy;

  dxdl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dxdl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl1 = np.zeros((_self.NUMRULE), dtype=np.float)
  dydl2 = np.zeros((_self.NUMRULE), dtype=np.float)
  jacob = [0.0]*_self.NUMRULE # area for each Gauss point
  for k in range(0,_self.NUMRULE): 
   valxl1 = 0.0;
   valxl2 = 0.0;
   valyl1 = 0.0;
   valyl2 = 0.0;
   for i in range(0,_self.NUMGLEU): 
    valxl1 += _self.X[v[i]]*_self.dphiJdl1[k][i]
    valxl2 += _self.X[v[i]]*_self.dphiJdl2[k][i]
    valyl1 += _self.Y[v[i]]*_self.dphiJdl1[k][i]
    valyl2 += _self.Y[v[i]]*_self.dphiJdl2[k][i]
   dxdl1[k] = valxl1
   dxdl2[k] = valxl2
   dydl1[k] = valyl1
   dydl2[k] = valyl2
   # compute det(jacobian)
   jacob[k] = abs(dxdl1[k]*dydl2[k] - dxdl2[k]*dydl1[k])

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = ( _self.dphiJdl1[k][i]*dydl2[k]-_self.dphiJdl2[k][i]\
                  *dydl1[k] )/jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdl1[k][i]*dxdl2[k]+_self.dphiJdl2[k][i]\
                  *dxdl1[k] )/jacob[k];

  _self.kxx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kxy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyx  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  _self.kyy  = np.zeros((_self.NUMGLEU,_self.NUMGLEU), dtype=np.float)
  for k in range(0,_self.NUMRULE): 
   for i in range(0,_self.NUMGLEU):
    for j in range(0,_self.NUMGLEU):
     _self.kxx[i][j] += dphiJdx[k][i]*dphiJdx[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kxy[i][j] += dphiJdx[k][i]*dphiJdy[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kyx[i][j] += dphiJdy[k][i]*dphiJdx[k][j]\
                     *jacob[k]*_self.gqWeights[k];
     _self.kyy[i][j] += dphiJdy[k][i]*dphiJdy[k][j]\
                     *jacob[k]*_self.gqWeights[k];

