#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Gustavo R. Anjos
# Email: gustavo.rabello@coppe.ufrj.br
# Date: 2025-03-01
# File: clElement.py

"""
Description:
"""

import numpy as np
from multiprocessing import get_context
from functools import partial
from tqdm import tqdm
from colorama import init, Fore, Style
import sys

# This class is aimed to transform coordinates in 1D.
#
# The gaussian points are used to compute the point jacobian (area
# associated to the transformation) and can be further used in
# isoparametric elements with curved boundary.
#
# The kxx,kyy,mass,gx,gy,dx and dy element matrices are computed based on
# the chosen element: triangle or quadrilateral.
#
#        ---o----o--->
#                    xi
#
class Element1D:
 def __init__(_self,_X,_Y):

  _self.X = _X
  _self.Y = _Y

  _self.length  = 0.0
  _self.jacob   = 0.0

  _self.local_t = 0.0 # computing time
  _self.jacob_t = 0.0
  _self.dphi_t  = 0.0
  _self.mat_t   = 0.0

 def getMatrices(_self,v):
  # convert to referential coordinates (xi)
  xi = []
  #        X                       xi
  # o--------------o  --->  o--------------o
  # |------h-------|        |------h1------|
  if len(v) == 2:
   h1 = np.sqrt( ((_self.X[v[0]]-_self.X[v[1]])**2) + \
                 ((_self.Y[v[0]]-_self.Y[v[1]])**2) )
   xi = [0.0,h1]
  #        X                       xi
  #      o
  #     / \
  # h1 /   \ h2
  #   /     \
  #  /       \
  # o         o  --->  o-------------o-------------o
  # |----h----|        |------h1-----|------h2-----|
  if len(v) == 3:
   h1 = np.sqrt( ((_self.X[v[0]]-_self.X[v[1]])**2) + \
                 ((_self.Y[v[0]]-_self.Y[v[1]])**2) )
   h2 = np.sqrt( ((_self.X[v[1]]-_self.X[v[2]])**2) + \
                 ((_self.Y[v[1]]-_self.Y[v[2]])**2) )
   xi = [0.0,h1,h1+h2]

  _self.localx = np.zeros((_self.NUMRULE), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    _self.localx[k] += xi[i] * _self.phiJ[k][i]

  dxdxi = np.zeros((_self.NUMRULE),dtype='float')
  _self.jacob = np.zeros((_self.NUMRULE,1),dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dxdxi[k]  +=  xi[i] * _self.dphiJdxi[k][i]
   # compute jacobian
   _self.jacob[k] = dxdxi[k]

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = _self.dphiJdxi[k][i]/_self.jacob[k];

  _self.mass = np.dot(    _self.phiJ.T,_self.phiJ*abs(_self.jacob)*_self.gqWeights)
  _self.kxx  = np.dot(       dphiJdx.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.gvx  = np.dot(    _self.phiJ.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.gx   = np.dot(       dphiJdx.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)

   # compute area
  _self.length = (_self.jacob*_self.gqWeights).sum()
  _self.xc = _self.localx.sum()/_self.NUMRULE

# This class is aimed to transform coordinates in 2D. This applies to
# triangles and quadrilaterals in the same way.
#
# The gaussian points are used to compute the point jacobian (area
# associated to the transformation) and can be further used in
# isoparametric elements with curved boundary.
#
# The kxx,kyy,mass,gx,gy,dx and dy element matrices are computed based on
# the chosen element: triangle or quadrilateral.
#
#                                   ^
#      eta ^                   eta  |
#          |                    o-------o
#          o                    |       |
#          |`.                  |       |--->
#          |  `.                |       |   xi
#        --o----o--->           o-------o
#                    xi
#
class Element2D:
 def __init__(_self,_X,_Y):

  _self.X = _X
  _self.Y = _Y

  _self.area    = 0.0
  _self.jacob   = 0.0

  _self.local_t = 0.0
  _self.jacob_t = 0.0
  _self.dphi_t  = 0.0
  _self.mat_t   = 0.0

 def getMatrices(_self,v):
  # compute the coordinate (x,y) of the gauss points in the mesh element
  # (from the referential to the mesh element), therefore localx,localy are
  # the coordinates of all gauss points (NUMRULE) in the mesh element.
  # Additionaly, NUMGLEU is used, i.e. the element is ISOPARAMETRIC since
  # its geometry and the interpolation of variables are in the same space
  # (NUMGLEU).
  _self.localx = np.zeros((_self.NUMRULE), dtype='float')
  _self.localy = np.zeros((_self.NUMRULE), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    _self.localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    _self.localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  # compute the derivatives (d/dxi, d/deta) at each gauss point from the
  # referential element to the mesh elements, therefore
  # dxdxi,dxdeta,dydxi,dydeta are the derivatives of all gauss points
  # (NUMRULE) in the mesh element
  dxdxi = np.zeros((_self.NUMRULE),dtype='float')
  dxdeta = np.zeros((_self.NUMRULE),dtype='float')
  dydxi = np.zeros((_self.NUMRULE),dtype='float')
  dydeta = np.zeros((_self.NUMRULE),dtype='float')
  _self.jacob = np.zeros((_self.NUMRULE,1),dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dxdxi[k]  +=  _self.X[v[i]] * _self.dphiJdxi[k][i]
    dxdeta[k] +=  _self.X[v[i]] * _self.dphiJdeta[k][i]
    dydxi[k]  +=  _self.Y[v[i]] * _self.dphiJdxi[k][i]
    dydeta[k] +=  _self.Y[v[i]] * _self.dphiJdeta[k][i]
   # compute det(jacobian) for J^-1
   _self.jacob[k] = dxdxi[k]*dydeta[k] - dxdeta[k]*dydxi[k]

  # compute dNi/dx, dNi/dy for all gauss points. This is done using the
  # chain rule as follows (consider Ni=phiJ):
  #
  # dNi/dxi  = (dNi/dx * dx/dxi)  + (dNi/dy * dy/dxi)
  # dNi/deta = (dNi/dx * dx/deta) + (dNi/dy * dy/deta)
  #
  # J = | dx/dxi  dy/dxi  |   Obs.: note here that J is usually written
  #     | dx/deta dy/deta |         transposed in math books.
  #
  # | dNi/dxi  | = J | dNi/dx |
  # | dNi/deta |     | dNi/dy |
  #
  # J^-1 = 1 / (detJ) * |  dy/deta -dy/dxi |
  #                     | -dx/deta  dx/dxi |
  #
  # detJ = (dx/dxi)*(dy/deta) - (dx/deta)*(dy/dxi)
  #
  # | dNi_dx | = J^-1 | dNi/dxi  |
  # | dNi_dy |        | dNi/deta |
  #
  # Obs.: dNi_dx = dphiJdx, dNi_dy = dphiJdy
  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = ( _self.dphiJdxi[k][i]*dydeta[k]-_self.dphiJdeta[k][i]\
                  *dydxi[k] )/_self.jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdxi[k][i]*dxdeta[k]+_self.dphiJdeta[k][i]\
                  *dxdxi[k] )/_self.jacob[k];

  _self.mass = np.dot(    _self.phiJ.T,_self.phiJ*abs(_self.jacob)*_self.gqWeights)
  _self.kxx  = np.dot(       dphiJdx.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.kyx  = np.dot(       dphiJdx.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.kxy  = np.dot(       dphiJdy.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.kyy  = np.dot(       dphiJdy.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.gvx  = np.dot(    _self.phiJ.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.gvy  = np.dot(    _self.phiJ.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.gx   = np.dot(       dphiJdx.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)
  _self.gy   = np.dot(       dphiJdy.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)
  _self.dx   = np.dot(_self.gqPoints.T,       dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.dy   = np.dot(_self.gqPoints.T,       dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.dmass= np.dot(_self.gqPoints.T,    _self.phiJ*abs(_self.jacob)*_self.gqWeights)

   # compute area
  _self.area = (_self.jacob*_self.gqWeights).sum()
  _self.xc = _self.localx.sum()/_self.NUMRULE
  _self.yc = _self.localy.sum()/_self.NUMRULE

 def getMatricesProp(_self,v,mu,rho):
  # pontos de gauss em coordenada fisica
  _self.localx = np.zeros((_self.NUMRULE), dtype='float')
  _self.localy = np.zeros((_self.NUMRULE), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    _self.localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    _self.localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]

  # interpolacao de mu,rho
  _self.localmu = np.zeros((_self.NUMRULE), dtype='float')
  _self.localrho = np.zeros((_self.NUMRULE), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    _self.localmu[k] += mu[v[i]] * _self.phiJ[k][i]
    _self.localrho[k] += rho[v[i]] * _self.phiJ[k][i]

  dxdxi = np.zeros((_self.NUMRULE),dtype='float')
  dxdeta = np.zeros((_self.NUMRULE),dtype='float')
  dydxi = np.zeros((_self.NUMRULE),dtype='float')
  dydeta = np.zeros((_self.NUMRULE),dtype='float')
  _self.jacob = np.zeros((_self.NUMRULE,1),dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dxdxi[k]  +=  _self.X[v[i]] * _self.dphiJdxi[k][i]
    dxdeta[k] +=  _self.X[v[i]] * _self.dphiJdeta[k][i]
    dydxi[k]  +=  _self.Y[v[i]] * _self.dphiJdxi[k][i]
    dydeta[k] +=  _self.Y[v[i]] * _self.dphiJdeta[k][i]
   # compute det(jacobian)
   _self.jacob[k] = dxdxi[k]*dydeta[k] - dxdeta[k]*dydxi[k]

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = ( _self.dphiJdxi[k][i]*dydeta[k]-_self.dphiJdeta[k][i]\
                  *dydxi[k] )/_self.jacob[k];
    dphiJdy[k][i] = ( -_self.dphiJdxi[k][i]*dxdeta[k]+_self.dphiJdeta[k][i]\
                  *dxdxi[k] )/_self.jacob[k];

  _self.mass = np.dot(_self.phiJ.T,_self.phiJ*abs(_self.jacob)*_self.gqWeights)
  _self.kxx  = np.dot(       dphiJdx.T, dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.kyx  = np.dot(       dphiJdx.T, dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.kxy  = np.dot(       dphiJdy.T, dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.kyy  = np.dot(       dphiJdy.T, dphiJdy*abs(_self.jacob)*_self.gqWeights)
  #_self.gvx  = np.dot(    _self.phiJ.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  #_self.gvy  = np.dot(    _self.phiJ.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.gx   = np.dot(       dphiJdx.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)
  _self.gy   = np.dot(       dphiJdy.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)
  _self.dx   = np.dot(_self.gqPoints.T,       dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.dy   = np.dot(_self.gqPoints.T,       dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.dmass= np.dot(_self.gqPoints.T,    _self.phiJ*abs(_self.jacob)*_self.gqWeights)

   # compute area
  _self.area = (_self.jacob*_self.gqWeights).sum()
  _self.xc = _self.localx.sum()/_self.NUMRULE
  _self.yc = _self.localy.sum()/_self.NUMRULE

 def getMatricesSlip(_self,v):
  _self.getMatrices(v)

  # gqPoints
  localxlin = [0.0]*_self.NUMRULE
  localylin = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    localxlin[k] += _self.X[v[i]] * _self.gqPoints[k][i]
    localylin[k] += _self.Y[v[i]] * _self.gqPoints[k][i]

  dxdxilin = np.zeros((_self.NUMRULE),dtype='float')
  dxdetalin = np.zeros((_self.NUMRULE),dtype='float')
  dydxilin = np.zeros((_self.NUMRULE),dtype='float')
  dydetalin = np.zeros((_self.NUMRULE),dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    dxdxilin[k]  +=  _self.X[v[i]] * _self.dgqPointsdxi[k][i]
    dxdetalin[k] +=  _self.X[v[i]] * _self.dgqPointsdeta[k][i]
    dydxilin[k]  +=  _self.Y[v[i]] * _self.dgqPointsdxi[k][i]
    dydetalin[k] +=  _self.Y[v[i]] * _self.dgqPointsdeta[k][i]

  dgqPointsdx = np.zeros((_self.NUMRULE,_self.NUMGLEP), dtype='float')
  dgqPointsdy = np.zeros((_self.NUMRULE,_self.NUMGLEP), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    dgqPointsdx[k][i] = ( _self.dgqPointsdxi[k][i]*dydetalin[k]-_self.dgqPointsdeta[k][i]\
                  *dydxilin[k] )/_self.jacob[k];
    dgqPointsdy[k][i] = (-_self.dgqPointsdxi[k][i]*dxdetalin[k]+_self.dgqPointsdeta[k][i]\
                  *dxdxilin[k] )/_self.jacob[k];

  # update gx,gy (slip)
  _self.gx   = np.dot(    _self.phiJ.T,dgqPointsdx*abs(_self.jacob)*_self.gqWeights)
  _self.gy   = np.dot(    _self.phiJ.T,dgqPointsdy*abs(_self.jacob)*_self.gqWeights)

# This class is aimed to transform coordinates in 3D. This applies to
# triangles and quadrilaterals in the same way.
#
# The gaussian points are used to compute the point jacobian (area
# associated to the transformation) and can be further used in
# isoparametric elements with curved boundary.
#
# The kxx,kyy,mass,gx,gy,dx and dy element matrices are computed based on
# the chosen element: tetrahedron or prismatic.
class Element3D:
 def __init__(_self,_X,_Y,_Z):

  _self.X = _X
  _self.Y = _Y
  _self.Z = _Z

  _self.volume  = 0.0
  _self.jacob   = 0.0

  _self.local_t = 0.0
  _self.jacob_t = 0.0
  _self.dphi_t  = 0.0
  _self.mat_t   = 0.0

 def getMatrices(_self,v):
  #_self.jacob =-( (_self.X[v[1]]-_self.X[v[0]])\
  #        *((_self.Y[v[2]]-_self.Y[v[0]])
  #         *(_self.Z[v[3]]-_self.Z[v[0]])\
  #         -(_self.Y[v[3]]-_self.Y[v[0]])\
  #         *(_self.Z[v[2]]-_self.Z[v[0]]))\
  #         -(_self.X[v[2]]-_self.X[v[0]])
  #         *((_self.Y[v[1]]-_self.Y[v[0]])\
  #         *(_self.Z[v[3]]-_self.Z[v[0]])
  #         -(_self.Y[v[3]]-_self.Y[v[0]])\
  #         *(_self.Z[v[1]]-_self.Z[v[0]]))\
  #         +(_self.X[v[3]]-_self.X[v[0]])\
  #         *((_self.Y[v[1]]-_self.Y[v[0]])
  #         *(_self.Z[v[2]]-_self.Z[v[0]])\
  #         -(_self.Y[v[2]]-_self.Y[v[0]])\
  #         *(_self.Z[v[1]]-_self.Z[v[0]])) )

  localx = np.zeros((_self.NUMRULE), dtype='float')
  localy = np.zeros((_self.NUMRULE), dtype='float')
  localz = np.zeros((_self.NUMRULE), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    localx[k] += _self.X[v[i]] * _self.phiJ[k][i]
    localy[k] += _self.Y[v[i]] * _self.phiJ[k][i]
    localz[k] += _self.Z[v[i]] * _self.phiJ[k][i]

  dxdxi = np.zeros((_self.NUMRULE), dtype='float')
  dxdeta = np.zeros((_self.NUMRULE), dtype='float')
  dxdzeta = np.zeros((_self.NUMRULE), dtype='float')
  dydxi = np.zeros((_self.NUMRULE), dtype='float')
  dydeta = np.zeros((_self.NUMRULE), dtype='float')
  dydzeta = np.zeros((_self.NUMRULE), dtype='float')
  dzdxi = np.zeros((_self.NUMRULE), dtype='float')
  dzdeta = np.zeros((_self.NUMRULE), dtype='float')
  dzdzeta = np.zeros((_self.NUMRULE), dtype='float')
  _self.jacob = np.zeros((_self.NUMRULE,1),dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dxdxi[k]   += _self.X[v[i]]*_self.dphiJdxi[k][i]
    dxdeta[k]  += _self.X[v[i]]*_self.dphiJdeta[k][i]
    dxdzeta[k] += _self.X[v[i]]*_self.dphiJdzeta[k][i]
    dydxi[k]   += _self.Y[v[i]]*_self.dphiJdxi[k][i]
    dydeta[k]  += _self.Y[v[i]]*_self.dphiJdeta[k][i]
    dydzeta[k] += _self.Y[v[i]]*_self.dphiJdzeta[k][i]
    dzdxi[k]   += _self.Z[v[i]]*_self.dphiJdxi[k][i]
    dzdeta[k]  += _self.Z[v[i]]*_self.dphiJdeta[k][i]
    dzdzeta[k] += _self.Z[v[i]]*_self.dphiJdzeta[k][i]
   # compute det(jacobian)
   _self.jacob[k] =  +dxdxi[k]*dydeta[k]*dzdzeta[k] \
                     +dydxi[k]*dzdeta[k]*dxdzeta[k] \
                     +dzdxi[k]*dxdeta[k]*dydzeta[k] \
                     -dydxi[k]*dxdeta[k]*dzdzeta[k] \
                     -dxdxi[k]*dzdeta[k]*dydzeta[k] \
                     -dzdxi[k]*dydeta[k]*dxdzeta[k]

  dphiJdx = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  dphiJdy = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  dphiJdz = np.zeros((_self.NUMRULE,_self.NUMGLEU), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEU):
    dphiJdx[k][i] = ( _self.dphiJdxi[k][i]\
                      *(dydeta[k]*dzdzeta[k]-dydzeta[k]*dzdeta[k]) -\
                      _self.dphiJdeta[k][i]\
                      *(dydxi[k]*dzdzeta[k]-dydzeta[k]*dzdxi[k]) +\
                      _self.dphiJdzeta[k][i]\
                      *(dydxi[k]*dzdeta[k]-dydeta[k]*dzdxi[k]))/_self.jacob[k]

    dphiJdy[k][i] = ( _self.dphiJdxi[k][i]\
                      *(dzdeta[k]*dxdzeta[k]-dzdzeta[k]*dxdeta[k]) -\
                      _self.dphiJdeta[k][i]\
                      *(dzdxi[k]*dxdzeta[k]-dzdzeta[k]*dxdxi[k]) +\
                      _self.dphiJdzeta[k][i]\
                      *(dzdxi[k]*dxdeta[k]-dzdeta[k]*dxdxi[k]))/_self.jacob[k]

    dphiJdz[k][i] = ( _self.dphiJdxi[k][i]\
                      *(dxdeta[k]*dydzeta[k]-dxdzeta[k]*dydeta[k]) -\
                      _self.dphiJdeta[k][i]\
                      *(dxdxi[k]*dydzeta[k]-dxdzeta[k]*dydxi[k]) +\
                     _self.dphiJdzeta[k][i]\
                     *(dxdxi[k]*dydeta[k]-dxdeta[k]*dydxi[k]))/_self.jacob[k]

  _self.mass = np.dot(    _self.phiJ.T,_self.phiJ*abs(_self.jacob)*_self.gqWeights)
  _self.kxx  = np.dot(       dphiJdx.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.kxy  = np.dot(       dphiJdx.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.kxz  = np.dot(       dphiJdx.T,   dphiJdz*abs(_self.jacob)*_self.gqWeights)
  _self.kyx  = np.dot(       dphiJdy.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.kyy  = np.dot(       dphiJdy.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.kyz  = np.dot(       dphiJdy.T,   dphiJdz*abs(_self.jacob)*_self.gqWeights)
  _self.kzx  = np.dot(       dphiJdz.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.kzy  = np.dot(       dphiJdz.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.kzz  = np.dot(       dphiJdz.T,   dphiJdz*abs(_self.jacob)*_self.gqWeights)
  #_self.gvx  = np.dot(    _self.phiJ.T,   dphiJdx*abs(_self.jacob)*_self.gqWeights)
  #_self.gvy  = np.dot(    _self.phiJ.T,   dphiJdy*abs(_self.jacob)*_self.gqWeights)
  #_self.gvz  = np.dot(    _self.phiJ.T,   dphiJdz*abs(_self.jacob)*_self.gqWeights)
  _self.gx   = np.dot(       dphiJdx.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)
  _self.gy   = np.dot(       dphiJdy.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)
  _self.gz   = np.dot(       dphiJdz.T,_self.gqPoints*abs(_self.jacob)*_self.gqWeights)
  _self.dx   = np.dot(_self.gqPoints.T,       dphiJdx*abs(_self.jacob)*_self.gqWeights)
  _self.dy   = np.dot(_self.gqPoints.T,       dphiJdy*abs(_self.jacob)*_self.gqWeights)
  _self.dz   = np.dot(_self.gqPoints.T,       dphiJdz*abs(_self.jacob)*_self.gqWeights)

 def getMatricesSlip(_self,v):
  _self.getMatrices(v)

  # gqPoints
  localxlin = [0.0]*_self.NUMRULE
  localylin = [0.0]*_self.NUMRULE
  localzlin = [0.0]*_self.NUMRULE
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    localxlin[k] += _self.X[v[i]] * _self.gqPoints[k][i]
    localylin[k] += _self.Y[v[i]] * _self.gqPoints[k][i]
    localzlin[k] += _self.Z[v[i]] * _self.gqPoints[k][i]

  dxdxilin = np.zeros((_self.NUMRULE),dtype='float')
  dxdetalin = np.zeros((_self.NUMRULE),dtype='float')
  dxdzetalin = np.zeros((_self.NUMRULE),dtype='float')
  dydxilin = np.zeros((_self.NUMRULE),dtype='float')
  dydetalin = np.zeros((_self.NUMRULE),dtype='float')
  dydzetalin = np.zeros((_self.NUMRULE),dtype='float')
  dzdxilin = np.zeros((_self.NUMRULE),dtype='float')
  dzdetalin = np.zeros((_self.NUMRULE),dtype='float')
  dzdzetalin = np.zeros((_self.NUMRULE),dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    dxdxilin[k]   +=  _self.X[v[i]] * _self.dgqPointsdxi[k][i]
    dxdetalin[k]  +=  _self.X[v[i]] * _self.dgqPointsdeta[k][i]
    dxdzetalin[k] +=  _self.X[v[i]] * _self.dgqPointsdzeta[k][i]
    dydxilin[k]   +=  _self.Y[v[i]] * _self.dgqPointsdxi[k][i]
    dydetalin[k]  +=  _self.Y[v[i]] * _self.dgqPointsdeta[k][i]
    dydzetalin[k] +=  _self.Y[v[i]] * _self.dgqPointsdzeta[k][i]
    dzdxilin[k]   +=  _self.Z[v[i]] * _self.dgqPointsdxi[k][i]
    dzdetalin[k]  +=  _self.Z[v[i]] * _self.dgqPointsdeta[k][i]
    dzdzetalin[k] +=  _self.Z[v[i]] * _self.dgqPointsdzeta[k][i]

  dgqPointsdx = np.zeros((_self.NUMRULE,_self.NUMGLEP), dtype='float')
  dgqPointsdy = np.zeros((_self.NUMRULE,_self.NUMGLEP), dtype='float')
  dgqPointsdz = np.zeros((_self.NUMRULE,_self.NUMGLEP), dtype='float')
  for k in range(0,_self.NUMRULE):
   for i in range(0,_self.NUMGLEP):
    dgqPointsdx[k][i] = (_self.dgqPointsdxi[k][i]*(dydetalin[k]*dzdzetalin[k]-\
                         dydzetalin[k]*dzdetalin[k])-_self.dgqPointsdeta[k][i]*\
                        (dydxilin[k]*dzdzetalin[k]-dydzetalin[k]*dzdxilin[k])+\
                         _self.dgqPointsdzeta[k][i]*(dydxilin[k]*dzdetalin[k]-\
                         dydetalin[k]*dzdxilin[k]))/_self.jacob[k]
    dgqPointsdy[k][i] = (_self.dgqPointsdxi[k][i]*(dzdetalin[k]*dxdzetalin[k]-\
                         dzdzetalin[k]*dxdetalin[k])-_self.dgqPointsdeta[k][i]*\
                        (dzdxilin[k]*dxdzetalin[k]-dzdzetalin[k]*dxdxilin[k])+\
                         _self.dgqPointsdzeta[k][i]*(dzdxilin[k]*dxdetalin[k]-\
                         dzdetalin[k]*dxdxilin[k]))/_self.jacob[k]
    dgqPointsdz[k][i] = (_self.dgqPointsdxi[k][i]*(dxdetalin[k]*dydzetalin[k]-\
                         dxdzetalin[k]*dydetalin[k])-_self.dgqPointsdeta[k][i]*\
                        (dxdxilin[k]*dydzetalin[k]-dxdzetalin[k]*dydxilin[k])+\
                         _self.dgqPointsdzeta[k][i]*(dxdxilin[k]*dydetalin[k]-\
                         dxdetalin[k]*dydxilin[k]))/_self.jacob[k]

  # update gx,gy (slip)
  _self.gx   = np.dot(    _self.phiJ.T,dgqPointsdx*abs(_self.jacob)*_self.gqWeights)
  _self.gy   = np.dot(    _self.phiJ.T,dgqPointsdy*abs(_self.jacob)*_self.gqWeights)
  _self.gz   = np.dot(    _self.phiJ.T,dgqPointsdz*abs(_self.jacob)*_self.gqWeights)

# This class sets the Gaussian points to the following line elements:
#
#   - lin2  (linear)
#   - lin3  (quadratic)
#
class Line(Element1D):
 def __init__(_self,_X,_Y,_NUMGLEU,_NUMGLEP,_NUMRULE):
  super().__init__(_X,_Y)

  _self.NUMRULE = _NUMRULE
  _self.NUMGLEU = _NUMGLEU
  _self.NUMGLEP = _NUMGLEP

  if _self.NUMGLEU == 2:
   _self.lin2()
  else:
   _self.lin3()

 def lin2(_self):
  # L1 e L2 de todos os pontos de Gauss
  _self.gqPoints =np.array( [ [0.78867513, 0.21132487],
                              [0.21132487, 0.78867513] ] )

  _self.gqWeights =np.array( [ [1],
                               [1] ] )

  _self.phiJ = np.array( [ [0.78867513, 0.21132487],
                           [0.21132487, 0.78867513] ] )

  _self.dphiJdxi = np.array( [ [-0.5, 0.5],
                               [-0.5, 0.5] ] )

 def lin3(_self):
  # L1 e L2 de todos os pontos de Gauss
  _self.gqPoints = np.array( [ [0.93056816, 0.06943184],
                               [0.66999052, 0.33000948],
                               [0.33000948, 0.66999052],
                               [0.06943184, 0.93056816] ] )

  _self.gqWeights = np.array( [ [0.3478548451],
                                [0.6521451549],
                                [0.6521451549],
                                [0.3478548451] ] )

  _self.phiJ = np.array( [ [ 0.80134603,  0.25844425, -0.05979028],
                           [ 0.22778408,  0.88441289, -0.11219697],
                           [-0.11219697,  0.88441289,  0.22778408],
                           [-0.05979028,  0.25844425,  0.80134603]] )

  _self.dphiJdxi = np.array( [ [-1.36113631,  1.72227262, -0.36113631],
                               [-0.83998104,  0.67996209,  0.16001896],
                               [-0.16001896, -0.67996209,  0.83998104],
                               [ 0.36113631, -1.72227262,  1.36113631]] )


# This class sets the Gaussian points to the following triangle elements:
#
#   - tri3  (linear)
#   - tri4  (mini)
#   - tri6  (quadratic)
#   - tri7  (quadratic+bubble)
#   - tri10 (cubic)
#
class Triangle(Element2D):
 def __init__(_self,_X,_Y,_NUMGLEU,_NUMGLEP,_NUMRULE):
  super().__init__(_X,_Y)

  _self.NUMRULE = _NUMRULE
  _self.NUMGLEU = _NUMGLEU
  _self.NUMGLEP = _NUMGLEP

  if _self.NUMGLEU == 3:
   _self.tri3()
  elif _self.NUMGLEU == 4:
   _self.tri4()
  elif _self.NUMGLEU == 6:
   _self.tri6()
  elif _self.NUMGLEU == 7:
   _self.tri7()
  else:
   _self.tri10()

 def tri3(_self):
  # tri6, tri7 e tri10 estao baseados nessa organizacao dos pontos de Gauss
  # L1, L2 e L3 de todos os pontos de Gauss
  _self.gqPoints=np.array([ [0.249286745171, 0.249286745171, 0.501426509658 ],
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
                            [0.053145049845, 0.636502499121, 0.310352451034 ] ])

  _self.gqWeights = np.array([ [0.116786275726],
                               [0.116786275726],
                               [0.116786275726],
                               [0.050844906370],
                               [0.050844906370],
                               [0.050844906370],
                               [0.082851075618],
                               [0.082851075618],
                               [0.082851075618],
                               [0.082851075618],
                               [0.082851075618],
                               [0.082851075618] ])

  # Essa organizacao dos pontos de Gauss esta no codigo femSIM2d em C++ e
  # NAO esta de acordo com as implementacoes do tri6, tri7 e tri10 deste
  # arquivo
#--------------------------------------------------
#   _self.gqPoints=np.array([ [0.873821971016996, 0.063089014491502, 0.063089014491502],
#                             [0.063089014491502, 0.873821971016996, 0.063089014491502],
#                             [0.063089014491502, 0.063089014491502, 0.873821971016996],
#                             [0.501426509658179, 0.249286745170910, 0.249286745170911],
#                             [0.249286745170910, 0.501426509658179, 0.249286745170911],
#                             [0.249286745170910, 0.249286745170910, 0.50142650965818 ],
#                             [0.636502499121399, 0.310352451033785, 0.053145049844816],
#                             [0.636502499121399, 0.053145049844816, 0.310352451033785],
#                             [0.310352451033785, 0.636502499121399, 0.053145049844816],
#                             [0.310352451033785, 0.053145049844816, 0.636502499121399],
#                             [0.053145049844816, 0.636502499121399, 0.310352451033785],
#                             [0.053145049844816, 0.310352451033785, 0.636502499121399] ])
#
#   _self.gqWeights = np.array([ [0.050844906370207],
#                                [0.050844906370207],
#                                [0.050844906370207],
#                                [0.116786275726379],
#                                [0.116786275726379],
#                                [0.116786275726379],
#                                [0.082851075618374],
#                                [0.082851075618374],
#                                [0.082851075618374],
#                                [0.082851075618374],
#                                [0.082851075618374],
#                                [0.082851075618374] ])
#--------------------------------------------------

  # N1, N2 e N3
  _self.phiJ = np.array(_self.gqPoints)

  # dL1/dL1, dL2/dL1 e dL3/dL1
  _self.dgqPointsdxi = np.array([ [1, 0, -1],
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
                                  [1, 0, -1] ])

  # dL1/dL2, dL2/dL2 e dL3/dL2
  _self.dgqPointsdeta = np.array([ [0, 1, -1],
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
                                   [0, 1, -1] ])

  _self.dphiJdxi = np.array(_self.dgqPointsdxi)
  _self.dphiJdeta = np.array(_self.dgqPointsdeta)

 def tri4(_self):
  _self.tri3()

  # tri6, tri7 e tri10 estao baseados nessa organizacao das matrizes
  # N1, N2 N3 e N4 de todos os pontos de Gauss
  _self.phiJ = np.array([
[-0.031158560381739114,-0.031158560381739114,0.2209812041055309,0.8413359166579473],
[-0.031158560381739614,0.2209812041055294,-0.031158560381738643,0.8413359166579488],
[0.2209812041055294,-0.031158560381739614,-0.031158560381738643,0.8413359166579488],
[0.03178695183454089,0.03178695183454089,0.8425199083600349,0.09390618797088336],
[0.031786951834540916,0.8425199083600349,0.03178695183454086,0.09390618797088328],
[0.8425199083600349,0.031786951834540916,0.03178695183454086,0.09390618797088328],
[0.21586793977235394,0.5420179878599679,-0.041339461416615014,0.2834535337842932],
[0.542017987859968,-0.041339461416614966,0.21586793977235397,0.2834535337842929],
[-0.04133946141661498,0.21586793977235402,0.542017987859968,0.28345353378429294],
[0.5420179878599679,0.21586793977235394,-0.041339461416615014,0.28345353378429317],
[0.21586793977235402,-0.04133946141661498,0.542017987859968,0.283453533784293],
[-0.041339461416614966,0.542017987859968,0.21586793977235397,0.2834535337842929],
])

  _self.dphiJdxi = np.array([
[0.43430408904527806,-0.5656959109547219,-1.565695910954722,1.697087732864166],
[0.999999999999996,-3.9968028886505635e-15,-1.000000000000004,1.199040866595169e-14],
[1.5656959109547177,0.5656959109547177,-0.4343040890452823,-1.6970877328641532],
[0.5396649108132237,-0.46033508918677635,-1.4603350891867763,1.381005267560329],
[1.0,0.0,-1.0,0.0],
[1.4603350891867763,0.46033508918677646,-0.5396649108132235,-1.3810052675603293],
[2.4734183828436915,1.4734183828436915,0.47341838284369153,-4.420255148531075],
[1.1559993450625485,0.1559993450625485,-0.8440006549374515,-0.4679980351876455],
[-0.6294177279062401,-1.62941772790624,-2.6294177279062403,4.88825318371872],
[2.6294177279062403,1.62941772790624,0.6294177279062401,-4.88825318371872],
[0.8440006549374516,-0.15599934506254845,-1.1559993450625485,0.4679980351876453],
[-0.47341838284369153,-1.4734183828436915,-2.4734183828436915,4.420255148531075],
])


  _self.dphiJdeta = np.array([
[-0.5656959109547218,0.4343040890452782,-1.5656959109547217,1.6970877328641656],
[0.5656959109547178,1.5656959109547177,-0.43430408904528217,-1.6970877328641536],
[-4.246603069191224e-15,0.9999999999999958,-1.0000000000000042,1.2739809207573671e-14],
[-0.46033508918677635,0.5396649108132237,-1.4603350891867763,1.381005267560329],
[0.46033508918677646,1.4603350891867763,-0.5396649108132235,-1.3810052675603293],
[3.7470027081099033e-16,1.0000000000000004,-0.9999999999999997,-1.124100812432971e-15],
[1.62941772790624,2.6294177279062403,0.6294177279062401,-4.88825318371872],
[-1.4734183828436915,-0.47341838284369153,-2.4734183828436915,4.420255148531075],
[-0.15599934506254848,0.8440006549374515,-1.1559993450625485,0.46799803518764543],
[1.4734183828436915,2.4734183828436915,0.47341838284369153,-4.420255148531075],
[-1.62941772790624,-0.6294177279062401,-2.6294177279062403,4.88825318371872],
[0.15599934506254848,1.1559993450625485,-0.8440006549374515,-0.46799803518764543],
])

  # Essa organizacao das matrizes esta no codigo femSIM2d em C++ e
  # NAO esta de acordo com as implementacoes do tri6, tri7 e tri10 deste
  # arquivo
#--------------------------------------------------
#   _self.phiJ = np.array([
# [0.8425199083600349,0.031786951834540916,0.03178695183454086,0.09390618797088328],
# [0.031786951834540916,0.8425199083600349,0.03178695183454086,0.09390618797088328],
# [0.03178695183454089,0.03178695183454089,0.8425199083600349,0.09390618797088336],
# [0.2209812041055294,-0.031158560381739614,-0.031158560381738643,0.8413359166579488],
# [-0.031158560381739614,0.2209812041055294,-0.031158560381738643,0.8413359166579488],
# [-0.031158560381739114,-0.031158560381739114,0.2209812041055309,0.8413359166579473],
# [0.5420179878599679,0.21586793977235394,-0.041339461416615014,0.28345353378429317],
# [0.542017987859968,-0.041339461416614966,0.21586793977235397,0.2834535337842929],
# [0.21586793977235394,0.5420179878599679,-0.041339461416615014,0.2834535337842932],
# [0.21586793977235402,-0.04133946141661498,0.542017987859968,0.283453533784293],
# [-0.041339461416614966,0.542017987859968,0.21586793977235397,0.2834535337842929],
# [-0.04133946141661498,0.21586793977235402,0.542017987859968,0.28345353378429294],
# ])
#
#   _self.dphiJdxi = np.array([
# [1.4603350891867763,0.46033508918677646,-0.5396649108132235,-1.3810052675603293],
# [1.0,0.0,-1.0,0.0],
# [0.5396649108132237,-0.46033508918677635,-1.4603350891867763,1.381005267560329],
# [1.5656959109547177,0.5656959109547177,-0.4343040890452823,-1.6970877328641532],
# [0.999999999999996,-3.9968028886505635e-15,-1.000000000000004,1.199040866595169e-14],
# [0.43430408904527806,-0.5656959109547219,-1.565695910954722,1.697087732864166],
# [2.6294177279062403,1.62941772790624,0.6294177279062401,-4.88825318371872],
# [1.1559993450625485,0.1559993450625485,-0.8440006549374515,-0.4679980351876455],
# [2.4734183828436915,1.4734183828436915,0.47341838284369153,-4.420255148531075],
# [0.8440006549374516,-0.15599934506254845,-1.1559993450625485,0.4679980351876453],
# [-0.47341838284369153,-1.4734183828436915,-2.4734183828436915,4.420255148531075],
# [-0.6294177279062401,-1.62941772790624,-2.6294177279062403,4.88825318371872],
# ])
#
#   _self.dphiJdeta = np.array([
# [3.7470027081099033e-16,1.0000000000000004,-0.9999999999999997,-1.124100812432971e-15],
# [0.46033508918677646,1.4603350891867763,-0.5396649108132235,-1.3810052675603293],
# [-0.46033508918677635,0.5396649108132237,-1.4603350891867763,1.381005267560329],
# [-4.246603069191224e-15,0.9999999999999958,-1.0000000000000042,1.2739809207573671e-14],
# [0.5656959109547178,1.5656959109547177,-0.43430408904528217,-1.6970877328641536],
# [-0.5656959109547218,0.4343040890452782,-1.5656959109547217,1.6970877328641656],
# [1.4734183828436915,2.4734183828436915,0.47341838284369153,-4.420255148531075],
# [-1.4734183828436915,-0.47341838284369153,-2.4734183828436915,4.420255148531075],
# [1.62941772790624,2.6294177279062403,0.6294177279062401,-4.88825318371872],
# [-1.62941772790624,-0.6294177279062401,-2.6294177279062403,4.88825318371872],
# [0.15599934506254848,1.1559993450625485,-0.8440006549374515,-0.46799803518764543],
# [-0.15599934506254848,0.8440006549374515,-1.1559993450625485,0.46799803518764543],
# ])
#--------------------------------------------------

 def tri6(_self):
  _self.tri3()

  _self.phiJ = np.array([
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
[-0.047496257199, 0.173768363654, -0.117715163308, 0.135307828169, 0.790160442766, 0.065974785919 ] ])



  _self.dphiJdxi = [
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


  _self.dphiJdeta = [
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


 def tri7(_self):
  _self.tri3()

  _self.phiJ = np.array([
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
[-0.016001420111631, 0.205263200740714, -0.086220326221172, 0.009328479819875, 0.664181094417758, -0.060004562429953, 0.283453533784409 ] ])


  _self.dphiJdxi = np.array([
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
[-0.296280339672388, 0.491139460948412, 0.249729656811612, 0.581452152690350, -4.510567840277649, -0.935728239036049, 4.420255148535712   ] ])

  _self.dphiJdeta = np.array([
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
[0.543139242635619, -0.244280557985181, -1.002870753849181, -0.931147166406477, 0.160872826563123, -3.413966774678477, 4.888253183720574  ], [-0.051999781687371, 1.494010214796629, -0.293409585824171, 0.420579326128683, -1.096601065597717, -0.004581072629717, -0.467998035186336 ] ])


 def tri10(_self):
  _self.tri3()

  _self.phiJ = np.array([
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
[0.041110728466, -0.026193226600, 0.011435826065, -0.127951879895, 0.138446419693, 0.808488952668, -0.061285221448, -0.005117035916, -0.062388096818, 0.283453533784 ] ])


  _self.dphiJdxi = np.array([
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
[0.559823901757, 0.000000000000, 0.492870367158, -1.950933405907, 2.605067077684, -2.605067077684, -2.469321742629, -0.302461418154, -0.750232850762, 4.420255148536] ])

  _self.dphiJdeta = np.array([
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
[0.000000000000, 0.740805831639, 0.492870367158, -0.201023373941, 0.674175115836, 1.331929881335, -2.565606080133, -0.206177080649, 0.201023373941, -0.467998035186  ] ])


# This class sets the Gaussian points to the following quadrilateral elements:
#
#   - quad4  (linear)
#   - quad5  (mini)
#   - quad8  (quadratic)
#   - quad9  (quadratic+bubble)
#
class Quadrilateral(Element2D):
 def __init__(_self,_X,_Y,_NUMGLEU,_NUMGLEP,_NUMRULE):
  super().__init__(_X,_Y)

  _self.NUMRULE = _NUMRULE
  _self.NUMGLEU = _NUMGLEU
  _self.NUMGLEP = _NUMGLEP

  if _self.NUMGLEU == 4:
   _self.quad4()
  elif _self.NUMGLEU == 5:
   _self.quad5()
  elif _self.NUMGLEU == 8:
   _self.quad8()
  else:
   _self.quad9()

 def quad4(_self):

  _self.gqPoints = np.array([
  [0.7872983346207417,0.09999999999999999,0.012701665379258308,0.09999999999999999],
  [0.44364916731037085,0.05635083268962915,0.05635083268962915,0.44364916731037085],
  [0.09999999999999999,0.012701665379258308,0.09999999999999999,0.7872983346207417],
  [0.44364916731037085,0.44364916731037085,0.05635083268962915,0.05635083268962915],
  [0.25,0.25,0.25,0.25],
  [0.05635083268962915,0.05635083268962915,0.44364916731037085,0.44364916731037085],
  [0.09999999999999999,0.7872983346207417,0.09999999999999999,0.012701665379258308],
  [0.05635083268962915,0.44364916731037085,0.44364916731037085,0.05635083268962915],
  [0.012701665379258308,0.09999999999999999,0.7872983346207417,0.09999999999999999]
  ])

  _self.gqWeights = np.array([
  [0.308641975308642],
  [0.49382716049382713],
  [0.308641975308642],
  [0.49382716049382713],
  [0.7901234567901234],
  [0.49382716049382713],
  [0.308641975308642],
  [0.49382716049382713],
  [0.308641975308642],
  ])

  _self.phiJ = np.array(_self.gqPoints)

  _self.dgqPointsdxi = np.array([
  [-0.44364916731037085,0.44364916731037085,0.05635083268962915,-0.05635083268962915],
  [-0.25,0.25,0.25,-0.25],
  [-0.05635083268962915,0.05635083268962915,0.44364916731037085,-0.44364916731037085],
  [-0.44364916731037085,0.44364916731037085,0.05635083268962915,-0.05635083268962915],
  [-0.25,0.25,0.25,-0.25],
  [-0.05635083268962915,0.05635083268962915,0.44364916731037085,-0.44364916731037085],
  [-0.44364916731037085,0.44364916731037085,0.05635083268962915,-0.05635083268962915],
  [-0.25,0.25,0.25,-0.25],
  [-0.05635083268962915,0.05635083268962915,0.44364916731037085,-0.44364916731037085]
  ])

  _self.dgqPointsdeta = np.array([
  [-0.44364916731037085,-0.05635083268962915,0.05635083268962915,0.44364916731037085],
  [-0.44364916731037085,-0.05635083268962915,0.05635083268962915,0.44364916731037085],
  [-0.44364916731037085,-0.05635083268962915,0.05635083268962915,0.44364916731037085],
  [-0.25,-0.25,0.25,0.25],
  [-0.25,-0.25,0.25,0.25],
  [-0.25,-0.25,0.25,0.25],
  [-0.05635083268962915,-0.44364916731037085,0.44364916731037085,0.05635083268962915],
  [-0.05635083268962915,-0.44364916731037085,0.44364916731037085,0.05635083268962915],
  [-0.05635083268962915,-0.44364916731037085,0.44364916731037085,0.05635083268962915]
  ])

  _self.dphiJdxi = np.array(_self.dgqPointsdxi)
  _self.dphiJdeta = np.array(_self.dgqPointsdeta)

 def quad5(_self):
  _self.quad4()

  _self.phiJ = np.array([
[0.7472983346207418,0.06000000000000001,-0.027298334620741674,0.06000000000000001,0.15999999999999992],
[0.3436491673103709,-0.04364916731037083,-0.04364916731037083,0.3436491673103709,0.3999999999999999],
[0.06000000000000001,-0.027298334620741674,0.06000000000000001,0.7472983346207418,0.15999999999999992],
[0.3436491673103709,0.3436491673103709,-0.04364916731037083,-0.04364916731037083,0.3999999999999999],
[0.0,0.0,0.0,0.0,1.0],
[-0.04364916731037083,-0.04364916731037083,0.3436491673103709,0.3436491673103709,0.3999999999999999],
[0.06000000000000001,0.7472983346207418,0.06000000000000001,-0.027298334620741674,0.15999999999999992],
[-0.04364916731037083,0.3436491673103709,0.3436491673103709,-0.04364916731037083,0.3999999999999999],
[-0.027298334620741674,0.06000000000000001,0.7472983346207418,0.06000000000000001,0.15999999999999992],
])

  _self.dphiJdxi = np.array([
[-0.5985685011586674,0.2887298334620742,-0.0985685011586675,-0.2112701665379258,0.6196773353931866],
[-0.6372983346207417,-0.1372983346207417,-0.1372983346207417,-0.6372983346207417,1.5491933384829668],
[-0.2112701665379258,-0.0985685011586675,0.2887298334620742,-0.5985685011586674,0.6196773353931866],
[-0.44364916731037085,0.44364916731037085,0.05635083268962915,-0.05635083268962915,-0.0],
[-0.25,0.25,0.25,-0.25,-0.0],
[-0.05635083268962915,0.05635083268962915,0.44364916731037085,-0.44364916731037085,-0.0],
[-0.2887298334620742,0.5985685011586674,0.2112701665379258,0.0985685011586675,-0.6196773353931866],
[0.1372983346207417,0.6372983346207417,0.6372983346207417,0.1372983346207417,-1.5491933384829668],
[0.0985685011586675,0.2112701665379258,0.5985685011586674,-0.2887298334620742,-0.6196773353931866],
])

  _self.dphiJdeta = np.array([
[-0.5985685011586674,-0.2112701665379258,-0.0985685011586675,0.2887298334620742,0.6196773353931866],
[-0.44364916731037085,-0.05635083268962915,0.05635083268962915,0.44364916731037085,-0.0],
[-0.2887298334620742,0.0985685011586675,0.2112701665379258,0.5985685011586674,-0.6196773353931866],
[-0.6372983346207417,-0.6372983346207417,-0.1372983346207417,-0.1372983346207417,1.5491933384829668],
[-0.25,-0.25,0.25,0.25,-0.0],
[0.1372983346207417,0.1372983346207417,0.6372983346207417,0.6372983346207417,-1.5491933384829668],
[-0.2112701665379258,-0.5985685011586674,0.2887298334620742,-0.0985685011586675,0.6196773353931866],
[-0.05635083268962915,-0.44364916731037085,0.44364916731037085,0.05635083268962915,-0.0],
[0.0985685011586675,-0.2887298334620742,0.5985685011586674,0.2112701665379258,-0.6196773353931866],
])

 def quad8(_self):
  _self.quad4()

  _self.phiJ = np.array([
[0.43237900077244507,-0.09999999999999999,-0.032379000772445,-0.09999999999999999,0.3549193338482966,0.04508066615170331,0.04508066615170331,0.3549193338482966],
[-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,0.19999999999999996,0.1127016653792583,0.19999999999999996,0.8872983346207417],
[-0.09999999999999999,-0.032379000772445,-0.09999999999999999,0.43237900077244507,0.04508066615170331,0.04508066615170331,0.3549193338482966,0.3549193338482966],
[-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,0.8872983346207417,0.19999999999999996,0.1127016653792583,0.19999999999999996],
[-0.25,-0.25,-0.25,-0.25,0.5,0.5,0.5,0.5],
[-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,0.1127016653792583,0.19999999999999996,0.8872983346207417,0.19999999999999996],
[-0.09999999999999999,0.43237900077244507,-0.09999999999999999,-0.032379000772445,0.3549193338482966,0.3549193338482966,0.04508066615170331,0.04508066615170331],
[-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,-0.09999999999999999,0.19999999999999996,0.8872983346207417,0.19999999999999996,0.1127016653792583],
[-0.032379000772445,-0.09999999999999999,0.43237900077244507,-0.09999999999999999,0.04508066615170331,0.3549193338482966,0.3549193338482966,0.04508066615170331],
])

  _self.dphiJdxi = np.array([
[-1.0309475019311125,-0.3436491673103709,-0.13094750193111251,-0.04364916731037084,1.3745966692414835,0.19999999999999996,0.17459666924148337,-0.19999999999999996],
[-0.3872983346207417,-0.3872983346207417,-0.3872983346207417,-0.3872983346207417,0.7745966692414834,0.5,0.7745966692414834,-0.5],
[-0.04364916731037084,-0.13094750193111251,-0.3436491673103709,-1.0309475019311125,0.17459666924148337,0.19999999999999996,1.3745966692414835,-0.19999999999999996],
[-0.3436491673103709,0.3436491673103709,-0.04364916731037084,0.04364916731037084,-0.0,0.19999999999999996,-0.0,-0.19999999999999996],
[0.0,0.0,0.0,0.0,-0.0,0.5,-0.0,-0.5],
[0.04364916731037084,-0.04364916731037084,0.3436491673103709,-0.3436491673103709,-0.0,0.19999999999999996,-0.0,-0.19999999999999996],
[0.3436491673103709,1.0309475019311125,0.04364916731037084,0.13094750193111251,-1.3745966692414835,0.19999999999999996,-0.17459666924148337,-0.19999999999999996],
[0.3872983346207417,0.3872983346207417,0.3872983346207417,0.3872983346207417,-0.7745966692414834,0.5,-0.7745966692414834,-0.5],
[0.13094750193111251,0.04364916731037084,1.0309475019311125,0.3436491673103709,-0.17459666924148337,0.19999999999999996,-1.3745966692414835,-0.19999999999999996],
])

  _self.dphiJdeta = np.array([
[-1.0309475019311125,-0.04364916731037084,-0.13094750193111251,-0.3436491673103709,-0.19999999999999996,0.17459666924148337,0.19999999999999996,1.3745966692414835],
[-0.3436491673103709,0.04364916731037084,-0.04364916731037084,0.3436491673103709,-0.19999999999999996,-0.0,0.19999999999999996,-0.0],
[0.3436491673103709,0.13094750193111251,0.04364916731037084,1.0309475019311125,-0.19999999999999996,-0.17459666924148337,0.19999999999999996,-1.3745966692414835],
[-0.3872983346207417,-0.3872983346207417,-0.3872983346207417,-0.3872983346207417,-0.5,0.7745966692414834,0.5,0.7745966692414834],
[0.0,0.0,0.0,0.0,-0.5,-0.0,0.5,-0.0],
[0.3872983346207417,0.3872983346207417,0.3872983346207417,0.3872983346207417,-0.5,-0.7745966692414834,0.5,-0.7745966692414834],
[-0.04364916731037084,-1.0309475019311125,-0.3436491673103709,-0.13094750193111251,-0.19999999999999996,1.3745966692414835,0.19999999999999996,0.17459666924148337],
[0.04364916731037084,-0.3436491673103709,0.3436491673103709,-0.04364916731037084,-0.19999999999999996,-0.0,0.19999999999999996,-0.0],
[0.13094750193111251,0.3436491673103709,1.0309475019311125,0.04364916731037084,-0.19999999999999996,-1.3745966692414835,0.19999999999999996,-0.17459666924148337],
])

 def quad9(_self):
  _self.quad4()
  _self.phiJ = np.array([
[0.4723790007724451,-0.059999999999999984,0.007620999227554982,-0.059999999999999984,0.27491933384829664,-0.03491933384829665,-0.03491933384829665,0.27491933384829664,0.15999999999999992],
[0.0,-0.0,-0.0,0.0,0.0,-0.08729833462074166,0.0,0.6872983346207417,0.3999999999999999],
[-0.059999999999999984,0.007620999227554982,-0.059999999999999984,0.4723790007724451,-0.03491933384829665,-0.03491933384829665,0.27491933384829664,0.27491933384829664,0.15999999999999992],
[0.0,0.0,-0.0,-0.0,0.6872983346207417,0.0,-0.08729833462074166,0.0,0.3999999999999999],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
[-0.0,-0.0,0.0,0.0,-0.08729833462074166,0.0,0.6872983346207417,0.0,0.3999999999999999],
[-0.059999999999999984,0.4723790007724451,-0.059999999999999984,0.007620999227554982,0.27491933384829664,0.27491933384829664,-0.03491933384829665,-0.03491933384829665,0.15999999999999992],
[-0.0,0.0,0.0,-0.0,0.0,0.6872983346207417,0.0,-0.08729833462074166,0.3999999999999999],
[0.007620999227554982,-0.059999999999999984,0.4723790007724451,-0.059999999999999984,-0.03491933384829665,0.27491933384829664,0.27491933384829664,-0.03491933384829665,0.15999999999999992],
])

  _self.dphiJdxi = np.array([
[-0.8760281680828159,-0.1887298334620742,0.023971831917184137,0.11127016653792579,1.06475800154489,-0.10983866769659334,-0.13524199845510992,-0.5098386676965933,0.6196773353931866],
[-0.0,-0.0,-0.0,-0.0,0.0,-0.2745966692414834,0.0,-1.2745966692414834,1.5491933384829668],
[0.11127016653792579,0.023971831917184137,-0.1887298334620742,-0.8760281680828159,-0.13524199845510992,-0.10983866769659334,1.06475800154489,-0.5098386676965933,0.6196773353931866],
[-0.3436491673103709,0.3436491673103709,-0.04364916731037083,0.04364916731037083,-0.0,0.19999999999999996,0.0,-0.19999999999999996,-0.0],
[-0.0,0.0,0.0,-0.0,-0.0,0.5,-0.0,-0.5,-0.0],
[0.04364916731037083,-0.04364916731037083,0.3436491673103709,-0.3436491673103709,0.0,0.19999999999999996,-0.0,-0.19999999999999996,-0.0],
[0.1887298334620742,0.8760281680828159,-0.11127016653792579,-0.023971831917184137,-1.06475800154489,0.5098386676965933,0.13524199845510992,0.10983866769659334,-0.6196773353931866],
[0.0,0.0,0.0,0.0,-0.0,1.2745966692414834,-0.0,0.2745966692414834,-1.5491933384829668],
[-0.023971831917184137,-0.11127016653792579,0.8760281680828159,0.1887298334620742,0.13524199845510992,0.5098386676965933,-1.06475800154489,0.10983866769659334,-0.6196773353931866],
])

  _self.dphiJdeta = np.array([
[-0.8760281680828159,0.11127016653792579,0.023971831917184137,-0.1887298334620742,-0.5098386676965933,-0.13524199845510992,-0.10983866769659334,1.06475800154489,0.6196773353931866],
[-0.3436491673103709,0.04364916731037083,-0.04364916731037083,0.3436491673103709,-0.19999999999999996,0.0,0.19999999999999996,-0.0,-0.0],
[0.1887298334620742,-0.023971831917184137,-0.11127016653792579,0.8760281680828159,0.10983866769659334,0.13524199845510992,0.5098386676965933,-1.06475800154489,-0.6196773353931866],
[-0.0,-0.0,-0.0,-0.0,-1.2745966692414834,0.0,-0.2745966692414834,0.0,1.5491933384829668],
[-0.0,-0.0,0.0,0.0,-0.5,-0.0,0.5,-0.0,-0.0],
[0.0,0.0,0.0,0.0,0.2745966692414834,-0.0,1.2745966692414834,-0.0,-1.5491933384829668],
[0.11127016653792579,-0.8760281680828159,-0.1887298334620742,0.023971831917184137,-0.5098386676965933,1.06475800154489,-0.10983866769659334,-0.13524199845510992,0.6196773353931866],
[0.04364916731037083,-0.3436491673103709,0.3436491673103709,-0.04364916731037083,-0.19999999999999996,-0.0,0.19999999999999996,0.0,-0.0],
[-0.023971831917184137,0.1887298334620742,0.8760281680828159,-0.11127016653792579,0.10983866769659334,-1.06475800154489,0.5098386676965933,0.13524199845510992,-0.6196773353931866],
])

class Tetrahedron(Element3D):
 def __init__(_self,_X,_Y,_Z,_NUMGLEU,_NUMGLEP,_NUMRULE):
  super().__init__(_X,_Y,_Z)

  _self.NUMRULE = _NUMRULE
  _self.NUMGLEU = _NUMGLEU
  _self.NUMGLEP = _NUMGLEP

  if _self.NUMGLEU == 4:
   _self.tet4()
  elif _self.NUMGLEU == 5:
   _self.tet5()
  elif _self.NUMGLEU == 10:
   _self.tet10()
  elif _self.NUMGLEU == 11:
   _self.tet11()
  else:
   _self.tet20()

 def tet4(_self):

  _self.gqPoints = np.array([
   [0.25,0.25,0.25,0.25],
   [0.617587190300083,0.127470936566639,0.127470936566639,0.127470936566639],
   [0.127470936566639,0.127470936566639,0.127470936566639,0.617587190300083],
   [0.127470936566639,0.127470936566639,0.617587190300083,0.127470936566639],
   [0.127470936566639,0.617587190300083,0.127470936566639,0.127470936566639],
   [0.9037635088221031,0.0320788303926323,0.0320788303926323,0.032078830392632374],
   [0.0320788303926323,0.0320788303926323,0.0320788303926323,0.9037635088221031],
   [0.0320788303926323,0.0320788303926323,0.9037635088221031,0.03207883039263226],
   [0.0320788303926323,0.9037635088221031,0.0320788303926323,0.032078830392632374],
   [0.450222904356719,0.049777095643281,0.049777095643281,0.45022290435671897],
   [0.049777095643281,0.450222904356719,0.049777095643281,0.45022290435671897],
   [0.049777095643281,0.049777095643281,0.450222904356719,0.45022290435671897],
   [0.049777095643281,0.450222904356719,0.450222904356719,0.04977709564328103],
   [0.450222904356719,0.049777095643281,0.450222904356719,0.04977709564328103],
   [0.450222904356719,0.450222904356719,0.049777095643281,0.04977709564328092],
   [0.3162695526014501,0.1837304473985499,0.1837304473985499,0.31626955260145007],
   [0.1837304473985499,0.3162695526014501,0.1837304473985499,0.31626955260145007],
   [0.1837304473985499,0.1837304473985499,0.3162695526014501,0.31626955260145007],
   [0.1837304473985499,0.3162695526014501,0.3162695526014501,0.18373044739854993],
   [0.3162695526014501,0.1837304473985499,0.3162695526014501,0.18373044739854993],
   [0.3162695526014501,0.3162695526014501,0.1837304473985499,0.18373044739854982],
   [0.0229177878448171,0.2319010893971509,0.2319010893971509,0.5132800333608811],
   [0.2319010893971509,0.0229177878448171,0.2319010893971509,0.5132800333608811],
   [0.2319010893971509,0.2319010893971509,0.0229177878448171,0.5132800333608811],
   [0.5132800333608811,0.2319010893971509,0.2319010893971509,0.022917787844817017],
   [0.2319010893971509,0.5132800333608811,0.2319010893971509,0.022917787844817017],
   [0.2319010893971509,0.2319010893971509,0.5132800333608811,0.022917787844817017],
   [0.2319010893971509,0.0229177878448171,0.5132800333608811,0.23190108939715093],
   [0.0229177878448171,0.5132800333608811,0.2319010893971509,0.23190108939715082],
   [0.5132800333608811,0.2319010893971509,0.0229177878448171,0.23190108939715082],
   [0.2319010893971509,0.5132800333608811,0.0229177878448171,0.23190108939715082],
   [0.0229177878448171,0.2319010893971509,0.5132800333608811,0.23190108939715093],
   [0.5132800333608811,0.0229177878448171,0.2319010893971509,0.23190108939715082],
   [0.7303134278075384,0.0379700484718286,0.0379700484718286,0.1937464752488044],
   [0.0379700484718286,0.7303134278075384,0.0379700484718286,0.1937464752488044],
   [0.0379700484718286,0.0379700484718286,0.7303134278075384,0.1937464752488044],
   [0.1937464752488044,0.0379700484718286,0.0379700484718286,0.7303134278075384],
   [0.0379700484718286,0.1937464752488044,0.0379700484718286,0.7303134278075384],
   [0.0379700484718286,0.0379700484718286,0.1937464752488044,0.7303134278075384],
   [0.0379700484718286,0.7303134278075384,0.1937464752488044,0.03797004847182861],
   [0.7303134278075384,0.1937464752488044,0.0379700484718286,0.03797004847182861],
   [0.1937464752488044,0.0379700484718286,0.7303134278075384,0.03797004847182861],
   [0.0379700484718286,0.1937464752488044,0.7303134278075384,0.03797004847182861],
   [0.7303134278075384,0.0379700484718286,0.1937464752488044,0.03797004847182861],
   [0.1937464752488044,0.7303134278075384,0.0379700484718286,0.03797004847182861]
  ])

  _self.gqWeights = np.array([
   [-0.2359620398477557],
   [0.0244878963560562],
   [0.0244878963560562],
   [0.0244878963560562],
   [0.0244878963560562],
   [0.0039485206398261],
   [0.0039485206398261],
   [0.0039485206398261],
   [0.0039485206398261],
   [0.0263055529507371],
   [0.0263055529507371],
   [0.0263055529507371],
   [0.0263055529507371],
   [0.0263055529507371],
   [0.0263055529507371],
   [0.0829803830550589],
   [0.0829803830550589],
   [0.0829803830550589],
   [0.0829803830550589],
   [0.0829803830550589],
   [0.0829803830550589],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0254426245481023],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852],
   [0.0134324384376852]
   ])

  # N1, N2, N3, N4
  _self.phiJ = np.array(_self.gqPoints)

  # dL1/dL1, dL2/dL1, dL3/dL1, dL4/dL1
  _self.dgqPointsdxi = np.array([ [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1],
                                  [1, 0, 0, -1] ])

  # dL1/dL2, dL2/dL2, dL3/dL2, dL4/dL2
  _self.dgqPointsdeta = np.array([ [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1],
                                   [0, 1, 0, -1] ])

  # dL1/dL3, dL2/dL3, dL3/dL3, dL4/dL3
  _self.dgqPointsdzeta = np.array([ [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1],
                                    [0, 0, 1, -1] ])

  _self.dphiJdxi = np.array(_self.dgqPointsdxi)
  _self.dphiJdeta = np.array(_self.dgqPointsdeta)
  _self.dphiJdzeta = np.array(_self.dgqPointsdzeta)


 def tet5(_self):
  _self.tet4()

  _self.phiJ = np.array([
   [0.0,0.0,0.0,0.0,1.0],
   [0.5357196422061783,0.04560338847273426,0.04560338847273426,0.04560338847273426,0.3274701923756189],
   [0.04560338847273426,0.04560338847273426,0.04560338847273426,0.5357196422061783,0.3274701923756189],
   [0.04560338847273426,0.04560338847273426,0.5357196422061783,0.04560338847273426,0.3274701923756189],
   [0.04560338847273426,0.5357196422061783,0.04560338847273426,0.04560338847273426,0.3274701923756189],
   [0.9018541376911209,0.030169459261650135,0.030169459261650135,0.03016945926165021,0.007637484523928654],
   [0.030169459261650138,0.030169459261650138,0.030169459261650138,0.901854137691121,0.007637484523928636],
   [0.03016945926165014,0.03016945926165014,0.901854137691121,0.030169459261650107,0.007637484523928628],
   [0.030169459261650135,0.9018541376911209,0.030169459261650135,0.03016945926165021,0.007637484523928654],
   [0.41807932391689323,0.017633515203455193,0.017633515203455193,0.4180793239168932,0.12857432175930322],
   [0.017633515203455193,0.41807932391689323,0.017633515203455193,0.4180793239168932,0.12857432175930322],
   [0.017633515203455193,0.017633515203455193,0.41807932391689323,0.4180793239168932,0.12857432175930322],
   [0.017633515203455165,0.4180793239168932,0.4180793239168932,0.0176335152034552,0.12857432175930333],
   [0.4180793239168932,0.017633515203455165,0.4180793239168932,0.0176335152034552,0.12857432175930333],
   [0.4180793239168933,0.4180793239168933,0.017633515203455234,0.017633515203455158,0.12857432175930306],
   [0.10016843765038563,-0.03237066755251458,-0.03237066755251458,0.10016843765038558,0.864404459804258],
   [-0.03237066755251458,0.10016843765038563,-0.03237066755251458,0.10016843765038558,0.864404459804258],
   [-0.032370667552514554,-0.032370667552514554,0.10016843765038566,0.1001684376503856,0.8644044598042578],
   [-0.03237066755251464,0.10016843765038558,0.10016843765038558,-0.03237066755251461,0.8644044598042582],
   [0.10016843765038558,-0.03237066755251464,0.10016843765038558,-0.03237066755251461,0.8644044598042582],
   [0.10016843765038569,0.10016843765038569,-0.032370667552514526,-0.03237066755251461,0.8644044598042577],
   [-0.017568934271220117,0.1914143672811137,0.1914143672811137,0.4727933112448439,0.16194688846414887],
   [0.1914143672811137,-0.017568934271220117,0.1914143672811137,0.4727933112448439,0.16194688846414887],
   [0.1914143672811137,0.1914143672811137,-0.01756893427122011,0.4727933112448439,0.16194688846414884],
   [0.4727933112448441,0.19141436728111383,0.19141436728111383,-0.017568934271220048,0.16194688846414826],
   [0.19141436728111383,0.4727933112448441,0.19141436728111383,-0.017568934271220048,0.16194688846414826],
   [0.19141436728111383,0.19141436728111383,0.4727933112448441,-0.017568934271220048,0.16194688846414826],
   [0.1914143672811137,-0.017568934271220117,0.4727933112448439,0.19141436728111372,0.16194688846414887],
   [-0.017568934271220103,0.4727933112448439,0.1914143672811137,0.1914143672811136,0.1619468884641488],
   [0.4727933112448439,0.1914143672811137,-0.017568934271220103,0.1914143672811136,0.1619468884641488],
   [0.1914143672811137,0.4727933112448439,-0.017568934271220103,0.1914143672811136,0.1619468884641488],
   [-0.017568934271220117,0.1914143672811137,0.4727933112448439,0.19141436728111372,0.16194688846414887],
   [0.4727933112448439,-0.017568934271220103,0.1914143672811137,0.1914143672811136,0.1619468884641488],
   [0.7172575711511433,0.02491419181543348,0.02491419181543348,0.18069061859240929,0.05222342662558049],
   [0.02491419181543348,0.7172575711511433,0.02491419181543348,0.18069061859240929,0.05222342662558049],
   [0.02491419181543348,0.02491419181543348,0.7172575711511433,0.18069061859240929,0.05222342662558049],
   [0.18069061859240926,0.02491419181543348,0.02491419181543348,0.7172575711511433,0.052223426625580494],
   [0.02491419181543348,0.18069061859240926,0.02491419181543348,0.7172575711511433,0.052223426625580494],
   [0.02491419181543348,0.02491419181543348,0.18069061859240926,0.7172575711511433,0.052223426625580494],
   [0.02491419181543348,0.7172575711511433,0.18069061859240926,0.024914191815433487,0.0522234266255805],
   [0.7172575711511433,0.18069061859240926,0.02491419181543348,0.024914191815433487,0.0522234266255805],
   [0.18069061859240926,0.02491419181543348,0.7172575711511433,0.024914191815433487,0.0522234266255805],
   [0.02491419181543348,0.18069061859240926,0.7172575711511433,0.024914191815433487,0.0522234266255805],
   [0.7172575711511433,0.02491419181543348,0.18069061859240926,0.024914191815433487,0.0522234266255805],
   [0.18069061859240926,0.7172575711511433,0.02491419181543348,0.024914191815433487,0.0522234266255805]
  ])

  _self.dphiJdxi = np.array([
   [1.0,0.0,0.0,-1.0,0.0],
   [1.5096845072750058,0.5096845072750058,0.5096845072750058,-0.4903154927249942,-2.038738029100023],
   [0.4903154927249942,-0.5096845072750058,-0.5096845072750058,-1.5096845072750058,2.038738029100023],
   [1.0,0.0,0.0,-1.0,0.0],
   [1.0,0.0,0.0,-1.0,0.0],
   [1.0574085314093116,0.05740853140931165,0.05740853140931165,-0.9425914685906884,-0.2296341256372466],
   [0.9425914685906884,-0.05740853140931166,-0.05740853140931166,-1.0574085314093116,0.22963412563724664],
   [1.0,6.245004513516506e-17,6.245004513516506e-17,-1.0,-2.498001805406602e-16],
   [0.9999999999999998,-1.457167719820518e-16,-1.457167719820518e-16,-1.0000000000000002,5.828670879282072e-16],
   [1.0,0.0,0.0,-1.0,0.0],
   [0.4256443933949432,-0.5743556066050568,-0.5743556066050568,-1.5743556066050568,2.297422426420227],
   [0.4256443933949432,-0.5743556066050568,-0.5743556066050568,-1.5743556066050568,2.297422426420227],
   [0.9999999999999996,-4.440892098500626e-16,-4.440892098500626e-16,-1.0000000000000004,1.7763568394002505e-15],
   [1.5743556066050568,0.5743556066050568,0.5743556066050568,-0.42564439339494325,-2.297422426420227],
   [1.574355606605057,0.5743556066050569,0.5743556066050569,-0.4256443933949431,-2.2974224264202276],
   [1.0000000000000002,2.220446049250313e-16,2.220446049250313e-16,-0.9999999999999998,-8.881784197001252e-16],
   [0.5070957928696596,-0.49290420713034044,-0.49290420713034044,-1.4929042071303404,1.9716168285213618],
   [0.5070957928696594,-0.49290420713034055,-0.49290420713034055,-1.4929042071303407,1.9716168285213622],
   [0.9999999999999998,-2.220446049250313e-16,-2.220446049250313e-16,-1.0000000000000002,8.881784197001252e-16],
   [1.4929042071303407,0.49290420713034055,0.49290420713034055,-0.5070957928696594,-1.9716168285213622],
   [1.492904207130341,0.492904207130341,0.492904207130341,-0.507095792869659,-1.971616828521364],
   [-0.6877284710575003,-1.6877284710575002,-1.6877284710575002,-2.6877284710575005,6.750913884230001],
   [0.9042922618072383,-0.09570773819276161,-0.09570773819276161,-1.0957077381927616,0.38283095277104645],
   [0.9042922618072383,-0.09570773819276163,-0.09570773819276163,-1.0957077381927616,0.3828309527710465],
   [2.6877284710575005,1.6877284710575007,1.6877284710575007,0.6877284710575007,-6.750913884230003],
   [2.5920207328647393,1.5920207328647393,1.5920207328647393,0.5920207328647393,-6.368082931458957],
   [2.5920207328647393,1.5920207328647393,1.5920207328647393,0.5920207328647393,-6.368082931458957],
   [1.0,-2.7755575615628914e-17,-2.7755575615628914e-17,-1.0,1.1102230246251565e-16],
   [-0.592020732864738,-1.592020732864738,-1.592020732864738,-2.592020732864738,6.368082931458952],
   [1.0957077381927616,0.09570773819276165,0.09570773819276165,-0.9042922618072383,-0.3828309527710466],
   [1.0,5.551115123125783e-17,5.551115123125783e-17,-0.9999999999999999,-2.220446049250313e-16],
   [-0.5920207328647389,-1.5920207328647389,-1.5920207328647389,-2.592020732864739,6.3680829314589555],
   [1.0957077381927616,0.09570773819276165,0.09570773819276165,-0.9042922618072383,-0.3828309527710466],
   [1.0495092329491824,0.04950923294918248,0.04950923294918248,-0.9504907670508176,-0.19803693179672993],
   [0.7235400984343655,-0.2764599015656345,-0.2764599015656345,-1.2764599015656346,1.105839606262538],
   [0.7235400984343655,-0.2764599015656345,-0.2764599015656345,-1.2764599015656346,1.105839606262538],
   [0.9504907670508176,-0.04950923294918248,-0.04950923294918248,-1.0495092329491824,0.19803693179672993],
   [0.6740308654851831,-0.32596913451481696,-0.32596913451481696,-1.3259691345148168,1.3038765380592678],
   [0.6740308654851831,-0.32596913451481696,-0.32596913451481696,-1.3259691345148168,1.3038765380592678],
   [0.9999999999999999,-5.551115123125783e-17,-5.551115123125783e-17,-1.0,2.220446049250313e-16],
   [1.3259691345148168,0.32596913451481696,0.32596913451481696,-0.6740308654851831,-1.3038765380592678],
   [1.2764599015656346,0.27645990156563444,0.27645990156563444,-0.7235400984343655,-1.1058396062625377],
   [0.9999999999999999,-5.551115123125783e-17,-5.551115123125783e-17,-1.0,2.220446049250313e-16],
   [1.3259691345148168,0.32596913451481696,0.32596913451481696,-0.6740308654851831,-1.3038765380592678],
   [1.2764599015656346,0.27645990156563444,0.27645990156563444,-0.7235400984343655,-1.1058396062625377]
  ])

  _self.dphiJdeta = np.array([
   [0.0,1.0,0.0,-1.0,0.0],
   [0.0,1.0,0.0,-1.0,0.0],
   [-0.5096845072750058,0.4903154927249942,-0.5096845072750058,-1.5096845072750058,2.038738029100023],
   [0.0,1.0,0.0,-1.0,0.0],
   [0.5096845072750058,1.5096845072750058,0.5096845072750058,-0.4903154927249942,-2.038738029100023],
   [-1.457167719820518e-16,0.9999999999999998,-1.457167719820518e-16,-1.0000000000000002,5.828670879282072e-16],
   [-0.05740853140931166,0.9425914685906884,-0.05740853140931166,-1.0574085314093116,0.22963412563724664],
   [6.245004513516506e-17,1.0,6.245004513516506e-17,-1.0,-2.498001805406602e-16],
   [0.05740853140931165,1.0574085314093116,0.05740853140931165,-0.9425914685906884,-0.2296341256372466],
   [-0.5743556066050568,0.4256443933949432,-0.5743556066050568,-1.5743556066050568,2.297422426420227],
   [0.0,1.0,0.0,-1.0,0.0],
   [-0.5743556066050568,0.4256443933949432,-0.5743556066050568,-1.5743556066050568,2.297422426420227],
   [0.5743556066050568,1.5743556066050568,0.5743556066050568,-0.42564439339494325,-2.297422426420227],
   [-4.440892098500626e-16,0.9999999999999996,-4.440892098500626e-16,-1.0000000000000004,1.7763568394002505e-15],
   [0.5743556066050569,1.574355606605057,0.5743556066050569,-0.4256443933949431,-2.2974224264202276],
   [-0.49290420713034044,0.5070957928696596,-0.49290420713034044,-1.4929042071303404,1.9716168285213618],
   [2.220446049250313e-16,1.0000000000000002,2.220446049250313e-16,-0.9999999999999998,-8.881784197001252e-16],
   [-0.49290420713034055,0.5070957928696594,-0.49290420713034055,-1.4929042071303407,1.9716168285213622],
   [0.49290420713034055,1.4929042071303407,0.49290420713034055,-0.5070957928696594,-1.9716168285213622],
   [-2.220446049250313e-16,0.9999999999999998,-2.220446049250313e-16,-1.0000000000000002,8.881784197001252e-16],
   [0.492904207130341,1.492904207130341,0.492904207130341,-0.507095792869659,-1.971616828521364],
   [-0.09570773819276161,0.9042922618072383,-0.09570773819276161,-1.0957077381927616,0.38283095277104645],
   [-1.6877284710575002,-0.6877284710575003,-1.6877284710575002,-2.6877284710575005,6.750913884230001],
   [-0.09570773819276163,0.9042922618072383,-0.09570773819276163,-1.0957077381927616,0.3828309527710465],
   [1.5920207328647393,2.5920207328647393,1.5920207328647393,0.5920207328647393,-6.368082931458957],
   [1.6877284710575007,2.6877284710575005,1.6877284710575007,0.6877284710575007,-6.750913884230003],
   [1.5920207328647393,2.5920207328647393,1.5920207328647393,0.5920207328647393,-6.368082931458957],
   [-1.5920207328647389,-0.5920207328647389,-1.5920207328647389,-2.592020732864739,6.3680829314589555],
   [0.09570773819276165,1.0957077381927616,0.09570773819276165,-0.9042922618072383,-0.3828309527710466],
   [5.551115123125783e-17,1.0,5.551115123125783e-17,-0.9999999999999999,-2.220446049250313e-16],
   [0.09570773819276165,1.0957077381927616,0.09570773819276165,-0.9042922618072383,-0.3828309527710466],
   [-2.7755575615628914e-17,1.0,-2.7755575615628914e-17,-1.0,1.1102230246251565e-16],
   [-1.592020732864738,-0.592020732864738,-1.592020732864738,-2.592020732864738,6.368082931458952],
   [-0.2764599015656345,0.7235400984343655,-0.2764599015656345,-1.2764599015656346,1.105839606262538],
   [0.04950923294918248,1.0495092329491824,0.04950923294918248,-0.9504907670508176,-0.19803693179672993],
   [-0.2764599015656345,0.7235400984343655,-0.2764599015656345,-1.2764599015656346,1.105839606262538],
   [-0.32596913451481696,0.6740308654851831,-0.32596913451481696,-1.3259691345148168,1.3038765380592678],
   [-0.04950923294918248,0.9504907670508176,-0.04950923294918248,-1.0495092329491824,0.19803693179672993],
   [-0.32596913451481696,0.6740308654851831,-0.32596913451481696,-1.3259691345148168,1.3038765380592678],
   [0.32596913451481696,1.3259691345148168,0.32596913451481696,-0.6740308654851831,-1.3038765380592678],
   [0.27645990156563444,1.2764599015656346,0.27645990156563444,-0.7235400984343655,-1.1058396062625377],
   [-5.551115123125783e-17,0.9999999999999999,-5.551115123125783e-17,-1.0,2.220446049250313e-16],
   [0.27645990156563444,1.2764599015656346,0.27645990156563444,-0.7235400984343655,-1.1058396062625377],
   [-5.551115123125783e-17,0.9999999999999999,-5.551115123125783e-17,-1.0,2.220446049250313e-16],
   [0.32596913451481696,1.3259691345148168,0.32596913451481696,-0.6740308654851831,-1.3038765380592678]
  ])

  _self.dphiJdzeta = np.array([
   [0.0,0.0,1.0,-1.0,0.0],
   [0.0,0.0,1.0,-1.0,0.0],
   [-0.5096845072750058,-0.5096845072750058,0.4903154927249942,-1.5096845072750058,2.038738029100023],
   [0.5096845072750058,0.5096845072750058,1.5096845072750058,-0.4903154927249942,-2.038738029100023],
   [0.0,0.0,1.0,-1.0,0.0],
   [-1.457167719820518e-16,-1.457167719820518e-16,0.9999999999999998,-1.0000000000000002,5.828670879282072e-16],
   [-0.05740853140931166,-0.05740853140931166,0.9425914685906884,-1.0574085314093116,0.22963412563724664],
   [0.05740853140931166,0.05740853140931166,1.0574085314093116,-0.9425914685906884,-0.22963412563724664],
   [-1.457167719820518e-16,-1.457167719820518e-16,0.9999999999999998,-1.0000000000000002,5.828670879282072e-16],
   [-0.5743556066050568,-0.5743556066050568,0.4256443933949432,-1.5743556066050568,2.297422426420227],
   [-0.5743556066050568,-0.5743556066050568,0.4256443933949432,-1.5743556066050568,2.297422426420227],
   [0.0,0.0,1.0,-1.0,0.0],
   [0.5743556066050568,0.5743556066050568,1.5743556066050568,-0.42564439339494325,-2.297422426420227],
   [0.5743556066050568,0.5743556066050568,1.5743556066050568,-0.42564439339494325,-2.297422426420227],
   [9.992007221626409e-16,9.992007221626409e-16,1.0000000000000009,-0.999999999999999,-3.9968028886505635e-15],
   [-0.49290420713034044,-0.49290420713034044,0.5070957928696596,-1.4929042071303404,1.9716168285213618],
   [-0.49290420713034044,-0.49290420713034044,0.5070957928696596,-1.4929042071303404,1.9716168285213618],
   [1.1102230246251565e-16,1.1102230246251565e-16,1.0,-0.9999999999999999,-4.440892098500626e-16],
   [0.49290420713034055,0.49290420713034055,1.4929042071303407,-0.5070957928696594,-1.9716168285213622],
   [0.49290420713034055,0.49290420713034055,1.4929042071303407,-0.5070957928696594,-1.9716168285213622],
   [6.661338147750939e-16,6.661338147750939e-16,1.0000000000000007,-0.9999999999999993,-2.6645352591003757e-15],
   [-0.09570773819276161,-0.09570773819276161,0.9042922618072383,-1.0957077381927616,0.38283095277104645],
   [-0.09570773819276161,-0.09570773819276161,0.9042922618072383,-1.0957077381927616,0.38283095277104645],
   [-1.6877284710575005,-1.6877284710575005,-0.6877284710575003,-2.6877284710575005,6.750913884230002],
   [1.5920207328647393,1.5920207328647393,2.5920207328647393,0.5920207328647393,-6.368082931458957],
   [1.5920207328647393,1.5920207328647393,2.5920207328647393,0.5920207328647393,-6.368082931458957],
   [1.6877284710575007,1.6877284710575007,2.6877284710575005,0.6877284710575007,-6.750913884230003],
   [0.09570773819276161,0.09570773819276161,1.0957077381927616,-0.9042922618072383,-0.38283095277104645],
   [5.551115123125783e-17,5.551115123125783e-17,1.0,-0.9999999999999999,-2.220446049250313e-16],
   [-1.592020732864738,-1.592020732864738,-0.592020732864738,-2.592020732864738,6.368082931458952],
   [-1.592020732864738,-1.592020732864738,-0.592020732864738,-2.592020732864738,6.368082931458952],
   [0.09570773819276161,0.09570773819276161,1.0957077381927616,-0.9042922618072383,-0.38283095277104645],
   [5.551115123125783e-17,5.551115123125783e-17,1.0,-0.9999999999999999,-2.220446049250313e-16],
   [-0.2764599015656345,-0.2764599015656345,0.7235400984343655,-1.2764599015656346,1.105839606262538],
   [-0.2764599015656345,-0.2764599015656345,0.7235400984343655,-1.2764599015656346,1.105839606262538],
   [0.04950923294918248,0.04950923294918248,1.0495092329491824,-0.9504907670508176,-0.19803693179672993],
   [-0.32596913451481696,-0.32596913451481696,0.6740308654851831,-1.3259691345148168,1.3038765380592678],
   [-0.32596913451481696,-0.32596913451481696,0.6740308654851831,-1.3259691345148168,1.3038765380592678],
   [-0.04950923294918248,-0.04950923294918248,0.9504907670508176,-1.0495092329491824,0.19803693179672993],
   [0.27645990156563444,0.27645990156563444,1.2764599015656346,-0.7235400984343655,-1.1058396062625377],
   [-5.551115123125783e-17,-5.551115123125783e-17,0.9999999999999999,-1.0,2.220446049250313e-16],
   [0.32596913451481696,0.32596913451481696,1.3259691345148168,-0.6740308654851831,-1.3038765380592678],
   [0.32596913451481696,0.32596913451481696,1.3259691345148168,-0.6740308654851831,-1.3038765380592678],
   [0.27645990156563444,0.27645990156563444,1.2764599015656346,-0.7235400984343655,-1.1058396062625377],
   [-5.551115123125783e-17,-5.551115123125783e-17,0.9999999999999999,-1.0,2.220446049250313e-16]
  ])

 def tet10(_self):
  _self.tet4()

  _self.phiJ = np.array([
 [-0.125,-0.125,-0.125,-0.125,0.25,0.25,0.25,0.25,0.25,0.25],
 [0.145240684945,-0.0949732572283,-0.0949732572283,-0.0949732572283,0.314897670236,0.0649953586767,0.314897670236,0.314897670236,0.0649953586767,0.0649953586767],
 [-0.0949732572283,-0.0949732572283,-0.0949732572283,0.145240684945,0.0649953586767,0.0649953586767,0.0649953586767,0.314897670236,0.314897670236,0.314897670236],
 [-0.0949732572283,-0.0949732572283,0.145240684945,-0.0949732572283,0.0649953586767,0.314897670236,0.314897670236,0.0649953586767,0.0649953586767,0.314897670236],
 [-0.0949732572283,0.145240684945,-0.0949732572283,-0.0949732572283,0.314897670236,0.314897670236,0.0649953586767,0.0649953586767,0.314897670236,0.0649953586767],
 [0.729813450935,-0.0300207276739,-0.0300207276739,-0.0300207276739,0.115966705258,0.00411620543744,0.115966705258,0.115966705258,0.00411620543744,0.00411620543744],
 [-0.0300207276739,-0.0300207276739,-0.0300207276739,0.729813450935,0.00411620543744,0.00411620543744,0.00411620543744,0.115966705258,0.115966705258,0.115966705258],
 [-0.0300207276739,-0.0300207276739,0.729813450935,-0.0300207276739,0.00411620543744,0.115966705258,0.115966705258,0.00411620543744,0.00411620543744,0.115966705258],
 [-0.0300207276739,0.729813450935,-0.0300207276739,-0.0300207276739,0.115966705258,0.115966705258,0.00411620543744,0.00411620543744,0.115966705258,0.00411620543744],
 [-0.0448215771419,-0.0448215771419,-0.0448215771419,-0.0448215771419,0.0896431542838,0.00991103700272,0.0896431542838,0.81080265443,0.0896431542838,0.0896431542838],
 [-0.0448215771419,-0.0448215771419,-0.0448215771419,-0.0448215771419,0.0896431542838,0.0896431542838,0.00991103700272,0.0896431542838,0.81080265443,0.0896431542838],
 [-0.0448215771419,-0.0448215771419,-0.0448215771419,-0.0448215771419,0.00991103700272,0.0896431542838,0.0896431542838,0.0896431542838,0.0896431542838,0.81080265443],
 [-0.0448215771419,-0.0448215771419,-0.0448215771419,-0.0448215771419,0.0896431542838,0.81080265443,0.0896431542838,0.00991103700272,0.0896431542838,0.0896431542838],
 [-0.0448215771419,-0.0448215771419,-0.0448215771419,-0.0448215771419,0.0896431542838,0.0896431542838,0.81080265443,0.0896431542838,0.00991103700272,0.0896431542838],
 [-0.0448215771419,-0.0448215771419,-0.0448215771419,-0.0448215771419,0.81080265443,0.0896431542838,0.0896431542838,0.0896431542838,0.0896431542838,0.00991103700272],
 [-0.116216692796,-0.116216692796,-0.116216692796,-0.116216692796,0.232433385592,0.135027509205,0.232433385592,0.400105719611,0.232433385592,0.232433385592],
 [-0.116216692796,-0.116216692796,-0.116216692796,-0.116216692796,0.232433385592,0.232433385592,0.135027509205,0.232433385592,0.400105719611,0.232433385592],
 [-0.116216692796,-0.116216692796,-0.116216692796,-0.116216692796,0.135027509205,0.232433385592,0.232433385592,0.232433385592,0.232433385592,0.400105719611],
 [-0.116216692796,-0.116216692796,-0.116216692796,-0.116216692796,0.232433385592,0.400105719611,0.232433385592,0.135027509205,0.232433385592,0.232433385592],
 [-0.116216692796,-0.116216692796,-0.116216692796,-0.116216692796,0.232433385592,0.232433385592,0.400105719611,0.232433385592,0.135027509205,0.232433385592],
 [-0.116216692796,-0.116216692796,-0.116216692796,-0.116216692796,0.400105719611,0.232433385592,0.232433385592,0.232433385592,0.232433385592,0.135027509205],
 [-0.0218673378454,-0.12434485887,-0.12434485887,0.013632751933,0.0212586398711,0.215112461054,0.0212586398711,0.0470529716382,0.476120795609,0.476120795609],
 [-0.12434485887,-0.0218673378454,-0.12434485887,0.013632751933,0.0212586398711,0.0212586398711,0.215112461054,0.476120795609,0.0470529716382,0.476120795609],
 [-0.12434485887,-0.12434485887,-0.0218673378454,0.013632751933,0.215112461054,0.0212586398711,0.0212586398711,0.476120795609,0.476120795609,0.0470529716382],
 [0.013632751933,-0.12434485887,-0.12434485887,-0.0218673378454,0.476120795609,0.215112461054,0.476120795609,0.0470529716382,0.0212586398711,0.0212586398711],
 [-0.12434485887,0.013632751933,-0.12434485887,-0.0218673378454,0.476120795609,0.476120795609,0.215112461054,0.0212586398711,0.0470529716382,0.0212586398711],
 [-0.12434485887,-0.12434485887,0.013632751933,-0.0218673378454,0.215112461054,0.476120795609,0.476120795609,0.0212586398711,0.0212586398711,0.0470529716382],
 [-0.12434485887,-0.0218673378454,0.013632751933,-0.12434485887,0.0212586398711,0.0470529716382,0.476120795609,0.215112461054,0.0212586398711,0.476120795609],
 [-0.0218673378454,0.013632751933,-0.12434485887,-0.12434485887,0.0470529716382,0.476120795609,0.0212586398711,0.0212586398711,0.476120795609,0.215112461054],
 [0.013632751933,-0.12434485887,-0.0218673378454,-0.12434485887,0.476120795609,0.0212586398711,0.0470529716382,0.476120795609,0.215112461054,0.0212586398711],
 [-0.12434485887,0.013632751933,-0.0218673378454,-0.12434485887,0.476120795609,0.0470529716382,0.0212586398711,0.215112461054,0.476120795609,0.0212586398711],
 [-0.0218673378454,-0.12434485887,0.013632751933,-0.12434485887,0.0212586398711,0.476120795609,0.0470529716382,0.0212586398711,0.215112461054,0.476120795609],
 [0.013632751933,-0.0218673378454,-0.12434485887,-0.12434485887,0.0470529716382,0.0212586398711,0.476120795609,0.476120795609,0.0212586398711,0.215112461054],
 [0.336401977864,-0.0350865993099,-0.0350865993099,-0.118671081906,0.110920145014,0.00576689832381,0.110920145014,0.565982609858,0.0294262522258,0.0294262522258],
 [-0.0350865993099,0.336401977864,-0.0350865993099,-0.118671081906,0.110920145014,0.110920145014,0.00576689832381,0.0294262522258,0.565982609858,0.0294262522258],
 [-0.0350865993099,-0.0350865993099,0.336401977864,-0.118671081906,0.00576689832381,0.110920145014,0.110920145014,0.0294262522258,0.0294262522258,0.565982609858],
 [-0.118671081906,-0.0350865993099,-0.0350865993099,0.336401977864,0.0294262522258,0.00576689832381,0.0294262522258,0.565982609858,0.110920145014,0.110920145014],
 [-0.0350865993099,-0.118671081906,-0.0350865993099,0.336401977864,0.0294262522258,0.0294262522258,0.00576689832381,0.110920145014,0.565982609858,0.110920145014],
 [-0.0350865993099,-0.0350865993099,-0.118671081906,0.336401977864,0.00576689832381,0.0294262522258,0.0294262522258,0.110920145014,0.110920145014,0.565982609858],
 [-0.0350865993099,0.336401977864,-0.118671081906,-0.0350865993099,0.110920145014,0.565982609858,0.0294262522258,0.00576689832381,0.110920145014,0.0294262522258],
 [0.336401977864,-0.118671081906,-0.0350865993099,-0.0350865993099,0.565982609858,0.0294262522258,0.110920145014,0.110920145014,0.0294262522258,0.00576689832381],
 [-0.118671081906,-0.0350865993099,0.336401977864,-0.0350865993099,0.0294262522258,0.110920145014,0.565982609858,0.0294262522258,0.00576689832381,0.110920145014],
 [-0.0350865993099,-0.118671081906,0.336401977864,-0.0350865993099,0.0294262522258,0.565982609858,0.110920145014,0.00576689832381,0.0294262522258,0.110920145014],
 [0.336401977864,-0.0350865993099,-0.118671081906,-0.0350865993099,0.110920145014,0.0294262522258,0.565982609858,0.110920145014,0.00576689832381,0.0294262522258],
 [-0.118671081906,0.336401977864,-0.0350865993099,-0.0350865993099,0.565982609858,0.110920145014,0.0294262522258,0.0294262522258,0.110920145014,0.00576689832381]
  ])

  _self.dphiJdxi = np.array([
 [0.0,0.0,0.0,-0,1.0,0.0,1.0,0.0,-1.0,-1.0],
 [1.4703487612,0.0,0.0,0.490116253733,0.509883746267,0.0,0.509883746267,-1.96046501493,-0.509883746267,-0.509883746267],
 [-0.490116253733,0.0,0.0,-1.4703487612,0.509883746267,0.0,0.509883746267,1.96046501493,-0.509883746267,-0.509883746267],
 [-0.490116253733,0.0,0.0,0.490116253733,0.509883746267,0.0,2.4703487612,0.0,-0.509883746267,-2.4703487612],
 [-0.490116253733,0.0,0.0,0.490116253733,2.4703487612,0.0,0.509883746267,0.0,-2.4703487612,-0.509883746267],
 [2.61505403529,0.0,0.0,0.871684678429,0.128315321571,0.0,0.128315321571,-3.48673871372,-0.128315321571,-0.128315321571],
 [-0.871684678429,0.0,0.0,-2.61505403529,0.128315321571,0.0,0.128315321571,3.48673871372,-0.128315321571,-0.128315321571],
 [-0.871684678429,0.0,0.0,0.871684678429,0.128315321571,0.0,3.61505403529,-1.38777878078e-16,-0.128315321571,-3.61505403529],
 [-0.871684678429,0.0,0.0,0.871684678429,3.61505403529,0.0,0.128315321571,3.05311331772e-16,-3.61505403529,-0.128315321571],
 [0.800891617427,0.0,0.0,-0.800891617427,0.199108382573,0.0,0.199108382573,-2.22044604925e-16,-0.199108382573,-0.199108382573],
 [-0.800891617427,0.0,0.0,-0.800891617427,1.80089161743,0.0,0.199108382573,1.60178323485,-1.80089161743,-0.199108382573],
 [-0.800891617427,0.0,0.0,-0.800891617427,0.199108382573,0.0,1.80089161743,1.60178323485,-0.199108382573,-1.80089161743],
 [-0.800891617427,0.0,0.0,0.800891617427,1.80089161743,0.0,1.80089161743,1.38777878078e-16,-1.80089161743,-1.80089161743],
 [0.800891617427,0.0,0.0,0.800891617427,0.199108382573,0.0,1.80089161743,-1.60178323485,-0.199108382573,-1.80089161743],
 [0.800891617427,0.0,0.0,0.800891617427,1.80089161743,0.0,0.199108382573,-1.60178323485,-1.80089161743,-0.199108382573],
 [0.265078210406,0.0,0.0,-0.265078210406,0.734921789594,0.0,0.734921789594,-2.22044604925e-16,-0.734921789594,-0.734921789594],
 [-0.265078210406,0.0,0.0,-0.265078210406,1.26507821041,0.0,0.734921789594,0.530156420812,-1.26507821041,-0.734921789594],
 [-0.265078210406,0.0,0.0,-0.265078210406,0.734921789594,0.0,1.26507821041,0.530156420812,-0.734921789594,-1.26507821041],
 [-0.265078210406,0.0,0.0,0.265078210406,1.26507821041,0.0,1.26507821041,1.11022302463e-16,-1.26507821041,-1.26507821041],
 [0.265078210406,0.0,0.0,0.265078210406,0.734921789594,0.0,1.26507821041,-0.530156420812,-0.734921789594,-1.26507821041],
 [0.265078210406,0.0,0.0,0.265078210406,1.26507821041,0.0,0.734921789594,-0.530156420812,-1.26507821041,-0.734921789594],
 [-0.908328848621,0.0,0.0,-1.05312013344,0.927604357589,0.0,0.927604357589,1.96144898206,-0.927604357589,-0.927604357589],
 [-0.0723956424114,0.0,0.0,-1.05312013344,0.0916711513793,0.0,0.927604357589,1.12551577585,-0.0916711513793,-0.927604357589],
 [-0.0723956424114,0.0,0.0,-1.05312013344,0.927604357589,0.0,0.0916711513793,1.12551577585,-0.927604357589,-0.0916711513793],
 [1.05312013344,0.0,0.0,0.908328848621,0.927604357589,0.0,0.927604357589,-1.96144898206,-0.927604357589,-0.927604357589],
 [-0.0723956424114,0.0,0.0,0.908328848621,2.05312013344,0.0,0.927604357589,-0.835933206209,-2.05312013344,-0.927604357589],
 [-0.0723956424114,0.0,0.0,0.908328848621,0.927604357589,0.0,2.05312013344,-0.835933206209,-0.927604357589,-2.05312013344],
 [-0.0723956424114,0.0,0.0,0.0723956424114,0.0916711513793,0.0,2.05312013344,1.11022302463e-16,-0.0916711513793,-2.05312013344],
 [-0.908328848621,0.0,0.0,0.0723956424114,2.05312013344,0.0,0.927604357589,0.835933206209,-2.05312013344,-0.927604357589],
 [1.05312013344,0.0,0.0,0.0723956424114,0.927604357589,0.0,0.0916711513793,-1.12551577585,-0.927604357589,-0.0916711513793],
 [-0.0723956424114,0.0,0.0,0.0723956424114,2.05312013344,0.0,0.0916711513793,-3.33066907388e-16,-2.05312013344,-0.0916711513793],
 [-0.908328848621,0.0,0.0,0.0723956424114,0.927604357589,0.0,2.05312013344,0.835933206209,-0.927604357589,-2.05312013344],
 [1.05312013344,0.0,0.0,0.0723956424114,0.0916711513793,0.0,0.927604357589,-1.12551577585,-0.0916711513793,-0.927604357589],
 [1.92125371123,0.0,0.0,0.225014099005,0.151880193887,0.0,0.151880193887,-2.14626781023,-0.151880193887,-0.151880193887],
 [-0.848119806113,0.0,0.0,0.225014099005,2.92125371123,0.0,0.151880193887,0.623105707108,-2.92125371123,-0.151880193887],
 [-0.848119806113,0.0,0.0,0.225014099005,0.151880193887,0.0,2.92125371123,0.623105707108,-0.151880193887,-2.92125371123],
 [-0.225014099005,0.0,0.0,-1.92125371123,0.151880193887,0.0,0.151880193887,2.14626781023,-0.151880193887,-0.151880193887],
 [-0.848119806113,0.0,0.0,-1.92125371123,0.774985900995,0.0,0.151880193887,2.76937351734,-0.774985900995,-0.151880193887],
 [-0.848119806113,0.0,0.0,-1.92125371123,0.151880193887,0.0,0.774985900995,2.76937351734,-0.151880193887,-0.774985900995],
 [-0.848119806113,0.0,0.0,0.848119806113,2.92125371123,0.0,0.774985900995,2.77555756156e-17,-2.92125371123,-0.774985900995],
 [1.92125371123,0.0,0.0,0.848119806113,0.774985900995,0.0,0.151880193887,-2.76937351734,-0.774985900995,-0.151880193887],
 [-0.225014099005,0.0,0.0,0.848119806113,0.151880193887,0.0,2.92125371123,-0.623105707108,-0.151880193887,-2.92125371123],
 [-0.848119806113,0.0,0.0,0.848119806113,0.774985900995,0.0,2.92125371123,2.77555756156e-17,-0.774985900995,-2.92125371123],
 [1.92125371123,0.0,0.0,0.848119806113,0.151880193887,0.0,0.774985900995,-2.76937351734,-0.151880193887,-0.774985900995],
 [-0.225014099005,0.0,0.0,0.848119806113,2.92125371123,0.0,0.151880193887,-0.623105707108,-2.92125371123,-0.151880193887]
  ])

  _self.dphiJdeta = np.array([
 [0.0,0.0,0.0,-0,1.0,1.0,0.0,-1.0,0.0,-1.0],
 [0.0,-0.490116253733,0.0,0.490116253733,2.4703487612,0.509883746267,0.0,-2.4703487612,0.0,-0.509883746267],
 [0.0,-0.490116253733,0.0,-1.4703487612,0.509883746267,0.509883746267,0.0,-0.509883746267,1.96046501493,-0.509883746267],
 [0.0,-0.490116253733,0.0,0.490116253733,0.509883746267,2.4703487612,0.0,-0.509883746267,0.0,-2.4703487612],
 [0.0,1.4703487612,0.0,0.490116253733,0.509883746267,0.509883746267,0.0,-0.509883746267,-1.96046501493,-0.509883746267],
 [0.0,-0.871684678429,0.0,0.871684678429,3.61505403529,0.128315321571,0.0,-3.61505403529,3.05311331772e-16,-0.128315321571],
 [0.0,-0.871684678429,0.0,-2.61505403529,0.128315321571,0.128315321571,0.0,-0.128315321571,3.48673871372,-0.128315321571],
 [0.0,-0.871684678429,0.0,0.871684678429,0.128315321571,3.61505403529,0.0,-0.128315321571,-1.38777878078e-16,-3.61505403529],
 [0.0,2.61505403529,0.0,0.871684678429,0.128315321571,0.128315321571,0.0,-0.128315321571,-3.48673871372,-0.128315321571],
 [0.0,-0.800891617427,0.0,-0.800891617427,1.80089161743,0.199108382573,0.0,-1.80089161743,1.60178323485,-0.199108382573],
 [0.0,0.800891617427,0.0,-0.800891617427,0.199108382573,0.199108382573,0.0,-0.199108382573,-2.22044604925e-16,-0.199108382573],
 [0.0,-0.800891617427,0.0,-0.800891617427,0.199108382573,1.80089161743,0.0,-0.199108382573,1.60178323485,-1.80089161743],
 [0.0,0.800891617427,0.0,0.800891617427,0.199108382573,1.80089161743,0.0,-0.199108382573,-1.60178323485,-1.80089161743],
 [0.0,-0.800891617427,0.0,0.800891617427,1.80089161743,1.80089161743,0.0,-1.80089161743,1.38777878078e-16,-1.80089161743],
 [0.0,0.800891617427,0.0,0.800891617427,1.80089161743,0.199108382573,0.0,-1.80089161743,-1.60178323485,-0.199108382573],
 [0.0,-0.265078210406,0.0,-0.265078210406,1.26507821041,0.734921789594,0.0,-1.26507821041,0.530156420812,-0.734921789594],
 [0.0,0.265078210406,0.0,-0.265078210406,0.734921789594,0.734921789594,0.0,-0.734921789594,-2.22044604925e-16,-0.734921789594],
 [0.0,-0.265078210406,0.0,-0.265078210406,0.734921789594,1.26507821041,0.0,-0.734921789594,0.530156420812,-1.26507821041],
 [0.0,0.265078210406,0.0,0.265078210406,0.734921789594,1.26507821041,0.0,-0.734921789594,-0.530156420812,-1.26507821041],
 [0.0,-0.265078210406,0.0,0.265078210406,1.26507821041,1.26507821041,0.0,-1.26507821041,1.11022302463e-16,-1.26507821041],
 [0.0,0.265078210406,0.0,0.265078210406,1.26507821041,0.734921789594,0.0,-1.26507821041,-0.530156420812,-0.734921789594],
 [0.0,-0.0723956424114,0.0,-1.05312013344,0.0916711513793,0.927604357589,0.0,-0.0916711513793,1.12551577585,-0.927604357589],
 [0.0,-0.908328848621,0.0,-1.05312013344,0.927604357589,0.927604357589,0.0,-0.927604357589,1.96144898206,-0.927604357589],
 [0.0,-0.0723956424114,0.0,-1.05312013344,0.927604357589,0.0916711513793,0.0,-0.927604357589,1.12551577585,-0.0916711513793],
 [0.0,-0.0723956424114,0.0,0.908328848621,2.05312013344,0.927604357589,0.0,-2.05312013344,-0.835933206209,-0.927604357589],
 [0.0,1.05312013344,0.0,0.908328848621,0.927604357589,0.927604357589,0.0,-0.927604357589,-1.96144898206,-0.927604357589],
 [0.0,-0.0723956424114,0.0,0.908328848621,0.927604357589,2.05312013344,0.0,-0.927604357589,-0.835933206209,-2.05312013344],
 [0.0,-0.908328848621,0.0,0.0723956424114,0.927604357589,2.05312013344,0.0,-0.927604357589,0.835933206209,-2.05312013344],
 [0.0,1.05312013344,0.0,0.0723956424114,0.0916711513793,0.927604357589,0.0,-0.0916711513793,-1.12551577585,-0.927604357589],
 [0.0,-0.0723956424114,0.0,0.0723956424114,2.05312013344,0.0916711513793,0.0,-2.05312013344,-3.33066907388e-16,-0.0916711513793],
 [0.0,1.05312013344,0.0,0.0723956424114,0.927604357589,0.0916711513793,0.0,-0.927604357589,-1.12551577585,-0.0916711513793],
 [0.0,-0.0723956424114,0.0,0.0723956424114,0.0916711513793,2.05312013344,0.0,-0.0916711513793,1.11022302463e-16,-2.05312013344],
 [0.0,-0.908328848621,0.0,0.0723956424114,2.05312013344,0.927604357589,0.0,-2.05312013344,0.835933206209,-0.927604357589],
 [0.0,-0.848119806113,0.0,0.225014099005,2.92125371123,0.151880193887,0.0,-2.92125371123,0.623105707108,-0.151880193887],
 [0.0,1.92125371123,0.0,0.225014099005,0.151880193887,0.151880193887,0.0,-0.151880193887,-2.14626781023,-0.151880193887],
 [0.0,-0.848119806113,0.0,0.225014099005,0.151880193887,2.92125371123,0.0,-0.151880193887,0.623105707108,-2.92125371123],
 [0.0,-0.848119806113,0.0,-1.92125371123,0.774985900995,0.151880193887,0.0,-0.774985900995,2.76937351734,-0.151880193887],
 [0.0,-0.225014099005,0.0,-1.92125371123,0.151880193887,0.151880193887,0.0,-0.151880193887,2.14626781023,-0.151880193887],
 [0.0,-0.848119806113,0.0,-1.92125371123,0.151880193887,0.774985900995,0.0,-0.151880193887,2.76937351734,-0.774985900995],
 [0.0,1.92125371123,0.0,0.848119806113,0.151880193887,0.774985900995,0.0,-0.151880193887,-2.76937351734,-0.774985900995],
 [0.0,-0.225014099005,0.0,0.848119806113,2.92125371123,0.151880193887,0.0,-2.92125371123,-0.623105707108,-0.151880193887],
 [0.0,-0.848119806113,0.0,0.848119806113,0.774985900995,2.92125371123,0.0,-0.774985900995,2.77555756156e-17,-2.92125371123],
 [0.0,-0.225014099005,0.0,0.848119806113,0.151880193887,2.92125371123,0.0,-0.151880193887,-0.623105707108,-2.92125371123],
 [0.0,-0.848119806113,0.0,0.848119806113,2.92125371123,0.774985900995,0.0,-2.92125371123,2.77555756156e-17,-0.774985900995],
 [0.0,1.92125371123,0.0,0.848119806113,0.774985900995,0.151880193887,0.0,-0.774985900995,-2.76937351734,-0.151880193887]
  ])

  _self.dphiJdzeta = np.array([
 [0.0,0.0,0.0,-0,0.0,1.0,1.0,-1.0,-1.0,0.0],
 [0.0,0.0,-0.490116253733,0.490116253733,0.0,0.509883746267,2.4703487612,-2.4703487612,-0.509883746267,0.0],
 [0.0,0.0,-0.490116253733,-1.4703487612,0.0,0.509883746267,0.509883746267,-0.509883746267,-0.509883746267,1.96046501493],
 [0.0,0.0,1.4703487612,0.490116253733,0.0,0.509883746267,0.509883746267,-0.509883746267,-0.509883746267,-1.96046501493],
 [0.0,0.0,-0.490116253733,0.490116253733,0.0,2.4703487612,0.509883746267,-0.509883746267,-2.4703487612,0.0],
 [0.0,0.0,-0.871684678429,0.871684678429,0.0,0.128315321571,3.61505403529,-3.61505403529,-0.128315321571,3.05311331772e-16],
 [0.0,0.0,-0.871684678429,-2.61505403529,0.0,0.128315321571,0.128315321571,-0.128315321571,-0.128315321571,3.48673871372],
 [0.0,0.0,2.61505403529,0.871684678429,0.0,0.128315321571,0.128315321571,-0.128315321571,-0.128315321571,-3.48673871372],
 [0.0,0.0,-0.871684678429,0.871684678429,0.0,3.61505403529,0.128315321571,-0.128315321571,-3.61505403529,3.05311331772e-16],
 [0.0,0.0,-0.800891617427,-0.800891617427,0.0,0.199108382573,1.80089161743,-1.80089161743,-0.199108382573,1.60178323485],
 [0.0,0.0,-0.800891617427,-0.800891617427,0.0,1.80089161743,0.199108382573,-0.199108382573,-1.80089161743,1.60178323485],
 [0.0,0.0,0.800891617427,-0.800891617427,0.0,0.199108382573,0.199108382573,-0.199108382573,-0.199108382573,-2.22044604925e-16],
 [0.0,0.0,0.800891617427,0.800891617427,0.0,1.80089161743,0.199108382573,-0.199108382573,-1.80089161743,-1.60178323485],
 [0.0,0.0,0.800891617427,0.800891617427,0.0,0.199108382573,1.80089161743,-1.80089161743,-0.199108382573,-1.60178323485],
 [0.0,0.0,-0.800891617427,0.800891617427,0.0,1.80089161743,1.80089161743,-1.80089161743,-1.80089161743,-3.05311331772e-16],
 [0.0,0.0,-0.265078210406,-0.265078210406,0.0,0.734921789594,1.26507821041,-1.26507821041,-0.734921789594,0.530156420812],
 [0.0,0.0,-0.265078210406,-0.265078210406,0.0,1.26507821041,0.734921789594,-0.734921789594,-1.26507821041,0.530156420812],
 [0.0,0.0,0.265078210406,-0.265078210406,0.0,0.734921789594,0.734921789594,-0.734921789594,-0.734921789594,-2.22044604925e-16],
 [0.0,0.0,0.265078210406,0.265078210406,0.0,1.26507821041,0.734921789594,-0.734921789594,-1.26507821041,-0.530156420812],
 [0.0,0.0,0.265078210406,0.265078210406,0.0,0.734921789594,1.26507821041,-1.26507821041,-0.734921789594,-0.530156420812],
 [0.0,0.0,-0.265078210406,0.265078210406,0.0,1.26507821041,1.26507821041,-1.26507821041,-1.26507821041,-3.33066907388e-16],
 [0.0,0.0,-0.0723956424114,-1.05312013344,0.0,0.927604357589,0.0916711513793,-0.0916711513793,-0.927604357589,1.12551577585],
 [0.0,0.0,-0.0723956424114,-1.05312013344,0.0,0.0916711513793,0.927604357589,-0.927604357589,-0.0916711513793,1.12551577585],
 [0.0,0.0,-0.908328848621,-1.05312013344,0.0,0.927604357589,0.927604357589,-0.927604357589,-0.927604357589,1.96144898206],
 [0.0,0.0,-0.0723956424114,0.908328848621,0.0,0.927604357589,2.05312013344,-2.05312013344,-0.927604357589,-0.835933206209],
 [0.0,0.0,-0.0723956424114,0.908328848621,0.0,2.05312013344,0.927604357589,-0.927604357589,-2.05312013344,-0.835933206209],
 [0.0,0.0,1.05312013344,0.908328848621,0.0,0.927604357589,0.927604357589,-0.927604357589,-0.927604357589,-1.96144898206],
 [0.0,0.0,1.05312013344,0.0723956424114,0.0,0.0916711513793,0.927604357589,-0.927604357589,-0.0916711513793,-1.12551577585],
 [0.0,0.0,-0.0723956424114,0.0723956424114,0.0,2.05312013344,0.0916711513793,-0.0916711513793,-2.05312013344,-3.33066907388e-16],
 [0.0,0.0,-0.908328848621,0.0723956424114,0.0,0.927604357589,2.05312013344,-2.05312013344,-0.927604357589,0.835933206209],
 [0.0,0.0,-0.908328848621,0.0723956424114,0.0,2.05312013344,0.927604357589,-0.927604357589,-2.05312013344,0.835933206209],
 [0.0,0.0,1.05312013344,0.0723956424114,0.0,0.927604357589,0.0916711513793,-0.0916711513793,-0.927604357589,-1.12551577585],
 [0.0,0.0,-0.0723956424114,0.0723956424114,0.0,0.0916711513793,2.05312013344,-2.05312013344,-0.0916711513793,-3.33066907388e-16],
 [0.0,0.0,-0.848119806113,0.225014099005,0.0,0.151880193887,2.92125371123,-2.92125371123,-0.151880193887,0.623105707108],
 [0.0,0.0,-0.848119806113,0.225014099005,0.0,2.92125371123,0.151880193887,-0.151880193887,-2.92125371123,0.623105707108],
 [0.0,0.0,1.92125371123,0.225014099005,0.0,0.151880193887,0.151880193887,-0.151880193887,-0.151880193887,-2.14626781023],
 [0.0,0.0,-0.848119806113,-1.92125371123,0.0,0.151880193887,0.774985900995,-0.774985900995,-0.151880193887,2.76937351734],
 [0.0,0.0,-0.848119806113,-1.92125371123,0.0,0.774985900995,0.151880193887,-0.151880193887,-0.774985900995,2.76937351734],
 [0.0,0.0,-0.225014099005,-1.92125371123,0.0,0.151880193887,0.151880193887,-0.151880193887,-0.151880193887,2.14626781023],
 [0.0,0.0,-0.225014099005,0.848119806113,0.0,2.92125371123,0.151880193887,-0.151880193887,-2.92125371123,-0.623105707108],
 [0.0,0.0,-0.848119806113,0.848119806113,0.0,0.774985900995,2.92125371123,-2.92125371123,-0.774985900995,2.77555756156e-17],
 [0.0,0.0,1.92125371123,0.848119806113,0.0,0.151880193887,0.774985900995,-0.774985900995,-0.151880193887,-2.76937351734],
 [0.0,0.0,1.92125371123,0.848119806113,0.0,0.774985900995,0.151880193887,-0.151880193887,-0.774985900995,-2.76937351734],
 [0.0,0.0,-0.225014099005,0.848119806113,0.0,0.151880193887,2.92125371123,-2.92125371123,-0.151880193887,-0.623105707108],
 [0.0,0.0,-0.848119806113,0.848119806113,0.0,2.92125371123,0.774985900995,-0.774985900995,-2.92125371123,2.77555756156e-17]
 ])

 def tet11(_self):
  _self.tet4()

  _self.phiJ = np.array([
 [0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000 ],
 [0.186174458992371, -0.054039483181334, -0.054039483181334, -0.054039483181334, 0.233030122142538, -0.016872189417200, 0.233030122142538, 0.233030122142538, -0.016872189417200, -0.016872189417200, 0.327470192375619 ],
 [-0.054039483181334, -0.054039483181334, -0.054039483181334, 0.186174458992371, -0.016872189417200, -0.016872189417200, -0.016872189417200, 0.233030122142538, 0.233030122142538, 0.233030122142538, 0.327470192375619 ],
 [-0.054039483181334, -0.054039483181334, 0.186174458992371, -0.054039483181334, -0.016872189417200, 0.233030122142538, 0.233030122142538, -0.016872189417200, -0.016872189417200, 0.233030122142538, 0.327470192375619 ],
 [-0.054039483181334, 0.186174458992371, -0.054039483181334, -0.054039483181334, 0.233030122142538, 0.233030122142538, -0.016872189417200, -0.016872189417200, 0.233030122142538, -0.016872189417200, 0.327470192375619 ],
 [0.730768136500267, -0.029066042108423, -0.029066042108423, -0.029066042108423, 0.114057334127236, 0.002206834306455, 0.114057334127236, 0.114057334127236, 0.002206834306455, 0.002206834306455, 0.007637484523929 ],
 [-0.029066042108423, -0.029066042108423, -0.029066042108423, 0.730768136500267, 0.002206834306455, 0.002206834306455, 0.002206834306455, 0.114057334127236, 0.114057334127236, 0.114057334127236, 0.007637484523929 ],
 [-0.029066042108423, -0.029066042108423, 0.730768136500267, -0.029066042108423, 0.002206834306455, 0.114057334127236, 0.114057334127236, 0.002206834306455, 0.002206834306455, 0.114057334127236, 0.007637484523929 ],
 [-0.029066042108423, 0.730768136500267, -0.029066042108423, -0.029066042108423, 0.114057334127236, 0.114057334127236, 0.002206834306455, 0.002206834306455, 0.114057334127236, 0.002206834306455, 0.007637484523929 ],
 [-0.028749786922007, -0.028749786922007, -0.028749786922007, -0.028749786922007, 0.057499573844015, -0.022232543437104, 0.057499573844015, 0.778659073989772, 0.057499573844015, 0.057499573844015, 0.128574321759303 ],
 [-0.028749786922007, -0.028749786922007, -0.028749786922007, -0.028749786922007, 0.057499573844015, 0.057499573844015, -0.022232543437104, 0.057499573844015, 0.778659073989772, 0.057499573844015, 0.128574321759303 ],
 [-0.028749786922007, -0.028749786922007, -0.028749786922007, -0.028749786922007, -0.022232543437104, 0.057499573844015, 0.057499573844015, 0.057499573844015, 0.057499573844015, 0.778659073989772, 0.128574321759303 ],
 [-0.028749786922007, -0.028749786922007, -0.028749786922007, -0.028749786922007, 0.057499573844015, 0.778659073989772, 0.057499573844015, -0.022232543437104, 0.057499573844015, 0.057499573844015, 0.128574321759303 ],
 [-0.028749786922007, -0.028749786922007, -0.028749786922007, -0.028749786922007, 0.057499573844015, 0.057499573844015, 0.778659073989772, 0.057499573844015, -0.022232543437104, 0.057499573844015, 0.128574321759303 ],
 [-0.028749786922007, -0.028749786922007, -0.028749786922007, -0.028749786922007, 0.778659073989772, 0.057499573844015, 0.057499573844015, 0.057499573844015, 0.057499573844015, -0.022232543437104, 0.128574321759303 ],
 [-0.008166135320475, -0.008166135320475, -0.008166135320475, -0.008166135320475, 0.016332270640950, -0.081073605745979, 0.016332270640950, 0.184004604659821, 0.016332270640950, 0.016332270640950, 0.864404459804258 ],
 [-0.008166135320475, -0.008166135320475, -0.008166135320475, -0.008166135320475, 0.016332270640950, 0.016332270640950, -0.081073605745979, 0.016332270640950, 0.184004604659821, 0.016332270640950, 0.864404459804258 ],
 [-0.008166135320475, -0.008166135320475, -0.008166135320475, -0.008166135320475, -0.081073605745979, 0.016332270640950, 0.016332270640950, 0.016332270640950, 0.016332270640950, 0.184004604659821, 0.864404459804258 ],
 [-0.008166135320475, -0.008166135320475, -0.008166135320475, -0.008166135320475, 0.016332270640950, 0.184004604659821, 0.016332270640950, -0.081073605745979, 0.016332270640950, 0.016332270640950, 0.864404459804258 ],
 [-0.008166135320475, -0.008166135320475, -0.008166135320475, -0.008166135320475, 0.016332270640950, 0.016332270640950, 0.184004604659821, 0.016332270640950, -0.081073605745979, 0.016332270640950, 0.864404459804258 ],
 [-0.008166135320475, -0.008166135320475, -0.008166135320475, -0.008166135320475, 0.184004604659821, 0.016332270640950, 0.016332270640950, 0.016332270640950, 0.016332270640950, -0.081073605745979, 0.864404459804258 ],
 [-0.001623976787398, -0.104101497811962, -0.104101497811962, 0.033876112991032, -0.019228082244894, 0.174625738938304, -0.019228082244894, 0.006566249522144, 0.435634073492740, 0.435634073492740, 0.161946888464149 ],
 [-0.104101497811962, -0.001623976787398, -0.104101497811962, 0.033876112991032, -0.019228082244894, -0.019228082244894, 0.174625738938304, 0.435634073492740, 0.006566249522144, 0.435634073492740, 0.161946888464149 ],
 [-0.104101497811962, -0.104101497811962, -0.001623976787398, 0.033876112991032, 0.174625738938304, -0.019228082244894, -0.019228082244894, 0.435634073492740, 0.435634073492740, 0.006566249522144, 0.161946888464149 ],
 [0.033876112991032, -0.104101497811962, -0.104101497811962, -0.001623976787398, 0.435634073492740, 0.174625738938304, 0.435634073492740, 0.006566249522144, -0.019228082244894, -0.019228082244894, 0.161946888464148 ],
 [-0.104101497811962, 0.033876112991032, -0.104101497811962, -0.001623976787398, 0.435634073492740, 0.435634073492740, 0.174625738938304, -0.019228082244894, 0.006566249522144, -0.019228082244894, 0.161946888464148 ],
 [-0.104101497811962, -0.104101497811962, 0.033876112991032, -0.001623976787398, 0.174625738938304, 0.435634073492740, 0.435634073492740, -0.019228082244894, -0.019228082244894, 0.006566249522144, 0.161946888464148 ],
 [-0.104101497811962, -0.001623976787398, 0.033876112991032, -0.104101497811962, -0.019228082244894, 0.006566249522144, 0.435634073492740, 0.174625738938304, -0.019228082244894, 0.435634073492740, 0.161946888464149 ],
 [-0.001623976787398, 0.033876112991032, -0.104101497811962, -0.104101497811962, 0.006566249522144, 0.435634073492740, -0.019228082244894, -0.019228082244894, 0.435634073492740, 0.174625738938304, 0.161946888464149 ],
 [0.033876112991032, -0.104101497811962, -0.001623976787398, -0.104101497811962, 0.435634073492740, -0.019228082244894, 0.006566249522144, 0.435634073492740, 0.174625738938304, -0.019228082244894, 0.161946888464149 ],
 [-0.104101497811962, 0.033876112991032, -0.001623976787398, -0.104101497811962, 0.435634073492740, 0.006566249522144, -0.019228082244894, 0.174625738938304, 0.435634073492740, -0.019228082244894, 0.161946888464149 ],
 [-0.001623976787398, -0.104101497811962, 0.033876112991032, -0.104101497811962, -0.019228082244894, 0.435634073492740, 0.006566249522144, -0.019228082244894, 0.174625738938304, 0.435634073492740, 0.161946888464149 ],
 [0.033876112991032, -0.001623976787398, -0.104101497811962, -0.104101497811962, 0.006566249522144, -0.019228082244894, 0.435634073492740, 0.435634073492740, -0.019228082244894, 0.174625738938304, 0.161946888464149 ],
 [0.342929906192652, -0.028558670981725, -0.028558670981725, -0.112143153577936, 0.097864288357523, -0.007288958332583, 0.097864288357523, 0.552926753201936, 0.016370395569377, 0.016370395569377, 0.052223426625580 ],
 [-0.028558670981725, 0.342929906192652, -0.028558670981725, -0.112143153577936, 0.097864288357523, 0.097864288357523, -0.007288958332583, 0.016370395569377, 0.552926753201936, 0.016370395569377, 0.052223426625580 ],
 [-0.028558670981725, -0.028558670981725, 0.342929906192652, -0.112143153577936, -0.007288958332583, 0.097864288357523, 0.097864288357523, 0.016370395569377, 0.016370395569377, 0.552926753201936, 0.052223426625580 ],
 [-0.112143153577936, -0.028558670981725, -0.028558670981725, 0.342929906192652, 0.016370395569377, -0.007288958332583, 0.016370395569377, 0.552926753201936, 0.097864288357523, 0.097864288357523, 0.052223426625580 ],
 [-0.028558670981725, -0.112143153577936, -0.028558670981725, 0.342929906192652, 0.016370395569377, 0.016370395569377, -0.007288958332583, 0.097864288357523, 0.552926753201936, 0.097864288357523, 0.052223426625580 ],
 [-0.028558670981725, -0.028558670981725, -0.112143153577936, 0.342929906192652, -0.007288958332583, 0.016370395569377, 0.016370395569377, 0.097864288357523, 0.097864288357523, 0.552926753201936, 0.052223426625580 ],
 [-0.028558670981725, 0.342929906192652, -0.112143153577936, -0.028558670981725, 0.097864288357523, 0.552926753201936, 0.016370395569377, -0.007288958332583, 0.097864288357523, 0.016370395569377, 0.052223426625581 ],
 [0.342929906192652, -0.112143153577936, -0.028558670981725, -0.028558670981725, 0.552926753201936, 0.016370395569377, 0.097864288357523, 0.097864288357523, 0.016370395569377, -0.007288958332583, 0.052223426625581 ],
 [-0.112143153577936, -0.028558670981725, 0.342929906192652, -0.028558670981725, 0.016370395569377, 0.097864288357523, 0.552926753201936, 0.016370395569377, -0.007288958332583, 0.097864288357523, 0.052223426625581 ],
 [-0.028558670981725, -0.112143153577936, 0.342929906192652, -0.028558670981725, 0.016370395569377, 0.552926753201936, 0.097864288357523, -0.007288958332583, 0.016370395569377, 0.097864288357523, 0.052223426625581 ],
 [0.342929906192652, -0.028558670981725, -0.112143153577936, -0.028558670981725, 0.097864288357523, 0.016370395569377, 0.552926753201936, 0.097864288357523, -0.007288958332583, 0.016370395569377, 0.052223426625581 ],
 [-0.112143153577936, 0.342929906192652, -0.028558670981725, -0.028558670981725, 0.552926753201936, 0.097864288357523, 0.016370395569377, 0.016370395569377, 0.097864288357523, -0.007288958332583, 0.052223426625581 ]
 ])

  _self.dphiJdxi = np.array([
 [0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000, 0.000000000000000, 1.000000000000000, 0.000000000000000, -1.000000000000000, -1.000000000000000, 0.000000000000000 ],
 [1.215506507562829, -0.254842253637503, -0.254842253637503, 0.235274000095941, 1.019568253541562, 0.509684507275006, 1.019568253541562, -1.450780507658770, -0.000199238991550, -0.000199238991550, -2.038738029100023 ],
 [-0.235274000095941, 0.254842253637503, 0.254842253637503, -1.215506507562829, 0.000199238991550, -0.509684507275006, 0.000199238991550, 1.450780507658770, -1.019568253541562, -1.019568253541562, 2.038738029100023 ],
 [-0.490116253733444, 0.000000000000000, 0.000000000000000, 0.490116253733444, 0.509883746266556, 0.000000000000000, 2.470348761200332, 0.000000000000000, -0.509883746266556, -2.470348761200332, 0.000000000000000 ],
 [-0.490116253733444, 0.000000000000000, 0.000000000000000, 0.490116253733444, 2.470348761200332, 0.000000000000000, 0.509883746266556, 0.000000000000000, -2.470348761200332, -0.509883746266556, 0.000000000000000 ],
 [2.586349769583757, -0.028704265704656, -0.028704265704656, 0.842980412724815, 0.185723852979841, 0.057408531409312, 0.185723852979841, -3.429330182308571, -0.070906790161218, -0.070906790161218, -0.229634125637247 ],
 [-0.842980412724815, 0.028704265704656, 0.028704265704656, -2.586349769583757, 0.070906790161218, -0.057408531409312, 0.070906790161218, 3.429330182308572, -0.185723852979841, -0.185723852979841, 0.229634125637247 ],
 [-0.871684678429471, -0.000000000000000, -0.000000000000000, 0.871684678429471, 0.128315321570529, -0.000000000000000, 3.615054035288412, -0.000000000000000, -0.128315321570529, -3.615054035288412, -0.000000000000000 ],
 [-0.871684678429471, 0.000000000000000, 0.000000000000000, 0.871684678429471, 3.615054035288412, -0.000000000000000, 0.128315321570529, 0.000000000000000, -3.615054035288412, -0.128315321570529, 0.000000000000001 ],
 [0.800891617426876, -0.000000000000000, -0.000000000000000, -0.800891617426876, 0.199108382573124, 0.000000000000000, 0.199108382573124, -0.000000000000000, -0.199108382573124, -0.199108382573124, -0.000000000000000 ],
 [-0.513713814124348, 0.287177803302528, 0.287177803302528, -0.513713814124347, 1.226536010821819, -0.574355606605057, -0.375247224031933, 1.027427628248695, -2.375247224031933, -0.773463989178181, 2.297422426420227 ],
 [-0.513713814124348, 0.287177803302528, 0.287177803302528, -0.513713814124347, -0.375247224031933, -0.574355606605057, 1.226536010821819, 1.027427628248695, -0.773463989178181, -2.375247224031933, 2.297422426420227 ],
 [-0.800891617426876, 0.000000000000000, 0.000000000000000, 0.800891617426876, 1.800891617426876, -0.000000000000000, 1.800891617426876, -0.000000000000000, -1.800891617426877, -1.800891617426877, 0.000000000000002 ],
 [0.513713814124348, -0.287177803302528, -0.287177803302528, 0.513713814124347, 0.773463989178181, 0.574355606605057, 2.375247224031933, -1.027427628248695, 0.375247224031933, -1.226536010821819, -2.297422426420227 ],
 [0.513713814124348, -0.287177803302529, -0.287177803302529, 0.513713814124348, 2.375247224031933, 0.574355606605057, 0.773463989178181, -1.027427628248695, -1.226536010821819, 0.375247224031933, -2.297422426420228 ],
 [0.265078210405800, -0.000000000000000, -0.000000000000000, -0.265078210405800, 0.734921789594200, 0.000000000000000, 0.734921789594200, -0.000000000000000, -0.734921789594200, -0.734921789594200, -0.000000000000000 ],
 [-0.018626106840630, 0.246452103565170, 0.246452103565170, -0.018626106840630, 0.772174003275460, -0.492904207130340, 0.242017582463859, 0.037252213681260, -1.757982417536141, -1.227825996724540, 1.971616828521362 ],
 [-0.018626106840630, 0.246452103565170, 0.246452103565170, -0.018626106840630, 0.242017582463859, -0.492904207130340, 0.772174003275460, 0.037252213681260, -1.227825996724540, -1.757982417536141, 1.971616828521362 ],
 [-0.265078210405800, 0.000000000000000, 0.000000000000000, 0.265078210405800, 1.265078210405800, -0.000000000000000, 1.265078210405800, -0.000000000000000, -1.265078210405801, -1.265078210405801, 0.000000000000001 ],
 [0.018626106840630, -0.246452103565170, -0.246452103565170, 0.018626106840630, 1.227825996724540, 0.492904207130341, 1.757982417536141, -0.037252213681260, -0.242017582463859, -0.772174003275460, -1.971616828521362 ],
 [0.018626106840630, -0.246452103565170, -0.246452103565170, 0.018626106840630, 1.757982417536141, 0.492904207130341, 1.227825996724541, -0.037252213681260, -0.772174003275459, -0.242017582463859, -1.971616828521364 ],
 [-0.064464613091981, 0.843864235528750, 0.843864235528750, -0.209255897914774, -0.760124113468897, -1.687728471057500, -0.760124113468897, 0.273720511006756, -2.615332828646104, -2.615332828646104, 6.750913884230001 ],
 [-0.024541773315016, 0.047853869096381, 0.047853869096381, -1.005266264347144, -0.004036586813493, -0.095707738192762, 0.831896619395842, 1.029808037662159, -0.187378889572030, -1.023312095781365, 0.382830952771046 ],
 [-0.024541773315016, 0.047853869096381, 0.047853869096381, -1.005266264347144, 0.831896619395842, -0.095707738192762, -0.004036586813493, 1.029808037662159, -1.023312095781365, -0.187378889572030, 0.382830952771046 ],
 [0.209255897914774, -0.843864235528750, -0.843864235528750, 0.064464613091982, 2.615332828646104, 1.687728471057500, 2.615332828646104, -0.273720511006756, 0.760124113468897, 0.760124113468897, -6.750913884230002 ],
 [-0.868406008843766, -0.796010366432370, -0.796010366432370, 0.112318482188362, 3.645140866308264, 1.592020732864739, 2.519625090453343, 0.756087526655404, -0.461099400578785, 0.664416375276136, -6.368082931458957 ],
 [-0.868406008843766, -0.796010366432370, -0.796010366432370, 0.112318482188362, 2.519625090453343, 1.592020732864739, 3.645140866308264, 0.756087526655404, 0.664416375276136, -0.461099400578785, -6.368082931458957 ],
 [-0.072395642411396, -0.000000000000000, -0.000000000000000, 0.072395642411396, 0.091671151379268, -0.000000000000000, 2.053120133443525, 0.000000000000000, -0.091671151379268, -2.053120133443525, -0.000000000000000 ],
 [-0.112318482188363, 0.796010366432369, 0.796010366432369, 0.868406008843766, 0.461099400578787, -1.592020732864738, -0.664416375276134, -0.756087526655403, -3.645140866308263, -2.519625090453342, 6.368082931458952 ],
 [1.005266264347144, -0.047853869096381, -0.047853869096381, 0.024541773315016, 1.023312095781365, 0.095707738192762, 0.187378889572030, -1.029808037662160, -0.831896619395842, 0.004036586813493, -0.382830952771047 ],
 [-0.072395642411396, -0.000000000000000, -0.000000000000000, 0.072395642411397, 2.053120133443525, 0.000000000000000, 0.091671151379268, -0.000000000000000, -2.053120133443525, -0.091671151379268, -0.000000000000000 ],
 [-0.112318482188362, 0.796010366432369, 0.796010366432369, 0.868406008843766, -0.664416375276135, -1.592020732864739, 0.461099400578786, -0.756087526655404, -2.519625090453343, -3.645140866308263, 6.368082931458956 ],
 [1.005266264347144, -0.047853869096381, -0.047853869096381, 0.024541773315016, 0.187378889572030, 0.095707738192762, 1.023312095781365, -1.029808037662160, 0.004036586813493, -0.831896619395842, -0.382830952771047 ],
 [1.896499094755562, -0.024754616474591, -0.024754616474591, 0.200259482530191, 0.201389426836497, 0.049509232949182, 0.201389426836497, -2.096758577285753, -0.102370960938132, -0.102370960938132, -0.198036931796730 ],
 [-0.709889855329868, 0.138229950782817, 0.138229950782817, 0.363244049787600, 2.644793809664519, -0.276459901565634, -0.124579707678320, 0.346645805542269, -3.197713612795788, -0.428340095452949, 1.105839606262538 ],
 [-0.709889855329868, 0.138229950782817, 0.138229950782817, 0.363244049787600, -0.124579707678320, -0.276459901565634, 2.644793809664519, 0.346645805542269, -0.428340095452949, -3.197713612795788, 1.105839606262538 ],
 [-0.200259482530191, 0.024754616474591, 0.024754616474591, -1.896499094755562, 0.102370960938132, -0.049509232949182, 0.102370960938132, 2.096758577285753, -0.201389426836497, -0.201389426836497, 0.198036931796730 ],
 [-0.685135238855277, 0.162984567257408, 0.162984567257408, -1.758269143972745, 0.449016766480401, -0.325969134514817, -0.174088940627503, 2.443404382828022, -1.100955035510035, -0.477849328402131, 1.303876538059268 ],
 [-0.685135238855277, 0.162984567257408, 0.162984567257408, -1.758269143972745, -0.174088940627503, -0.325969134514817, 0.449016766480401, 2.443404382828022, -0.477849328402131, -1.100955035510035, 1.303876538059268 ],
 [-0.848119806112686, 0.000000000000000, 0.000000000000000, 0.848119806112686, 2.921253711230154, -0.000000000000000, 0.774985900995217, -0.000000000000000, -2.921253711230154, -0.774985900995218, 0.000000000000000 ],
 [1.758269143972745, -0.162984567257408, -0.162984567257408, 0.685135238855277, 1.100955035510035, 0.325969134514817, 0.477849328402131, -2.443404382828022, -0.449016766480401, 0.174088940627503, -1.303876538059268 ],
 [-0.363244049787600, -0.138229950782817, -0.138229950782817, 0.709889855329868, 0.428340095452949, 0.276459901565634, 3.197713612795788, -0.346645805542269, 0.124579707678320, -2.644793809664519, -1.105839606262538 ],
 [-0.848119806112686, -0.000000000000000, -0.000000000000000, 0.848119806112686, 0.774985900995217, -0.000000000000000, 2.921253711230154, -0.000000000000000, -0.774985900995218, -2.921253711230154, -0.000000000000001 ],
 [1.758269143972745, -0.162984567257408, -0.162984567257408, 0.685135238855277, 0.477849328402131, 0.325969134514817, 1.100955035510035, -2.443404382828022, 0.174088940627503, -0.449016766480401, -1.303876538059268 ],
 [-0.363244049787600, -0.138229950782817, -0.138229950782817, 0.709889855329868, 3.197713612795788, 0.276459901565634, 0.428340095452949, -0.346645805542269, -2.644793809664519, 0.124579707678320, -1.105839606262538 ]
 ])

  _self.dphiJdeta = np.array([
 [0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000, 1.000000000000000, 0.000000000000000, -1.000000000000000, 0.000000000000000, -1.000000000000000, 0.000000000000000 ],
 [0.000000000000000, -0.490116253733444, 0.000000000000000, 0.490116253733444, 2.470348761200332, 0.509883746266556, 0.000000000000000, -2.470348761200332, 0.000000000000000, -0.509883746266556, 0.000000000000000 ],
 [0.254842253637503, -0.235274000095941, 0.254842253637503, -1.215506507562829, 0.000199238991550, 0.000199238991550, -0.509684507275006, -1.019568253541562, 1.450780507658770, -1.019568253541562, 2.038738029100023 ],
 [0.000000000000000, -0.490116253733444, 0.000000000000000, 0.490116253733444, 0.509883746266556, 2.470348761200332, 0.000000000000000, -0.509883746266556, 0.000000000000000, -2.470348761200332, 0.000000000000000 ],
 [-0.254842253637503, 1.215506507562829, -0.254842253637503, 0.235274000095941, 1.019568253541562, 1.019568253541562, 0.509684507275006, -0.000199238991550, -1.450780507658770, -0.000199238991550, -2.038738029100023 ],
 [-0.000000000000000, -0.871684678429471, -0.000000000000000, 0.871684678429471, 3.615054035288412, 0.128315321570529, -0.000000000000000, -3.615054035288412, -0.000000000000001, -0.128315321570529, -0.000000000000001 ],
 [0.028704265704656, -0.842980412724815, 0.028704265704656, -2.586349769583757, 0.070906790161218, 0.070906790161218, -0.057408531409312, -0.185723852979841, 3.429330182308572, -0.185723852979841, 0.229634125637247 ],
 [-0.000000000000000, -0.871684678429471, -0.000000000000000, 0.871684678429471, 0.128315321570529, 3.615054035288412, -0.000000000000000, -0.128315321570529, -0.000000000000000, -3.615054035288412, -0.000000000000000 ],
 [-0.028704265704656, 2.586349769583757, -0.028704265704656, 0.842980412724815, 0.185723852979841, 0.185723852979841, 0.057408531409312, -0.070906790161218, -3.429330182308571, -0.070906790161218, -0.229634125637247 ],
 [0.287177803302528, -0.513713814124348, 0.287177803302528, -0.513713814124347, 1.226536010821819, -0.375247224031933, -0.574355606605057, -2.375247224031933, 1.027427628248695, -0.773463989178181, 2.297422426420227 ],
 [-0.000000000000000, 0.800891617426876, -0.000000000000000, -0.800891617426876, 0.199108382573124, 0.199108382573124, -0.000000000000000, -0.199108382573124, -0.000000000000000, -0.199108382573124, -0.000000000000000 ],
 [0.287177803302528, -0.513713814124348, 0.287177803302528, -0.513713814124347, -0.375247224031933, 1.226536010821819, -0.574355606605057, -0.773463989178181, 1.027427628248695, -2.375247224031933, 2.297422426420227 ],
 [-0.287177803302528, 0.513713814124348, -0.287177803302528, 0.513713814124347, 0.773463989178181, 2.375247224031933, 0.574355606605057, 0.375247224031933, -1.027427628248695, -1.226536010821819, -2.297422426420227 ],
 [-0.000000000000000, -0.800891617426876, -0.000000000000000, 0.800891617426876, 1.800891617426876, 1.800891617426876, -0.000000000000000, -1.800891617426877, -0.000000000000000, -1.800891617426877, -0.000000000000008 ],
 [-0.287177803302529, 0.513713814124348, -0.287177803302529, 0.513713814124348, 2.375247224031933, 0.773463989178181, 0.574355606605057, -1.226536010821819, -1.027427628248695, 0.375247224031933, -2.297422426420228 ],
 [0.246452103565170, -0.018626106840630, 0.246452103565170, -0.018626106840630, 0.772174003275460, 0.242017582463859, -0.492904207130340, -1.757982417536141, 0.037252213681260, -1.227825996724540, 1.971616828521362 ],
 [-0.000000000000000, 0.265078210405800, -0.000000000000000, -0.265078210405800, 0.734921789594200, 0.734921789594200, -0.000000000000000, -0.734921789594200, -0.000000000000000, -0.734921789594200, -0.000000000000000 ],
 [0.246452103565170, -0.018626106840630, 0.246452103565170, -0.018626106840630, 0.242017582463859, 0.772174003275460, -0.492904207130340, -1.227825996724540, 0.037252213681260, -1.757982417536141, 1.971616828521362 ],
 [-0.246452103565170, 0.018626106840630, -0.246452103565170, 0.018626106840630, 1.227825996724540, 1.757982417536141, 0.492904207130341, -0.242017582463859, -0.037252213681260, -0.772174003275460, -1.971616828521362 ],
 [-0.000000000000000, -0.265078210405800, -0.000000000000000, 0.265078210405800, 1.265078210405800, 1.265078210405800, -0.000000000000000, -1.265078210405801, -0.000000000000000, -1.265078210405801, -0.000000000000000 ],
 [-0.246452103565170, 0.018626106840630, -0.246452103565170, 0.018626106840630, 1.757982417536141, 1.227825996724541, 0.492904207130341, -0.772174003275459, -0.037252213681260, -0.242017582463859, -1.971616828521364 ],
 [0.047853869096381, -0.024541773315016, 0.047853869096381, -1.005266264347144, -0.004036586813493, 0.831896619395842, -0.095707738192762, -0.187378889572030, 1.029808037662159, -1.023312095781365, 0.382830952771046 ],
 [0.843864235528750, -0.064464613091981, 0.843864235528750, -0.209255897914774, -0.760124113468897, -0.760124113468897, -1.687728471057500, -2.615332828646104, 0.273720511006756, -2.615332828646104, 6.750913884230001 ],
 [0.047853869096381, -0.024541773315016, 0.047853869096381, -1.005266264347144, 0.831896619395842, -0.004036586813493, -0.095707738192762, -1.023312095781365, 1.029808037662159, -0.187378889572030, 0.382830952771046 ],
 [-0.796010366432370, -0.868406008843766, -0.796010366432370, 0.112318482188362, 3.645140866308264, 2.519625090453343, 1.592020732864739, -0.461099400578785, 0.756087526655404, 0.664416375276136, -6.368082931458957 ],
 [-0.843864235528750, 0.209255897914774, -0.843864235528750, 0.064464613091982, 2.615332828646104, 2.615332828646104, 1.687728471057500, 0.760124113468897, -0.273720511006756, 0.760124113468897, -6.750913884230002 ],
 [-0.796010366432370, -0.868406008843766, -0.796010366432370, 0.112318482188362, 2.519625090453343, 3.645140866308264, 1.592020732864739, 0.664416375276136, 0.756087526655404, -0.461099400578785, -6.368082931458957 ],
 [0.796010366432369, -0.112318482188362, 0.796010366432369, 0.868406008843766, -0.664416375276135, 0.461099400578786, -1.592020732864739, -2.519625090453343, -0.756087526655404, -3.645140866308263, 6.368082931458956 ],
 [-0.047853869096381, 1.005266264347144, -0.047853869096381, 0.024541773315016, 0.187378889572030, 1.023312095781365, 0.095707738192762, 0.004036586813493, -1.029808037662160, -0.831896619395842, -0.382830952771047 ],
 [-0.000000000000000, -0.072395642411396, -0.000000000000000, 0.072395642411397, 2.053120133443525, 0.091671151379268, 0.000000000000000, -2.053120133443525, -0.000000000000000, -0.091671151379268, -0.000000000000000 ],
 [-0.047853869096381, 1.005266264347144, -0.047853869096381, 0.024541773315016, 1.023312095781365, 0.187378889572030, 0.095707738192762, -0.831896619395842, -1.029808037662160, 0.004036586813493, -0.382830952771047 ],
 [-0.000000000000000, -0.072395642411396, -0.000000000000000, 0.072395642411396, 0.091671151379268, 2.053120133443525, -0.000000000000000, -0.091671151379268, -0.000000000000000, -2.053120133443525, -0.000000000000000 ],
 [0.796010366432369, -0.112318482188363, 0.796010366432369, 0.868406008843766, 0.461099400578787, -0.664416375276134, -1.592020732864738, -3.645140866308263, -0.756087526655403, -2.519625090453342, 6.368082931458952 ],
 [0.138229950782817, -0.709889855329868, 0.138229950782817, 0.363244049787600, 2.644793809664519, -0.124579707678320, -0.276459901565634, -3.197713612795788, 0.346645805542269, -0.428340095452949, 1.105839606262538 ],
 [-0.024754616474591, 1.896499094755562, -0.024754616474591, 0.200259482530191, 0.201389426836497, 0.201389426836497, 0.049509232949182, -0.102370960938132, -2.096758577285753, -0.102370960938132, -0.198036931796730 ],
 [0.138229950782817, -0.709889855329868, 0.138229950782817, 0.363244049787600, -0.124579707678320, 2.644793809664519, -0.276459901565634, -0.428340095452949, 0.346645805542269, -3.197713612795788, 1.105839606262538 ],
 [0.162984567257408, -0.685135238855277, 0.162984567257408, -1.758269143972745, 0.449016766480401, -0.174088940627503, -0.325969134514817, -1.100955035510035, 2.443404382828022, -0.477849328402131, 1.303876538059268 ],
 [0.024754616474591, -0.200259482530191, 0.024754616474591, -1.896499094755562, 0.102370960938132, 0.102370960938132, -0.049509232949182, -0.201389426836497, 2.096758577285753, -0.201389426836497, 0.198036931796730 ],
 [0.162984567257408, -0.685135238855277, 0.162984567257408, -1.758269143972745, -0.174088940627503, 0.449016766480401, -0.325969134514817, -0.477849328402131, 2.443404382828022, -1.100955035510035, 1.303876538059268 ],
 [-0.162984567257408, 1.758269143972745, -0.162984567257408, 0.685135238855277, 0.477849328402131, 1.100955035510035, 0.325969134514817, 0.174088940627503, -2.443404382828022, -0.449016766480401, -1.303876538059268 ],
 [-0.138229950782817, -0.363244049787600, -0.138229950782817, 0.709889855329868, 3.197713612795788, 0.428340095452949, 0.276459901565634, -2.644793809664519, -0.346645805542269, 0.124579707678320, -1.105839606262538 ],
 [-0.000000000000000, -0.848119806112686, -0.000000000000000, 0.848119806112686, 0.774985900995217, 2.921253711230154, -0.000000000000000, -0.774985900995218, -0.000000000000000, -2.921253711230154, -0.000000000000001 ],
 [-0.138229950782817, -0.363244049787600, -0.138229950782817, 0.709889855329868, 0.428340095452949, 3.197713612795788, 0.276459901565634, 0.124579707678320, -0.346645805542269, -2.644793809664519, -1.105839606262538 ],
 [-0.000000000000000, -0.848119806112686, -0.000000000000000, 0.848119806112686, 2.921253711230154, 0.774985900995217, -0.000000000000000, -2.921253711230154, -0.000000000000000, -0.774985900995218, -0.000000000000001 ],
 [-0.162984567257408, 1.758269143972745, -0.162984567257408, 0.685135238855277, 1.100955035510035, 0.477849328402131, 0.325969134514817, -0.449016766480401, -2.443404382828022, 0.174088940627503, -1.303876538059268 ]
 ])

  _self.dphiJdzeta = np.array([
 [0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000, 1.000000000000000, -1.000000000000000, -1.000000000000000, 0.000000000000000, 0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, -0.490116253733444, 0.490116253733444, 0.000000000000000, 0.509883746266556, 2.470348761200332, -2.470348761200332, -0.509883746266556, 0.000000000000000, 0.000000000000000 ],
 [0.254842253637503, 0.254842253637503, -0.235274000095941, -1.215506507562829, -0.509684507275006, 0.000199238991550, 0.000199238991550, -1.019568253541562, -1.019568253541562, 1.450780507658770, 2.038738029100023 ],
 [-0.254842253637503, -0.254842253637503, 1.215506507562829, 0.235274000095941, 0.509684507275006, 1.019568253541562, 1.019568253541562, -0.000199238991550, -0.000199238991550, -1.450780507658770, -2.038738029100023 ],
 [0.000000000000000, 0.000000000000000, -0.490116253733444, 0.490116253733444, 0.000000000000000, 2.470348761200332, 0.509883746266556, -0.509883746266556, -2.470348761200332, 0.000000000000000, 0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, -0.871684678429471, 0.871684678429471, -0.000000000000000, 0.128315321570529, 3.615054035288412, -3.615054035288412, -0.128315321570529, 0.000000000000000, 0.000000000000001 ],
 [0.028704265704656, 0.028704265704656, -0.842980412724815, -2.586349769583757, -0.057408531409312, 0.070906790161218, 0.070906790161218, -0.185723852979841, -0.185723852979841, 3.429330182308572, 0.229634125637247 ],
 [-0.028704265704656, -0.028704265704656, 2.586349769583757, 0.842980412724815, 0.057408531409312, 0.185723852979841, 0.185723852979841, -0.070906790161218, -0.070906790161218, -3.429330182308572, -0.229634125637247 ],
 [-0.000000000000000, -0.000000000000000, -0.871684678429471, 0.871684678429471, -0.000000000000000, 3.615054035288412, 0.128315321570529, -0.128315321570529, -3.615054035288412, -0.000000000000001, -0.000000000000001 ],
 [0.287177803302528, 0.287177803302528, -0.513713814124348, -0.513713814124347, -0.574355606605057, -0.375247224031933, 1.226536010821819, -2.375247224031933, -0.773463989178181, 1.027427628248695, 2.297422426420227 ],
 [0.287177803302528, 0.287177803302528, -0.513713814124348, -0.513713814124347, -0.574355606605057, 1.226536010821819, -0.375247224031933, -0.773463989178181, -2.375247224031933, 1.027427628248695, 2.297422426420227 ],
 [-0.000000000000000, -0.000000000000000, 0.800891617426876, -0.800891617426876, -0.000000000000000, 0.199108382573124, 0.199108382573124, -0.199108382573124, -0.199108382573124, -0.000000000000000, -0.000000000000000 ],
 [-0.287177803302528, -0.287177803302528, 0.513713814124348, 0.513713814124347, 0.574355606605057, 2.375247224031933, 0.773463989178181, 0.375247224031933, -1.226536010821819, -1.027427628248695, -2.297422426420227 ],
 [-0.287177803302528, -0.287177803302528, 0.513713814124348, 0.513713814124347, 0.574355606605057, 0.773463989178181, 2.375247224031933, -1.226536010821819, 0.375247224031933, -1.027427628248695, -2.297422426420227 ],
 [-0.000000000000000, -0.000000000000000, -0.800891617426877, 0.800891617426876, 0.000000000000001, 1.800891617426877, 1.800891617426877, -1.800891617426875, -1.800891617426875, -0.000000000000001, -0.000000000000004 ],
 [0.246452103565170, 0.246452103565170, -0.018626106840630, -0.018626106840630, -0.492904207130340, 0.242017582463859, 0.772174003275460, -1.757982417536141, -1.227825996724540, 0.037252213681260, 1.971616828521362 ],
 [0.246452103565170, 0.246452103565170, -0.018626106840630, -0.018626106840630, -0.492904207130340, 0.772174003275460, 0.242017582463859, -1.227825996724540, -1.757982417536141, 0.037252213681260, 1.971616828521362 ],
 [-0.000000000000000, -0.000000000000000, 0.265078210405800, -0.265078210405800, -0.000000000000000, 0.734921789594200, 0.734921789594200, -0.734921789594200, -0.734921789594200, -0.000000000000000, -0.000000000000000 ],
 [-0.246452103565170, -0.246452103565170, 0.018626106840630, 0.018626106840630, 0.492904207130341, 1.757982417536141, 1.227825996724540, -0.242017582463859, -0.772174003275460, -0.037252213681260, -1.971616828521362 ],
 [-0.246452103565170, -0.246452103565170, 0.018626106840630, 0.018626106840630, 0.492904207130341, 1.227825996724540, 1.757982417536141, -0.772174003275460, -0.242017582463859, -0.037252213681260, -1.971616828521362 ],
 [-0.000000000000000, -0.000000000000000, -0.265078210405801, 0.265078210405800, 0.000000000000001, 1.265078210405801, 1.265078210405801, -1.265078210405800, -1.265078210405800, -0.000000000000001, -0.000000000000002 ],
 [0.047853869096381, 0.047853869096381, -0.024541773315016, -1.005266264347144, -0.095707738192762, 0.831896619395842, -0.004036586813493, -0.187378889572030, -1.023312095781365, 1.029808037662159, 0.382830952771046 ],
 [0.047853869096381, 0.047853869096381, -0.024541773315016, -1.005266264347144, -0.095707738192762, -0.004036586813493, 0.831896619395842, -1.023312095781365, -0.187378889572030, 1.029808037662159, 0.382830952771046 ],
 [0.843864235528750, 0.843864235528750, -0.064464613091981, -0.209255897914774, -1.687728471057500, -0.760124113468897, -0.760124113468897, -2.615332828646104, -2.615332828646104, 0.273720511006756, 6.750913884230001 ],
 [-0.796010366432370, -0.796010366432370, -0.868406008843766, 0.112318482188362, 1.592020732864739, 2.519625090453343, 3.645140866308264, -0.461099400578785, 0.664416375276136, 0.756087526655404, -6.368082931458957 ],
 [-0.796010366432370, -0.796010366432370, -0.868406008843766, 0.112318482188362, 1.592020732864739, 3.645140866308264, 2.519625090453343, 0.664416375276136, -0.461099400578785, 0.756087526655404, -6.368082931458957 ],
 [-0.843864235528750, -0.843864235528750, 0.209255897914774, 0.064464613091982, 1.687728471057500, 2.615332828646104, 2.615332828646104, 0.760124113468897, 0.760124113468897, -0.273720511006756, -6.750913884230002 ],
 [-0.047853869096381, -0.047853869096381, 1.005266264347144, 0.024541773315015, 0.095707738192762, 0.187378889572030, 1.023312095781365, -0.831896619395842, 0.004036586813493, -1.029808037662159, -0.382830952771046 ],
 [-0.000000000000000, -0.000000000000000, -0.072395642411396, 0.072395642411397, 0.000000000000000, 2.053120133443525, 0.091671151379268, -0.091671151379268, -2.053120133443525, -0.000000000000000, -0.000000000000000 ],
 [0.796010366432369, 0.796010366432369, -0.112318482188363, 0.868406008843766, -1.592020732864738, -0.664416375276134, 0.461099400578787, -3.645140866308263, -2.519625090453342, -0.756087526655403, 6.368082931458952 ],
 [0.796010366432369, 0.796010366432369, -0.112318482188363, 0.868406008843766, -1.592020732864738, 0.461099400578787, -0.664416375276134, -2.519625090453342, -3.645140866308263, -0.756087526655403, 6.368082931458952 ],
 [-0.047853869096381, -0.047853869096381, 1.005266264347144, 0.024541773315015, 0.095707738192762, 1.023312095781365, 0.187378889572030, 0.004036586813493, -0.831896619395842, -1.029808037662159, -0.382830952771046 ],
 [-0.000000000000000, -0.000000000000000, -0.072395642411396, 0.072395642411397, 0.000000000000000, 0.091671151379268, 2.053120133443525, -2.053120133443525, -0.091671151379268, -0.000000000000000, -0.000000000000000 ],
 [0.138229950782817, 0.138229950782817, -0.709889855329868, 0.363244049787600, -0.276459901565634, -0.124579707678320, 2.644793809664519, -3.197713612795788, -0.428340095452949, 0.346645805542269, 1.105839606262538 ],
 [0.138229950782817, 0.138229950782817, -0.709889855329868, 0.363244049787600, -0.276459901565634, 2.644793809664519, -0.124579707678320, -0.428340095452949, -3.197713612795788, 0.346645805542269, 1.105839606262538 ],
 [-0.024754616474591, -0.024754616474591, 1.896499094755562, 0.200259482530191, 0.049509232949182, 0.201389426836497, 0.201389426836497, -0.102370960938132, -0.102370960938132, -2.096758577285753, -0.198036931796730 ],
 [0.162984567257408, 0.162984567257408, -0.685135238855277, -1.758269143972745, -0.325969134514817, -0.174088940627503, 0.449016766480401, -1.100955035510035, -0.477849328402131, 2.443404382828022, 1.303876538059268 ],
 [0.162984567257408, 0.162984567257408, -0.685135238855277, -1.758269143972745, -0.325969134514817, 0.449016766480401, -0.174088940627503, -0.477849328402131, -1.100955035510035, 2.443404382828022, 1.303876538059268 ],
 [0.024754616474591, 0.024754616474591, -0.200259482530191, -1.896499094755562, -0.049509232949182, 0.102370960938132, 0.102370960938132, -0.201389426836497, -0.201389426836497, 2.096758577285753, 0.198036931796730 ],
 [-0.138229950782817, -0.138229950782817, -0.363244049787600, 0.709889855329868, 0.276459901565634, 3.197713612795788, 0.428340095452949, 0.124579707678320, -2.644793809664519, -0.346645805542269, -1.105839606262538 ],
 [-0.000000000000000, -0.000000000000000, -0.848119806112686, 0.848119806112686, -0.000000000000000, 0.774985900995217, 2.921253711230154, -2.921253711230154, -0.774985900995218, -0.000000000000000, -0.000000000000001 ],
 [-0.162984567257408, -0.162984567257408, 1.758269143972745, 0.685135238855277, 0.325969134514817, 0.477849328402131, 1.100955035510035, -0.449016766480401, 0.174088940627503, -2.443404382828022, -1.303876538059268 ],
 [-0.162984567257408, -0.162984567257408, 1.758269143972745, 0.685135238855277, 0.325969134514817, 1.100955035510035, 0.477849328402131, 0.174088940627503, -0.449016766480401, -2.443404382828022, -1.303876538059268 ],
 [-0.138229950782817, -0.138229950782817, -0.363244049787600, 0.709889855329868, 0.276459901565634, 0.428340095452949, 3.197713612795788, -2.644793809664519, 0.124579707678320, -0.346645805542269, -1.105839606262538 ],
 [-0.000000000000000, -0.000000000000000, -0.848119806112686, 0.848119806112686, -0.000000000000000, 2.921253711230154, 0.774985900995217, -0.774985900995218, -2.921253711230154, -0.000000000000000, -0.000000000000001 ]
 ])

 def tet20(_self):
  _self.tet4()

  _self.phiJ = np.array([
 [0.039062500000000, 0.039062500000000, 0.039062500000000, 0.039062500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, -0.070312500000000, 0.421875000000000, 0.421875000000000, 0.421875000000000, 0.421875000000000 ],
 [-0.038771899652530, 0.063671804703726, 0.063671804703726, 0.063671804703726, 0.302099210936615, -0.218786363317538, 0.302099210936615, -0.218786363317538, 0.302099210936615, -0.218786363317538, -0.045157838566153, -0.045157838566153, -0.045157838566153, -0.045157838566153, -0.045157838566153, -0.045157838566153, 0.270947031396921, 0.270947031396921, 0.270947031396921, 0.055923879890278 ],
 [0.063671804703726, 0.063671804703726, 0.063671804703726, -0.038771899652530, -0.045157838566153, -0.045157838566153, -0.045157838566153, -0.045157838566153, -0.218786363317538, 0.302099210936615, -0.045157838566153, -0.045157838566153, -0.218786363317538, 0.302099210936615, -0.218786363317538, 0.302099210936615, 0.055923879890278, 0.270947031396921, 0.270947031396921, 0.270947031396921 ],
 [0.063671804703726, 0.063671804703726, -0.038771899652530, 0.063671804703726, -0.045157838566153, -0.045157838566153, -0.218786363317538, 0.302099210936615, -0.045157838566153, -0.045157838566153, -0.218786363317538, 0.302099210936615, 0.302099210936615, -0.218786363317538, -0.045157838566153, -0.045157838566153, 0.270947031396921, 0.055923879890278, 0.270947031396921, 0.270947031396921 ],
 [0.063671804703726, -0.038771899652530, 0.063671804703726, 0.063671804703726, -0.218786363317538, 0.302099210936615, -0.045157838566153, -0.045157838566153, -0.045157838566153, -0.045157838566153, 0.302099210936615, -0.218786363317538, -0.045157838566153, -0.045157838566153, 0.302099210936615, -0.218786363317538, 0.270947031396921, 0.270947031396921, 0.055923879890278, 0.270947031396921 ],
 [0.550041650800971, 0.027596647713615, 0.027596647713615, 0.027596647713616, 0.223259314605636, -0.117907286007044, 0.223259314605636, -0.117907286007044, 0.223259314605637, -0.117907286007044, -0.004185085802817, -0.004185085802817, -0.004185085802817, -0.004185085802817, -0.004185085802817, -0.004185085802817, 0.025110514816903, 0.025110514816903, 0.025110514816903, 0.000891290628599 ],
 [0.027596647713615, 0.027596647713615, 0.027596647713615, 0.550041650800971, -0.004185085802817, -0.004185085802817, -0.004185085802817, -0.004185085802817, -0.117907286007044, 0.223259314605636, -0.004185085802817, -0.004185085802817, -0.117907286007044, 0.223259314605636, -0.117907286007044, 0.223259314605636, 0.000891290628599, 0.025110514816903, 0.025110514816903, 0.025110514816903 ],
 [0.027596647713615, 0.027596647713615, 0.550041650800971, 0.027596647713615, -0.004185085802817, -0.004185085802817, -0.117907286007044, 0.223259314605636, -0.004185085802817, -0.004185085802817, -0.117907286007044, 0.223259314605636, 0.223259314605636, -0.117907286007044, -0.004185085802817, -0.004185085802817, 0.025110514816903, 0.000891290628599, 0.025110514816903, 0.025110514816903 ],
 [0.027596647713615, 0.550041650800971, 0.027596647713615, 0.027596647713616, -0.117907286007044, 0.223259314605636, -0.004185085802817, -0.004185085802817, -0.004185085802817, -0.004185085802817, 0.223259314605636, -0.117907286007044, -0.004185085802817, -0.004185085802817, 0.223259314605637, -0.117907286007044, 0.025110514816903, 0.025110514816903, 0.000891290628599, 0.025110514816903 ],
 [-0.051257915196969, 0.039182189481629, 0.039182189481629, -0.051257915196969, 0.035364430741797, -0.085788705026457, 0.035364430741797, -0.085788705026457, 0.319863513805531, 0.319863513805531, -0.009484885228833, -0.009484885228833, -0.085788705026457, 0.035364430741797, -0.085788705026457, 0.035364430741797, 0.030119687085727, 0.272425958622235, 0.272425958622235, 0.030119687085727 ],
 [0.039182189481629, -0.051257915196969, 0.039182189481629, -0.051257915196969, -0.085788705026457, 0.035364430741797, -0.009484885228833, -0.009484885228833, -0.085788705026457, 0.035364430741797, 0.035364430741797, -0.085788705026457, -0.085788705026457, 0.035364430741797, 0.319863513805531, 0.319863513805531, 0.030119687085727, 0.272425958622235, 0.030119687085727, 0.272425958622235 ],
 [0.039182189481629, 0.039182189481629, -0.051257915196969, -0.051257915196969, -0.009484885228833, -0.009484885228833, -0.085788705026457, 0.035364430741797, -0.085788705026457, 0.035364430741797, -0.085788705026457, 0.035364430741797, 0.319863513805531, 0.319863513805531, -0.085788705026457, 0.035364430741797, 0.030119687085727, 0.030119687085727, 0.272425958622235, 0.272425958622235 ],
 [0.039182189481629, -0.051257915196969, -0.051257915196969, 0.039182189481629, -0.085788705026457, 0.035364430741797, -0.085788705026457, 0.035364430741797, -0.009484885228833, -0.009484885228833, 0.319863513805531, 0.319863513805531, 0.035364430741797, -0.085788705026457, 0.035364430741797, -0.085788705026457, 0.272425958622235, 0.030119687085727, 0.030119687085727, 0.272425958622235 ],
 [-0.051257915196969, 0.039182189481629, -0.051257915196969, 0.039182189481629, 0.035364430741797, -0.085788705026457, 0.319863513805531, 0.319863513805531, 0.035364430741797, -0.085788705026457, -0.085788705026457, 0.035364430741797, 0.035364430741797, -0.085788705026457, -0.009484885228833, -0.009484885228833, 0.272425958622235, 0.030119687085727, 0.272425958622235, 0.030119687085727 ],
 [-0.051257915196969, -0.051257915196969, 0.039182189481629, 0.039182189481629, 0.319863513805531, 0.319863513805531, 0.035364430741797, -0.085788705026457, 0.035364430741797, -0.085788705026457, 0.035364430741797, -0.085788705026457, -0.009484885228833, -0.009484885228833, 0.035364430741797, -0.085788705026457, 0.272425958622235, 0.272425958622235, 0.030119687085727, 0.030119687085727 ],
 [0.008509532090647, 0.059734247304861, 0.059734247304861, 0.008509532090647, -0.013385899101976, -0.117357880293532, -0.013385899101976, -0.117357880293532, -0.023042192407917, -0.023042192407917, -0.068176704569624, -0.068176704569624, -0.117357880293532, -0.013385899101976, -0.117357880293532, -0.013385899101976, 0.288259356994968, 0.496203319378081, 0.496203319378081, 0.288259356994968 ],
 [0.059734247304861, 0.008509532090647, 0.059734247304861, 0.008509532090647, -0.117357880293532, -0.013385899101976, -0.068176704569624, -0.068176704569624, -0.117357880293532, -0.013385899101976, -0.013385899101976, -0.117357880293532, -0.117357880293532, -0.013385899101976, -0.023042192407917, -0.023042192407917, 0.288259356994968, 0.496203319378081, 0.288259356994968, 0.496203319378081 ],
 [0.059734247304861, 0.059734247304861, 0.008509532090647, 0.008509532090647, -0.068176704569624, -0.068176704569624, -0.117357880293532, -0.013385899101976, -0.117357880293532, -0.013385899101976, -0.117357880293532, -0.013385899101976, -0.023042192407917, -0.023042192407917, -0.117357880293532, -0.013385899101976, 0.288259356994968, 0.288259356994968, 0.496203319378081, 0.496203319378081 ],
 [0.059734247304861, 0.008509532090647, 0.008509532090647, 0.059734247304861, -0.117357880293532, -0.013385899101976, -0.117357880293532, -0.013385899101976, -0.068176704569624, -0.068176704569624, -0.023042192407917, -0.023042192407917, -0.013385899101976, -0.117357880293532, -0.013385899101976, -0.117357880293532, 0.496203319378081, 0.288259356994968, 0.288259356994968, 0.496203319378081 ],
 [0.008509532090647, 0.059734247304861, 0.008509532090647, 0.059734247304861, -0.013385899101976, -0.117357880293532, -0.023042192407917, -0.023042192407917, -0.013385899101976, -0.117357880293532, -0.117357880293532, -0.013385899101976, -0.013385899101976, -0.117357880293532, -0.068176704569624, -0.068176704569624, 0.496203319378081, 0.288259356994968, 0.496203319378081, 0.288259356994968 ],
 [0.008509532090647, 0.008509532090647, 0.059734247304861, 0.059734247304861, -0.023042192407917, -0.023042192407917, -0.013385899101976, -0.117357880293532, -0.013385899101976, -0.117357880293532, -0.013385899101976, -0.117357880293532, -0.068176704569624, -0.068176704569624, -0.013385899101976, -0.117357880293532, 0.496203319378081, 0.496203319378081, 0.288259356994968, 0.288259356994968 ],
 [0.020608441824180, 0.046019986530096, 0.046019986530096, -0.063752656519050, -0.022271666485314, -0.007277551464919, -0.022271666485314, -0.007277551464919, -0.049295161770486, 0.028576216033134, -0.073640271228896, -0.073640271228896, -0.162992252306066, 0.289157735196979, -0.162992252306066, 0.289157735196979, 0.033276836780234, 0.073653538830305, 0.073653538830305, 0.745287285507617 ],
 [0.046019986530096, 0.020608441824180, 0.046019986530096, -0.063752656519050, -0.007277551464919, -0.022271666485314, -0.073640271228896, -0.073640271228896, -0.162992252306066, 0.289157735196979, -0.022271666485314, -0.007277551464919, -0.162992252306066, 0.289157735196979, -0.049295161770486, 0.028576216033134, 0.033276836780234, 0.073653538830305, 0.745287285507617, 0.073653538830305 ],
 [0.046019986530096, 0.046019986530096, 0.020608441824180, -0.063752656519050, -0.073640271228896, -0.073640271228896, -0.007277551464919, -0.022271666485314, -0.162992252306066, 0.289157735196979, -0.007277551464919, -0.022271666485314, -0.049295161770486, 0.028576216033134, -0.162992252306066, 0.289157735196979, 0.033276836780234, 0.745287285507617, 0.073653538830305, 0.073653538830305 ],
 [-0.063752656519050, 0.046019986530096, 0.046019986530096, 0.020608441824179, 0.289157735196979, -0.162992252306066, 0.289157735196979, -0.162992252306066, 0.028576216033134, -0.049295161770486, -0.073640271228896, -0.073640271228896, -0.007277551464919, -0.022271666485314, -0.007277551464919, -0.022271666485314, 0.745287285507617, 0.073653538830305, 0.073653538830305, 0.033276836780234 ],
 [0.046019986530096, -0.063752656519050, 0.046019986530096, 0.020608441824179, -0.162992252306066, 0.289157735196979, -0.073640271228896, -0.073640271228896, -0.007277551464919, -0.022271666485314, 0.289157735196979, -0.162992252306066, -0.007277551464919, -0.022271666485314, 0.028576216033134, -0.049295161770486, 0.745287285507617, 0.073653538830305, 0.033276836780234, 0.073653538830305 ],
 [0.046019986530096, 0.046019986530096, -0.063752656519050, 0.020608441824179, -0.073640271228896, -0.073640271228896, -0.162992252306066, 0.289157735196979, -0.007277551464919, -0.022271666485314, -0.162992252306066, 0.289157735196979, 0.028576216033134, -0.049295161770486, -0.007277551464919, -0.022271666485314, 0.745287285507617, 0.033276836780234, 0.073653538830305, 0.073653538830305 ],
 [0.046019986530096, 0.020608441824180, -0.063752656519050, 0.046019986530096, -0.007277551464919, -0.022271666485314, -0.162992252306066, 0.289157735196979, -0.073640271228896, -0.073640271228896, -0.049295161770486, 0.028576216033134, 0.289157735196979, -0.162992252306066, -0.022271666485314, -0.007277551464919, 0.073653538830305, 0.033276836780234, 0.745287285507617, 0.073653538830305 ],
 [0.020608441824180, -0.063752656519050, 0.046019986530096, 0.046019986530096, -0.049295161770486, 0.028576216033134, -0.022271666485314, -0.007277551464919, -0.022271666485314, -0.007277551464919, 0.289157735196979, -0.162992252306066, -0.073640271228896, -0.073640271228896, 0.289157735196979, -0.162992252306066, 0.073653538830305, 0.073653538830305, 0.033276836780234, 0.745287285507617 ],
 [-0.063752656519050, 0.046019986530096, 0.020608441824180, 0.046019986530096, 0.289157735196979, -0.162992252306066, 0.028576216033134, -0.049295161770486, 0.289157735196979, -0.162992252306066, -0.007277551464919, -0.022271666485314, -0.022271666485314, -0.007277551464919, -0.073640271228896, -0.073640271228896, 0.073653538830305, 0.745287285507617, 0.073653538830305, 0.033276836780234 ],
 [0.046019986530096, -0.063752656519050, 0.020608441824180, 0.046019986530096, -0.162992252306066, 0.289157735196979, -0.007277551464919, -0.022271666485314, -0.073640271228896, -0.073640271228896, 0.028576216033134, -0.049295161770486, -0.022271666485314, -0.007277551464919, 0.289157735196979, -0.162992252306066, 0.073653538830305, 0.745287285507617, 0.033276836780234, 0.073653538830305 ],
 [0.020608441824180, 0.046019986530096, -0.063752656519050, 0.046019986530096, -0.022271666485314, -0.007277551464919, -0.049295161770486, 0.028576216033134, -0.022271666485314, -0.007277551464919, -0.162992252306066, 0.289157735196979, 0.289157735196979, -0.162992252306066, -0.073640271228896, -0.073640271228896, 0.073653538830305, 0.033276836780234, 0.073653538830305, 0.745287285507617 ],
 [-0.063752656519050, 0.020608441824180, 0.046019986530096, 0.046019986530096, 0.028576216033134, -0.049295161770486, 0.289157735196979, -0.162992252306066, 0.289157735196979, -0.162992252306066, -0.022271666485314, -0.007277551464919, -0.073640271228896, -0.073640271228896, -0.022271666485314, -0.007277551464919, 0.073653538830305, 0.073653538830305, 0.745287285507617, 0.033276836780234 ],
 [0.083036079971254, 0.031728628442538, 0.031728628442538, 0.057554424113430, 0.148611677557672, -0.110570867061610, 0.148611677557672, -0.110570867061610, 0.758307926021571, -0.266637603062046, -0.005748738859294, -0.005748738859294, -0.029333591500164, -0.013862873565220, -0.029333591500164, -0.013862873565220, 0.028428592158095, 0.145060113131097, 0.145060113131097, 0.007541884507659 ],
 [0.031728628442538, 0.083036079971254, 0.031728628442538, 0.057554424113430, -0.110570867061610, 0.148611677557672, -0.005748738859294, -0.005748738859294, -0.029333591500164, -0.013862873565220, 0.148611677557672, -0.110570867061610, -0.029333591500164, -0.013862873565220, 0.758307926021571, -0.266637603062046, 0.028428592158095, 0.145060113131097, 0.007541884507659, 0.145060113131097 ],
 [0.031728628442538, 0.031728628442538, 0.083036079971254, 0.057554424113430, -0.005748738859294, -0.005748738859294, -0.110570867061610, 0.148611677557672, -0.029333591500164, -0.013862873565220, -0.110570867061610, 0.148611677557672, 0.758307926021571, -0.266637603062046, -0.029333591500164, -0.013862873565220, 0.028428592158095, 0.007541884507659, 0.145060113131097, 0.145060113131097 ],
 [0.057554424113430, 0.031728628442538, 0.031728628442538, 0.083036079971254, -0.013862873565220, -0.029333591500164, -0.013862873565220, -0.029333591500164, -0.266637603062046, 0.758307926021571, -0.005748738859294, -0.005748738859294, -0.110570867061610, 0.148611677557672, -0.110570867061610, 0.148611677557672, 0.007541884507659, 0.145060113131097, 0.145060113131097, 0.028428592158095 ],
 [0.031728628442538, 0.057554424113430, 0.031728628442538, 0.083036079971254, -0.029333591500164, -0.013862873565220, -0.005748738859294, -0.005748738859294, -0.110570867061610, 0.148611677557672, -0.013862873565220, -0.029333591500164, -0.110570867061610, 0.148611677557672, -0.266637603062046, 0.758307926021571, 0.007541884507659, 0.145060113131097, 0.028428592158095, 0.145060113131097 ],
 [0.031728628442538, 0.031728628442538, 0.057554424113430, 0.083036079971254, -0.005748738859294, -0.005748738859294, -0.029333591500164, -0.013862873565220, -0.110570867061610, 0.148611677557672, -0.029333591500164, -0.013862873565220, -0.266637603062046, 0.758307926021571, -0.110570867061610, 0.148611677557672, 0.007541884507659, 0.028428592158095, 0.145060113131097, 0.145060113131097 ],
 [0.031728628442538, 0.083036079971254, 0.057554424113430, 0.031728628442538, -0.110570867061610, 0.148611677557672, -0.029333591500164, -0.013862873565220, -0.005748738859294, -0.005748738859294, 0.758307926021571, -0.266637603062046, -0.013862873565220, -0.029333591500164, 0.148611677557672, -0.110570867061610, 0.145060113131097, 0.028428592158095, 0.007541884507659, 0.145060113131097 ],
 [0.083036079971254, 0.057554424113430, 0.031728628442538, 0.031728628442538, 0.758307926021571, -0.266637603062046, 0.148611677557672, -0.110570867061610, 0.148611677557672, -0.110570867061610, -0.013862873565220, -0.029333591500164, -0.005748738859294, -0.005748738859294, -0.013862873565220, -0.029333591500164, 0.145060113131097, 0.145060113131097, 0.028428592158095, 0.007541884507659 ],
 [0.057554424113430, 0.031728628442538, 0.083036079971254, 0.031728628442538, -0.013862873565220, -0.029333591500164, -0.266637603062046, 0.758307926021571, -0.013862873565220, -0.029333591500164, -0.110570867061610, 0.148611677557672, 0.148611677557672, -0.110570867061610, -0.005748738859294, -0.005748738859294, 0.145060113131097, 0.007541884507659, 0.145060113131097, 0.028428592158095 ],
 [0.031728628442538, 0.057554424113430, 0.083036079971254, 0.031728628442538, -0.029333591500164, -0.013862873565220, -0.110570867061610, 0.148611677557672, -0.005748738859294, -0.005748738859294, -0.266637603062046, 0.758307926021571, 0.148611677557672, -0.110570867061610, -0.013862873565220, -0.029333591500164, 0.145060113131097, 0.007541884507659, 0.028428592158095, 0.145060113131097 ],
 [0.083036079971254, 0.031728628442538, 0.057554424113430, 0.031728628442538, 0.148611677557672, -0.110570867061610, 0.758307926021571, -0.266637603062046, 0.148611677557672, -0.110570867061610, -0.029333591500164, -0.013862873565220, -0.013862873565220, -0.029333591500164, -0.005748738859294, -0.005748738859294, 0.145060113131097, 0.028428592158095, 0.145060113131097, 0.007541884507659 ],
 [0.057554424113430, 0.083036079971254, 0.031728628442538, 0.031728628442538, -0.266637603062046, 0.758307926021571, -0.013862873565220, -0.029333591500164, -0.013862873565220, -0.029333591500164, 0.148611677557672, -0.110570867061610, -0.005748738859294, -0.005748738859294, 0.148611677557672, -0.110570867061610, 0.145060113131097, 0.145060113131097, 0.007541884507659, 0.028428592158095 ]
 ])
  _self.dphiJdxi = np.array([
 [-0.406250000000000, 0.000000000000000, 0.000000000000000, 0.406250000000000, 0.562500000000000, -0.281250000000000, 0.562500000000000, -0.281250000000000, 0.843750000000000, -0.843750000000000, 0.000000000000000, 0.000000000000000, 0.281250000000000, -0.562500000000000, 0.281250000000000, -0.562500000000000, 1.687500000000000, 0.000000000000000, 0.000000000000000, -1.687500000000000 ],
 [0.590803445206391, 0.000000000000000, 0.000000000000000, -0.072120906434126, 1.551940059546113, -0.354259879015998, 1.551940059546113, -0.354259879015998, -0.818005742010651, 0.299323203238387, 0.000000000000000, 0.000000000000000, 0.354259879015998, 0.134900543482121, 0.354259879015998, 0.134900543482121, 0.438718671067755, -1.686840603028234, -1.686840603028234, -0.438718671067755 ],
 [0.072120906434126, 0.000000000000000, 0.000000000000000, -0.590803445206391, -0.134900543482121, -0.354259879015998, -0.134900543482121, -0.354259879015998, -0.299323203238387, 0.818005742010651, 0.000000000000000, 0.000000000000000, 0.354259879015998, -1.551940059546113, 0.354259879015998, -1.551940059546113, 0.438718671067755, 1.686840603028234, 1.686840603028234, -0.438718671067755 ],
 [0.072120906434126, 0.000000000000000, 0.000000000000000, -0.072120906434126, -0.134900543482121, -0.354259879015998, -0.653583082254385, 2.369945801556764, 0.219359335533877, -0.219359335533877, 0.000000000000000, 0.000000000000000, -2.369945801556764, 0.653583082254385, 0.354259879015998, 0.134900543482121, 2.125559274095989, 0.000000000000000, 0.000000000000000, -2.125559274095989 ],
 [0.072120906434126, 0.000000000000000, 0.000000000000000, -0.072120906434126, -0.653583082254385, 2.369945801556764, -0.134900543482121, -0.354259879015998, 0.219359335533877, -0.219359335533877, 0.000000000000000, 0.000000000000000, 0.354259879015998, 0.134900543482121, -2.369945801556764, 0.653583082254385, 2.125559274095989, 0.000000000000000, 0.000000000000000, -2.125559274095989 ],
 [3.892772898960007, 0.000000000000000, 0.000000000000000, -0.725182719817659, 0.638420523726126, -0.130462543415495, 0.638420523726126, -0.130462543415495, -6.321288164933343, 3.153697985790995, 0.000000000000000, 0.000000000000000, 0.130462543415495, 0.116570350064145, 0.130462543415495, 0.116570350064145, 0.027784386702700, -0.754990873790271, -0.754990873790271, -0.027784386702700 ],
 [0.725182719817659, 0.000000000000000, 0.000000000000000, -3.892772898960007, -0.116570350064145, -0.130462543415495, -0.116570350064145, -0.130462543415495, -3.153697985790997, 6.321288164933345, 0.000000000000000, 0.000000000000000, 0.130462543415495, -0.638420523726126, 0.130462543415495, -0.638420523726126, 0.027784386702700, 0.754990873790271, 0.754990873790271, -0.027784386702700 ],
 [0.725182719817659, 0.000000000000000, 0.000000000000000, -0.725182719817660, -0.116570350064145, -0.130462543415495, -3.284160529206493, 6.959708688659471, 0.013892193351350, -0.013892193351350, 0.000000000000000, 0.000000000000000, -6.959708688659471, 3.284160529206494, 0.130462543415495, 0.116570350064145, 0.782775260492971, -0.000000000000000, -0.000000000000001, -0.782775260492971 ],
 [0.725182719817659, 0.000000000000000, 0.000000000000000, -0.725182719817659, -3.284160529206493, 6.959708688659471, -0.116570350064145, -0.130462543415495, 0.013892193351350, -0.013892193351350, 0.000000000000000, 0.000000000000000, 0.130462543415495, 0.116570350064145, -6.959708688659471, 3.284160529206491, 0.782775260492971, 0.000000000000002, 0.000000000000000, -0.782775260492971 ],
 [-0.315547180510580, 0.000000000000000, 0.000000000000000, 0.315547180510580, 0.381094361021160, -0.190547180510580, 0.381094361021160, -0.190547180510580, 2.736458958699891, -2.736458958699891, 0.000000000000000, 0.000000000000000, 0.190547180510580, -0.381094361021160, 0.190547180510580, -0.381094361021160, 0.066899499768369, -0.000000000000000, -0.000000000000000, -0.066899499768369 ],
 [0.585455889094656, 0.000000000000000, 0.000000000000000, 0.315547180510580, -1.420911778189311, 0.710455889094656, -0.157097430626395, -0.190547180510580, -1.230364597678731, 0.329361528073496, 0.000000000000000, 0.000000000000000, 0.190547180510580, -0.381094361021160, -0.710455889094656, -3.446914847794547, 0.605091291415924, 4.867826625983858, 0.538191791647555, -0.605091291415924 ],
 [0.585455889094656, 0.000000000000000, 0.000000000000000, 0.315547180510580, -0.157097430626395, -0.190547180510580, -1.420911778189311, 0.710455889094656, -1.230364597678731, 0.329361528073496, 0.000000000000000, 0.000000000000000, -0.710455889094656, -3.446914847794547, 0.190547180510580, -0.381094361021160, 0.605091291415924, 0.538191791647555, 4.867826625983858, -0.605091291415924 ],
 [0.585455889094656, 0.000000000000000, 0.000000000000000, -0.585455889094655, -1.420911778189311, 0.710455889094656, -1.420911778189311, 0.710455889094656, 0.033449749884185, -0.033449749884185, 0.000000000000000, 0.000000000000000, -0.710455889094656, 1.420911778189311, -0.710455889094656, 1.420911778189311, 5.472917917399783, 0.000000000000000, 0.000000000000000, -5.472917917399783 ],
 [-0.315547180510580, 0.000000000000000, 0.000000000000000, -0.585455889094655, 0.381094361021160, -0.190547180510580, 3.446914847794547, 0.710455889094656, -0.329361528073496, 1.230364597678731, 0.000000000000000, 0.000000000000000, -0.710455889094656, 1.420911778189311, 0.190547180510580, 0.157097430626395, 0.605091291415924, -0.538191791647555, -4.867826625983859, -0.605091291415924 ],
 [-0.315547180510580, 0.000000000000000, 0.000000000000000, -0.585455889094656, 3.446914847794547, 0.710455889094656, 0.381094361021160, -0.190547180510580, -0.329361528073497, 1.230364597678733, 0.000000000000000, 0.000000000000000, 0.190547180510580, 0.157097430626395, -0.710455889094656, 1.420911778189312, 0.605091291415924, -4.867826625983859, -0.538191791647555, -0.605091291415924 ],
 [-0.496069169726312, 0.000000000000000, 0.000000000000000, 0.496069169726312, 0.742138339452624, -0.371069169726312, 0.742138339452624, -0.371069169726312, 1.350356803686739, -1.350356803686739, 0.000000000000000, 0.000000000000000, 0.371069169726312, -0.742138339452624, 0.371069169726312, -0.742138339452624, 0.911435687134325, -0.000000000000000, -0.000000000000000, -0.911435687134325 ],
 [-0.197856183019786, 0.000000000000000, 0.000000000000000, 0.496069169726312, 0.145712366039573, -0.072856183019786, 0.084648673840851, -0.371069169726312, 0.516781535765885, -0.814994522472410, 0.000000000000000, 0.000000000000000, 0.371069169726312, -0.742138339452624, 0.072856183019786, -1.277500620666952, 1.568925352746098, 1.131788254627380, 0.657489665611773, -1.568925352746098 ],
 [-0.197856183019786, 0.000000000000000, 0.000000000000000, 0.496069169726312, 0.084648673840851, -0.371069169726312, 0.145712366039573, -0.072856183019786, 0.516781535765885, -0.814994522472410, 0.000000000000000, 0.000000000000000, 0.072856183019786, -1.277500620666952, 0.371069169726312, -0.742138339452624, 1.568925352746098, 0.657489665611773, 1.131788254627380, -1.568925352746098 ],
 [-0.197856183019786, 0.000000000000000, 0.000000000000000, 0.197856183019787, 0.145712366039573, -0.072856183019786, 0.145712366039573, -0.072856183019786, 0.455717843567163, -0.455717843567163, 0.000000000000000, 0.000000000000000, 0.072856183019786, -0.145712366039573, 0.072856183019786, -0.145712366039573, 2.700713607373479, 0.000000000000000, 0.000000000000000, -2.700713607373479 ],
 [-0.496069169726312, 0.000000000000000, 0.000000000000000, 0.197856183019787, 0.742138339452624, -0.371069169726312, 1.277500620666953, -0.072856183019786, 0.814994522472410, -0.516781535765885, 0.000000000000000, 0.000000000000000, 0.072856183019786, -0.145712366039573, 0.371069169726312, -0.084648673840851, 1.568925352746098, -0.657489665611773, -1.131788254627380, -1.568925352746098 ],
 [-0.496069169726312, 0.000000000000000, 0.000000000000000, 0.197856183019786, 1.277500620666953, -0.072856183019786, 0.742138339452624, -0.371069169726312, 0.814994522472410, -0.516781535765884, 0.000000000000000, 0.000000000000000, 0.371069169726312, -0.084648673840850, 0.072856183019786, -0.145712366039572, 1.568925352746098, -1.131788254627381, -0.657489665611773, -1.568925352746098 ],
 [0.800830446892597, 0.000000000000000, 0.000000000000000, 0.062858999514142, -0.900059083156961, -0.317550346228776, -0.900059083156961, -0.317550346228776, -1.896113083760515, 1.032423637353776, 0.000000000000000, 0.000000000000000, 0.317550346228776, -2.170260468072067, 0.317550346228776, -2.170260468072067, 1.452009112116805, 3.070319551229027, 3.070319551229027, -1.452009112116805 ],
 [-0.361105248515956, 0.000000000000000, 0.000000000000000, 0.062858999514142, 0.040365773828541, -0.096039507805726, 0.408454209829626, -0.317550346228776, 1.221605566464057, -0.923359317462244, 0.000000000000000, 0.000000000000000, 0.317550346228776, -2.170260468072067, 0.096039507805726, -0.214477513256047, 0.143495819130218, 0.174111739427505, 1.761806258242441, -0.143495819130218 ],
 [-0.361105248515956, 0.000000000000000, 0.000000000000000, 0.062858999514142, 0.408454209829626, -0.317550346228776, 0.040365773828541, -0.096039507805726, 1.221605566464057, -0.923359317462244, 0.000000000000000, 0.000000000000000, 0.096039507805726, -0.214477513256047, 0.317550346228776, -2.170260468072067, 0.143495819130218, 1.761806258242441, 0.174111739427505, -0.143495819130218 ],
 [-0.062858999514142, 0.000000000000000, 0.000000000000000, -0.800830446892597, 2.170260468072067, -0.317550346228776, 2.170260468072067, -0.317550346228776, -1.032423637353777, 1.896113083760517, 0.000000000000000, 0.000000000000000, 0.317550346228776, 0.900059083156961, 0.317550346228776, 0.900059083156961, 1.452009112116805, -3.070319551229028, -3.070319551229028, -1.452009112116805 ],
 [-0.361105248515956, 0.000000000000000, 0.000000000000000, -0.800830446892597, 0.904055220235281, 1.246901150609823, 0.408454209829626, -0.317550346228776, 0.357916120057318, 0.804019575351235, 0.000000000000000, 0.000000000000000, 0.317550346228776, 0.900059083156961, -1.246901150609823, 1.992152591566243, 3.213815370359246, -2.896207811801523, -1.308513292986587, -3.213815370359246 ],
 [-0.361105248515956, 0.000000000000000, 0.000000000000000, -0.800830446892597, 0.408454209829626, -0.317550346228776, 0.904055220235281, 1.246901150609823, 0.357916120057318, 0.804019575351235, 0.000000000000000, 0.000000000000000, -1.246901150609823, 1.992152591566243, 0.317550346228776, 0.900059083156961, 3.213815370359246, -1.308513292986587, -2.896207811801523, -3.213815370359246 ],
 [-0.361105248515956, 0.000000000000000, 0.000000000000000, 0.361105248515956, 0.040365773828541, -0.096039507805726, 0.904055220235281, 1.246901150609823, 0.726004556058403, -0.726004556058403, 0.000000000000000, 0.000000000000000, -1.246901150609823, -0.904055220235281, 0.096039507805726, -0.040365773828542, 0.317607558557724, -0.000000000000000, -0.000000000000001, -0.317607558557724 ],
 [0.800830446892597, 0.000000000000000, 0.000000000000000, 0.361105248515955, -1.992152591566241, 1.246901150609823, -0.900059083156961, -0.317550346228776, -0.804019575351234, -0.357916120057318, 0.000000000000000, 0.000000000000000, 0.317550346228776, -0.408454209829625, -1.246901150609823, -0.904055220235280, 3.213815370359246, 2.896207811801521, 1.308513292986586, -3.213815370359246 ],
 [-0.062858999514142, 0.000000000000000, 0.000000000000000, 0.361105248515955, 2.170260468072067, -0.317550346228776, 0.214477513256047, -0.096039507805726, 0.923359317462243, -1.221605566464056, 0.000000000000000, 0.000000000000000, 0.096039507805726, -0.040365773828541, 0.317550346228776, -0.408454209829625, 0.143495819130218, -1.761806258242441, -0.174111739427505, -0.143495819130218 ],
 [-0.361105248515956, 0.000000000000000, 0.000000000000000, 0.361105248515955, 0.904055220235281, 1.246901150609823, 0.040365773828541, -0.096039507805726, 0.726004556058402, -0.726004556058402, 0.000000000000000, 0.000000000000000, 0.096039507805726, -0.040365773828541, -1.246901150609823, -0.904055220235280, 0.317607558557724, -0.000000000000001, -0.000000000000000, -0.317607558557724 ],
 [0.800830446892597, 0.000000000000000, 0.000000000000000, 0.361105248515956, -0.900059083156961, -0.317550346228776, -1.992152591566241, 1.246901150609823, -0.804019575351234, -0.357916120057318, 0.000000000000000, 0.000000000000000, -1.246901150609823, -0.904055220235281, 0.317550346228776, -0.408454209829626, 3.213815370359246, 1.308513292986587, 2.896207811801523, -3.213815370359246 ],
 [-0.062858999514142, 0.000000000000000, 0.000000000000000, 0.361105248515955, 0.214477513256047, -0.096039507805726, 2.170260468072067, -0.317550346228776, 0.923359317462243, -1.221605566464056, 0.000000000000000, 0.000000000000000, 0.317550346228776, -0.408454209829625, 0.096039507805726, -0.040365773828541, 0.143495819130218, -0.174111739427505, -1.761806258242441, -0.143495819130218 ],
 [1.627508138018108, 0.000000000000000, 0.000000000000000, 0.236959372176209, 0.577845760720719, -0.151401936280363, 0.577845760720719, -0.151401936280363, -0.965395085227917, -0.899072424966400, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.027761984400733, 0.151401936280363, -0.027761984400733, 0.038926563685731, -0.550083776319985, -0.550083776319985, -0.038926563685731 ],
 [0.677732845596408, 0.000000000000000, 0.000000000000000, 0.236959372176209, -2.537699446289976, 3.913918563152031, -0.131938654437497, -0.151401936280363, -0.521829999815295, -0.392862217957323, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.027761984400733, -3.913918563152031, -0.533972191409811, 0.748710978843947, 3.071671637699787, 0.159700638838231, -0.748710978843947 ],
 [0.677732845596408, 0.000000000000000, 0.000000000000000, 0.236959372176209, -0.131938654437497, -0.151401936280363, -2.537699446289976, 3.913918563152031, -0.521829999815295, -0.392862217957323, 0.000000000000000, 0.000000000000000, -3.913918563152031, -0.533972191409811, 0.151401936280363, -0.027761984400733, 0.748710978843947, 0.159700638838231, 3.071671637699787, -0.748710978843947 ],
 [-0.236959372176209, 0.000000000000000, 0.000000000000000, -1.627508138018108, 0.027761984400733, -0.151401936280363, 0.027761984400733, -0.151401936280363, 0.899072424966400, 0.965395085227917, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.577845760720719, 0.151401936280363, -0.577845760720719, 0.038926563685731, 0.550083776319985, 0.550083776319985, -0.038926563685731 ],
 [0.677732845596408, 0.000000000000000, 0.000000000000000, -1.627508138018108, -0.673231936095658, -0.365100233556590, -0.131938654437497, -0.151401936280363, -2.386297510009612, 3.336072802431313, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.577845760720719, 0.365100233556590, -2.948523477924114, 0.198627202523962, 3.621755414019772, 0.709784415158216, -0.198627202523962 ],
 [0.677732845596408, 0.000000000000000, 0.000000000000000, -1.627508138018108, -0.131938654437497, -0.151401936280363, -0.673231936095658, -0.365100233556590, -2.386297510009612, 3.336072802431313, 0.000000000000000, 0.000000000000000, 0.365100233556590, -2.948523477924114, 0.151401936280363, -0.577845760720719, 0.198627202523962, 0.709784415158216, 3.621755414019772, -0.198627202523962 ],
 [0.677732845596408, 0.000000000000000, 0.000000000000000, -0.677732845596408, -2.537699446289976, 3.913918563152031, -0.673231936095658, -0.365100233556590, 0.019463281842866, -0.019463281842866, 0.000000000000000, 0.000000000000000, 0.365100233556590, 0.673231936095658, -3.913918563152031, 2.537699446289975, 3.820382616543734, 0.000000000000000, 0.000000000000000, -3.820382616543734 ],
 [1.627508138018108, 0.000000000000000, 0.000000000000000, -0.677732845596408, 2.948523477924114, -0.365100233556590, 0.577845760720719, -0.151401936280363, -3.336072802431312, 2.386297510009612, 0.000000000000000, 0.000000000000000, 0.151401936280363, 0.131938654437497, 0.365100233556590, 0.673231936095658, 0.198627202523962, -3.621755414019772, -0.709784415158216, -0.198627202523962 ],
 [-0.236959372176209, 0.000000000000000, 0.000000000000000, -0.677732845596408, 0.027761984400733, -0.151401936280363, 0.533972191409811, 3.913918563152031, 0.392862217957323, 0.521829999815295, 0.000000000000000, 0.000000000000000, -3.913918563152031, 2.537699446289975, 0.151401936280363, 0.131938654437497, 0.748710978843947, -0.159700638838231, -3.071671637699786, -0.748710978843947 ],
 [0.677732845596408, 0.000000000000000, 0.000000000000000, -0.677732845596408, -0.673231936095658, -0.365100233556590, -2.537699446289976, 3.913918563152031, 0.019463281842866, -0.019463281842866, 0.000000000000000, 0.000000000000000, -3.913918563152031, 2.537699446289975, 0.365100233556590, 0.673231936095658, 3.820382616543734, -0.000000000000000, -0.000000000000000, -3.820382616543734 ],
 [1.627508138018108, 0.000000000000000, 0.000000000000000, -0.677732845596408, 0.577845760720719, -0.151401936280363, 2.948523477924114, -0.365100233556590, -3.336072802431312, 2.386297510009612, 0.000000000000000, 0.000000000000000, 0.365100233556590, 0.673231936095658, 0.151401936280363, 0.131938654437497, 0.198627202523962, -0.709784415158216, -3.621755414019772, -0.198627202523962 ],
 [-0.236959372176209, 0.000000000000000, 0.000000000000000, -0.677732845596408, 0.533972191409811, 3.913918563152031, 0.027761984400733, -0.151401936280363, 0.392862217957323, 0.521829999815295, 0.000000000000000, 0.000000000000000, 0.151401936280363, 0.131938654437497, -3.913918563152031, 2.537699446289975, 0.748710978843947, -3.071671637699786, -0.159700638838231, -0.748710978843947 ]
 ])
  _self.dphiJdeta = np.array([
 [0.000000000000000, -0.406250000000000, 0.000000000000000, 0.406250000000000, -0.281250000000000, 0.562500000000000, 0.000000000000000, 0.000000000000000, 0.281250000000000, -0.562500000000000, 0.562500000000000, -0.281250000000000, 0.281250000000000, -0.562500000000000, 0.843750000000000, -0.843750000000000, 1.687500000000000, 0.000000000000000, -1.687500000000000, 0.000000000000000 ],
 [0.000000000000000, 0.072120906434126, 0.000000000000000, -0.072120906434126, 2.369945801556764, -0.653583082254385, 0.000000000000000, 0.000000000000000, -2.369945801556764, 0.653583082254385, -0.134900543482121, -0.354259879015998, 0.354259879015998, 0.134900543482121, 0.219359335533877, -0.219359335533877, 2.125559274095989, 0.000000000000000, -2.125559274095989, 0.000000000000000 ],
 [0.000000000000000, 0.072120906434126, 0.000000000000000, -0.590803445206391, -0.354259879015998, -0.134900543482121, 0.000000000000000, 0.000000000000000, 0.354259879015998, -1.551940059546113, -0.134900543482121, -0.354259879015998, 0.354259879015998, -1.551940059546113, -0.299323203238387, 0.818005742010651, 0.438718671067755, 1.686840603028234, -0.438718671067755, 1.686840603028234 ],
 [0.000000000000000, 0.072120906434126, 0.000000000000000, -0.072120906434126, -0.354259879015998, -0.134900543482121, 0.000000000000000, 0.000000000000000, 0.354259879015998, 0.134900543482121, -0.653583082254385, 2.369945801556764, -2.369945801556764, 0.653583082254385, 0.219359335533877, -0.219359335533877, 2.125559274095989, 0.000000000000000, -2.125559274095989, 0.000000000000000 ],
 [0.000000000000000, 0.590803445206391, 0.000000000000000, -0.072120906434126, -0.354259879015998, 1.551940059546113, 0.000000000000000, 0.000000000000000, 0.354259879015998, 0.134900543482121, 1.551940059546113, -0.354259879015998, 0.354259879015998, 0.134900543482121, -0.818005742010651, 0.299323203238387, 0.438718671067755, -1.686840603028234, -0.438718671067755, -1.686840603028234 ],
 [0.000000000000000, 0.725182719817659, 0.000000000000000, -0.725182719817659, 6.959708688659471, -3.284160529206493, 0.000000000000000, 0.000000000000000, -6.959708688659471, 3.284160529206491, -0.116570350064145, -0.130462543415495, 0.130462543415495, 0.116570350064145, 0.013892193351350, -0.013892193351350, 0.782775260492971, -0.000000000000009, -0.782775260492971, -0.000000000000000 ],
 [0.000000000000000, 0.725182719817659, 0.000000000000000, -3.892772898960007, -0.130462543415495, -0.116570350064145, 0.000000000000000, 0.000000000000000, 0.130462543415495, -0.638420523726126, -0.116570350064145, -0.130462543415495, 0.130462543415495, -0.638420523726126, -3.153697985790997, 6.321288164933345, 0.027784386702700, 0.754990873790271, -0.027784386702700, 0.754990873790271 ],
 [0.000000000000000, 0.725182719817659, 0.000000000000000, -0.725182719817660, -0.130462543415495, -0.116570350064145, 0.000000000000000, 0.000000000000000, 0.130462543415495, 0.116570350064145, -3.284160529206493, 6.959708688659471, -6.959708688659471, 3.284160529206494, 0.013892193351350, -0.013892193351350, 0.782775260492971, -0.000000000000000, -0.782775260492971, -0.000000000000001 ],
 [0.000000000000000, 3.892772898960007, 0.000000000000000, -0.725182719817659, -0.130462543415495, 0.638420523726126, 0.000000000000000, 0.000000000000000, 0.130462543415495, 0.116570350064145, 0.638420523726126, -0.130462543415495, 0.130462543415495, 0.116570350064145, -6.321288164933343, 3.153697985790995, 0.027784386702700, -0.754990873790271, -0.027784386702700, -0.754990873790271 ],
 [0.000000000000000, 0.585455889094656, 0.000000000000000, 0.315547180510580, 0.710455889094656, -1.420911778189311, 0.000000000000000, 0.000000000000000, -0.710455889094656, -3.446914847794547, -0.157097430626395, -0.190547180510580, 0.190547180510580, -0.381094361021160, -1.230364597678731, 0.329361528073496, 0.605091291415924, 4.867826625983858, -0.605091291415924, 0.538191791647555 ],
 [0.000000000000000, -0.315547180510580, 0.000000000000000, 0.315547180510580, -0.190547180510580, 0.381094361021160, 0.000000000000000, 0.000000000000000, 0.190547180510580, -0.381094361021160, 0.381094361021160, -0.190547180510580, 0.190547180510580, -0.381094361021160, 2.736458958699891, -2.736458958699891, 0.066899499768369, -0.000000000000000, -0.066899499768369, -0.000000000000000 ],
 [0.000000000000000, 0.585455889094656, 0.000000000000000, 0.315547180510580, -0.190547180510580, -0.157097430626395, 0.000000000000000, 0.000000000000000, 0.190547180510580, -0.381094361021160, -1.420911778189311, 0.710455889094656, -0.710455889094656, -3.446914847794547, -1.230364597678731, 0.329361528073496, 0.605091291415924, 0.538191791647555, -0.605091291415924, 4.867826625983858 ],
 [0.000000000000000, -0.315547180510580, 0.000000000000000, -0.585455889094655, -0.190547180510580, 0.381094361021160, 0.000000000000000, 0.000000000000000, 0.190547180510580, 0.157097430626395, 3.446914847794547, 0.710455889094656, -0.710455889094656, 1.420911778189311, -0.329361528073496, 1.230364597678731, 0.605091291415924, -0.538191791647555, -0.605091291415924, -4.867826625983859 ],
 [0.000000000000000, 0.585455889094656, 0.000000000000000, -0.585455889094655, 0.710455889094656, -1.420911778189311, 0.000000000000000, 0.000000000000000, -0.710455889094656, 1.420911778189311, -1.420911778189311, 0.710455889094656, -0.710455889094656, 1.420911778189311, 0.033449749884185, -0.033449749884185, 5.472917917399783, -0.000000000000000, -5.472917917399783, -0.000000000000000 ],
 [0.000000000000000, -0.315547180510580, 0.000000000000000, -0.585455889094656, 0.710455889094656, 3.446914847794547, 0.000000000000000, 0.000000000000000, -0.710455889094656, 1.420911778189312, 0.381094361021160, -0.190547180510580, 0.190547180510580, 0.157097430626395, -0.329361528073497, 1.230364597678733, 0.605091291415924, -4.867826625983859, -0.605091291415924, -0.538191791647555 ],
 [0.000000000000000, -0.197856183019786, 0.000000000000000, 0.496069169726312, -0.072856183019786, 0.145712366039573, 0.000000000000000, 0.000000000000000, 0.072856183019786, -1.277500620666952, 0.084648673840851, -0.371069169726312, 0.371069169726312, -0.742138339452624, 0.516781535765885, -0.814994522472410, 1.568925352746098, 1.131788254627380, -1.568925352746098, 0.657489665611773 ],
 [0.000000000000000, -0.496069169726312, 0.000000000000000, 0.496069169726312, -0.371069169726312, 0.742138339452624, 0.000000000000000, 0.000000000000000, 0.371069169726312, -0.742138339452624, 0.742138339452624, -0.371069169726312, 0.371069169726312, -0.742138339452624, 1.350356803686739, -1.350356803686739, 0.911435687134325, -0.000000000000000, -0.911435687134325, -0.000000000000000 ],
 [0.000000000000000, -0.197856183019786, 0.000000000000000, 0.496069169726312, -0.371069169726312, 0.084648673840851, 0.000000000000000, 0.000000000000000, 0.371069169726312, -0.742138339452624, 0.145712366039573, -0.072856183019786, 0.072856183019786, -1.277500620666952, 0.516781535765885, -0.814994522472410, 1.568925352746098, 0.657489665611773, -1.568925352746098, 1.131788254627380 ],
 [0.000000000000000, -0.496069169726312, 0.000000000000000, 0.197856183019787, -0.371069169726312, 0.742138339452624, 0.000000000000000, 0.000000000000000, 0.371069169726312, -0.084648673840851, 1.277500620666953, -0.072856183019786, 0.072856183019786, -0.145712366039573, 0.814994522472410, -0.516781535765885, 1.568925352746098, -0.657489665611773, -1.568925352746098, -1.131788254627380 ],
 [0.000000000000000, -0.197856183019786, 0.000000000000000, 0.197856183019787, -0.072856183019786, 0.145712366039573, 0.000000000000000, 0.000000000000000, 0.072856183019786, -0.145712366039573, 0.145712366039573, -0.072856183019786, 0.072856183019786, -0.145712366039573, 0.455717843567163, -0.455717843567163, 2.700713607373479, -0.000000000000000, -2.700713607373479, -0.000000000000000 ],
 [0.000000000000000, -0.496069169726312, 0.000000000000000, 0.197856183019786, -0.072856183019786, 1.277500620666953, 0.000000000000000, 0.000000000000000, 0.072856183019786, -0.145712366039572, 0.742138339452624, -0.371069169726312, 0.371069169726312, -0.084648673840850, 0.814994522472410, -0.516781535765884, 1.568925352746098, -1.131788254627381, -1.568925352746098, -0.657489665611773 ],
 [0.000000000000000, -0.361105248515956, 0.000000000000000, 0.062858999514142, -0.096039507805726, 0.040365773828541, 0.000000000000000, 0.000000000000000, 0.096039507805726, -0.214477513256047, 0.408454209829626, -0.317550346228776, 0.317550346228776, -2.170260468072067, 1.221605566464057, -0.923359317462244, 0.143495819130218, 0.174111739427505, -0.143495819130218, 1.761806258242441 ],
 [0.000000000000000, 0.800830446892597, 0.000000000000000, 0.062858999514142, -0.317550346228776, -0.900059083156961, 0.000000000000000, 0.000000000000000, 0.317550346228776, -2.170260468072067, -0.900059083156961, -0.317550346228776, 0.317550346228776, -2.170260468072067, -1.896113083760515, 1.032423637353776, 1.452009112116805, 3.070319551229027, -1.452009112116805, 3.070319551229027 ],
 [0.000000000000000, -0.361105248515956, 0.000000000000000, 0.062858999514142, -0.317550346228776, 0.408454209829626, 0.000000000000000, 0.000000000000000, 0.317550346228776, -2.170260468072067, 0.040365773828541, -0.096039507805726, 0.096039507805726, -0.214477513256047, 1.221605566464057, -0.923359317462244, 0.143495819130218, 1.761806258242441, -0.143495819130218, 0.174111739427505 ],
 [0.000000000000000, -0.361105248515956, 0.000000000000000, -0.800830446892597, 1.246901150609823, 0.904055220235281, 0.000000000000000, 0.000000000000000, -1.246901150609823, 1.992152591566243, 0.408454209829626, -0.317550346228776, 0.317550346228776, 0.900059083156961, 0.357916120057318, 0.804019575351235, 3.213815370359246, -2.896207811801523, -3.213815370359246, -1.308513292986587 ],
 [0.000000000000000, -0.062858999514142, 0.000000000000000, -0.800830446892597, -0.317550346228776, 2.170260468072067, 0.000000000000000, 0.000000000000000, 0.317550346228776, 0.900059083156961, 2.170260468072067, -0.317550346228776, 0.317550346228776, 0.900059083156961, -1.032423637353777, 1.896113083760517, 1.452009112116805, -3.070319551229028, -1.452009112116805, -3.070319551229028 ],
 [0.000000000000000, -0.361105248515956, 0.000000000000000, -0.800830446892597, -0.317550346228776, 0.408454209829626, 0.000000000000000, 0.000000000000000, 0.317550346228776, 0.900059083156961, 0.904055220235281, 1.246901150609823, -1.246901150609823, 1.992152591566243, 0.357916120057318, 0.804019575351235, 3.213815370359246, -1.308513292986587, -3.213815370359246, -2.896207811801523 ],
 [0.000000000000000, 0.800830446892597, 0.000000000000000, 0.361105248515956, -0.317550346228776, -0.900059083156961, 0.000000000000000, 0.000000000000000, 0.317550346228776, -0.408454209829626, -1.992152591566241, 1.246901150609823, -1.246901150609823, -0.904055220235281, -0.804019575351234, -0.357916120057318, 3.213815370359246, 1.308513292986587, -3.213815370359246, 2.896207811801523 ],
 [0.000000000000000, -0.062858999514142, 0.000000000000000, 0.361105248515955, -0.096039507805726, 0.214477513256047, 0.000000000000000, 0.000000000000000, 0.096039507805726, -0.040365773828541, 2.170260468072067, -0.317550346228776, 0.317550346228776, -0.408454209829625, 0.923359317462243, -1.221605566464056, 0.143495819130218, -0.174111739427505, -0.143495819130218, -1.761806258242441 ],
 [0.000000000000000, -0.361105248515956, 0.000000000000000, 0.361105248515955, 1.246901150609823, 0.904055220235281, 0.000000000000000, 0.000000000000000, -1.246901150609823, -0.904055220235280, 0.040365773828541, -0.096039507805726, 0.096039507805726, -0.040365773828541, 0.726004556058402, -0.726004556058402, 0.317607558557724, -0.000000000000001, -0.317607558557724, -0.000000000000000 ],
 [0.000000000000000, -0.062858999514142, 0.000000000000000, 0.361105248515955, -0.317550346228776, 2.170260468072067, 0.000000000000000, 0.000000000000000, 0.317550346228776, -0.408454209829625, 0.214477513256047, -0.096039507805726, 0.096039507805726, -0.040365773828541, 0.923359317462243, -1.221605566464056, 0.143495819130218, -1.761806258242441, -0.143495819130218, -0.174111739427505 ],
 [0.000000000000000, -0.361105248515956, 0.000000000000000, 0.361105248515956, -0.096039507805726, 0.040365773828541, 0.000000000000000, 0.000000000000000, 0.096039507805726, -0.040365773828542, 0.904055220235281, 1.246901150609823, -1.246901150609823, -0.904055220235281, 0.726004556058403, -0.726004556058403, 0.317607558557724, -0.000000000000000, -0.317607558557724, -0.000000000000001 ],
 [0.000000000000000, 0.800830446892597, 0.000000000000000, 0.361105248515955, 1.246901150609823, -1.992152591566241, 0.000000000000000, 0.000000000000000, -1.246901150609823, -0.904055220235280, -0.900059083156961, -0.317550346228776, 0.317550346228776, -0.408454209829625, -0.804019575351234, -0.357916120057318, 3.213815370359246, 2.896207811801521, -3.213815370359246, 1.308513292986586 ],
 [0.000000000000000, 0.677732845596408, 0.000000000000000, 0.236959372176209, 3.913918563152031, -2.537699446289976, 0.000000000000000, 0.000000000000000, -3.913918563152031, -0.533972191409811, -0.131938654437497, -0.151401936280363, 0.151401936280363, -0.027761984400733, -0.521829999815295, -0.392862217957323, 0.748710978843947, 3.071671637699787, -0.748710978843947, 0.159700638838231 ],
 [0.000000000000000, 1.627508138018108, 0.000000000000000, 0.236959372176209, -0.151401936280363, 0.577845760720719, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.027761984400733, 0.577845760720719, -0.151401936280363, 0.151401936280363, -0.027761984400733, -0.965395085227917, -0.899072424966400, 0.038926563685731, -0.550083776319985, -0.038926563685731, -0.550083776319985 ],
 [0.000000000000000, 0.677732845596408, 0.000000000000000, 0.236959372176209, -0.151401936280363, -0.131938654437497, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.027761984400733, -2.537699446289976, 3.913918563152031, -3.913918563152031, -0.533972191409811, -0.521829999815295, -0.392862217957323, 0.748710978843947, 0.159700638838231, -0.748710978843947, 3.071671637699787 ],
 [0.000000000000000, 0.677732845596408, 0.000000000000000, -1.627508138018108, -0.365100233556590, -0.673231936095658, 0.000000000000000, 0.000000000000000, 0.365100233556590, -2.948523477924114, -0.131938654437497, -0.151401936280363, 0.151401936280363, -0.577845760720719, -2.386297510009612, 3.336072802431313, 0.198627202523962, 3.621755414019772, -0.198627202523962, 0.709784415158216 ],
 [0.000000000000000, -0.236959372176209, 0.000000000000000, -1.627508138018108, -0.151401936280363, 0.027761984400733, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.577845760720719, 0.027761984400733, -0.151401936280363, 0.151401936280363, -0.577845760720719, 0.899072424966400, 0.965395085227917, 0.038926563685731, 0.550083776319985, -0.038926563685731, 0.550083776319985 ],
 [0.000000000000000, 0.677732845596408, 0.000000000000000, -1.627508138018108, -0.151401936280363, -0.131938654437497, 0.000000000000000, 0.000000000000000, 0.151401936280363, -0.577845760720719, -0.673231936095658, -0.365100233556590, 0.365100233556590, -2.948523477924114, -2.386297510009612, 3.336072802431313, 0.198627202523962, 0.709784415158216, -0.198627202523962, 3.621755414019772 ],
 [0.000000000000000, 1.627508138018108, 0.000000000000000, -0.677732845596408, -0.151401936280363, 0.577845760720719, 0.000000000000000, 0.000000000000000, 0.151401936280363, 0.131938654437497, 2.948523477924114, -0.365100233556590, 0.365100233556590, 0.673231936095658, -3.336072802431312, 2.386297510009612, 0.198627202523962, -0.709784415158216, -0.198627202523962, -3.621755414019772 ],
 [0.000000000000000, -0.236959372176209, 0.000000000000000, -0.677732845596408, 3.913918563152031, 0.533972191409811, 0.000000000000000, 0.000000000000000, -3.913918563152031, 2.537699446289975, 0.027761984400733, -0.151401936280363, 0.151401936280363, 0.131938654437497, 0.392862217957323, 0.521829999815295, 0.748710978843947, -3.071671637699786, -0.748710978843947, -0.159700638838231 ],
 [0.000000000000000, 0.677732845596408, 0.000000000000000, -0.677732845596408, -0.365100233556590, -0.673231936095658, 0.000000000000000, 0.000000000000000, 0.365100233556590, 0.673231936095658, -2.537699446289976, 3.913918563152031, -3.913918563152031, 2.537699446289975, 0.019463281842866, -0.019463281842866, 3.820382616543734, -0.000000000000000, -3.820382616543734, -0.000000000000000 ],
 [0.000000000000000, -0.236959372176209, 0.000000000000000, -0.677732845596408, -0.151401936280363, 0.027761984400733, 0.000000000000000, 0.000000000000000, 0.151401936280363, 0.131938654437497, 0.533972191409811, 3.913918563152031, -3.913918563152031, 2.537699446289975, 0.392862217957323, 0.521829999815295, 0.748710978843947, -0.159700638838231, -0.748710978843947, -3.071671637699786 ],
 [0.000000000000000, 0.677732845596408, 0.000000000000000, -0.677732845596408, 3.913918563152031, -2.537699446289976, 0.000000000000000, 0.000000000000000, -3.913918563152031, 2.537699446289975, -0.673231936095658, -0.365100233556590, 0.365100233556590, 0.673231936095658, 0.019463281842866, -0.019463281842866, 3.820382616543734, -0.000000000000000, -3.820382616543734, -0.000000000000000 ],
 [0.000000000000000, 1.627508138018108, 0.000000000000000, -0.677732845596408, -0.365100233556590, 2.948523477924114, 0.000000000000000, 0.000000000000000, 0.365100233556590, 0.673231936095658, 0.577845760720719, -0.151401936280363, 0.151401936280363, 0.131938654437497, -3.336072802431312, 2.386297510009612, 0.198627202523962, -3.621755414019772, -0.198627202523962, -0.709784415158216 ]
 ])
  _self.dphiJdzeta = np.array([
 [0.000000000000000, 0.000000000000000, -0.406250000000000, 0.406250000000000, 0.000000000000000, 0.000000000000000, -0.281250000000000, 0.562500000000000, 0.281250000000000, -0.562500000000000, -0.281250000000000, 0.562500000000000, 0.843750000000000, -0.843750000000000, 0.281250000000000, -0.562500000000000, 1.687500000000000, -1.687500000000000, 0.000000000000000, 0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, 0.072120906434126, -0.072120906434126, 0.000000000000000, 0.000000000000000, 2.369945801556764, -0.653583082254385, -2.369945801556764, 0.653583082254385, -0.354259879015998, -0.134900543482121, 0.219359335533877, -0.219359335533877, 0.354259879015998, 0.134900543482121, 2.125559274095989, -2.125559274095989, 0.000000000000000, 0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, 0.072120906434126, -0.590803445206391, 0.000000000000000, 0.000000000000000, -0.354259879015998, -0.134900543482121, 0.354259879015998, -1.551940059546113, -0.354259879015998, -0.134900543482121, -0.299323203238387, 0.818005742010651, 0.354259879015998, -1.551940059546113, 0.438718671067755, -0.438718671067755, 1.686840603028234, 1.686840603028234 ],
 [0.000000000000000, 0.000000000000000, 0.590803445206391, -0.072120906434126, 0.000000000000000, 0.000000000000000, -0.354259879015998, 1.551940059546113, 0.354259879015998, 0.134900543482121, -0.354259879015998, 1.551940059546113, -0.818005742010651, 0.299323203238387, 0.354259879015998, 0.134900543482121, 0.438718671067755, -0.438718671067755, -1.686840603028234, -1.686840603028234 ],
 [0.000000000000000, 0.000000000000000, 0.072120906434126, -0.072120906434126, 0.000000000000000, 0.000000000000000, -0.354259879015998, -0.134900543482121, 0.354259879015998, 0.134900543482121, 2.369945801556764, -0.653583082254385, 0.219359335533877, -0.219359335533877, -2.369945801556764, 0.653583082254385, 2.125559274095989, -2.125559274095989, 0.000000000000000, 0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, 0.725182719817659, -0.725182719817659, 0.000000000000000, 0.000000000000000, 6.959708688659471, -3.284160529206493, -6.959708688659471, 3.284160529206491, -0.130462543415495, -0.116570350064145, 0.013892193351350, -0.013892193351350, 0.130462543415495, 0.116570350064145, 0.782775260492971, -0.782775260492971, 0.000000000000002, 0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, 0.725182719817659, -3.892772898960007, 0.000000000000000, 0.000000000000000, -0.130462543415495, -0.116570350064145, 0.130462543415495, -0.638420523726126, -0.130462543415495, -0.116570350064145, -3.153697985790997, 6.321288164933345, 0.130462543415495, -0.638420523726126, 0.027784386702700, -0.027784386702700, 0.754990873790271, 0.754990873790271 ],
 [0.000000000000000, 0.000000000000000, 3.892772898960007, -0.725182719817660, 0.000000000000000, 0.000000000000000, -0.130462543415495, 0.638420523726126, 0.130462543415495, 0.116570350064145, -0.130462543415495, 0.638420523726126, -6.321288164933346, 3.153697985790998, 0.130462543415495, 0.116570350064145, 0.027784386702700, -0.027784386702700, -0.754990873790271, -0.754990873790271 ],
 [0.000000000000000, 0.000000000000000, 0.725182719817659, -0.725182719817659, 0.000000000000000, 0.000000000000000, -0.130462543415495, -0.116570350064145, 0.130462543415495, 0.116570350064145, 6.959708688659471, -3.284160529206493, 0.013892193351350, -0.013892193351350, -6.959708688659471, 3.284160529206491, 0.782775260492971, -0.782775260492971, -0.000000000000000, -0.000000000000009 ],
 [0.000000000000000, 0.000000000000000, 0.585455889094656, 0.315547180510580, 0.000000000000000, 0.000000000000000, 0.710455889094656, -1.420911778189311, -0.710455889094656, -3.446914847794547, -0.190547180510580, -0.157097430626395, -1.230364597678731, 0.329361528073496, 0.190547180510580, -0.381094361021160, 0.605091291415924, -0.605091291415924, 4.867826625983858, 0.538191791647555 ],
 [0.000000000000000, 0.000000000000000, 0.585455889094656, 0.315547180510580, 0.000000000000000, 0.000000000000000, -0.190547180510580, -0.157097430626395, 0.190547180510580, -0.381094361021160, 0.710455889094656, -1.420911778189311, -1.230364597678731, 0.329361528073496, -0.710455889094656, -3.446914847794547, 0.605091291415924, -0.605091291415924, 0.538191791647555, 4.867826625983858 ],
 [0.000000000000000, 0.000000000000000, -0.315547180510580, 0.315547180510580, 0.000000000000000, 0.000000000000000, -0.190547180510580, 0.381094361021160, 0.190547180510580, -0.381094361021160, -0.190547180510580, 0.381094361021160, 2.736458958699891, -2.736458958699891, 0.190547180510580, -0.381094361021160, 0.066899499768369, -0.066899499768369, -0.000000000000000, -0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, -0.315547180510580, -0.585455889094655, 0.000000000000000, 0.000000000000000, -0.190547180510580, 0.381094361021160, 0.190547180510580, 0.157097430626395, 0.710455889094656, 3.446914847794547, -0.329361528073496, 1.230364597678731, -0.710455889094656, 1.420911778189311, 0.605091291415924, -0.605091291415924, -0.538191791647555, -4.867826625983859 ],
 [0.000000000000000, 0.000000000000000, -0.315547180510580, -0.585455889094655, 0.000000000000000, 0.000000000000000, 0.710455889094656, 3.446914847794547, -0.710455889094656, 1.420911778189311, -0.190547180510580, 0.381094361021160, -0.329361528073496, 1.230364597678731, 0.190547180510580, 0.157097430626395, 0.605091291415924, -0.605091291415924, -4.867826625983859, -0.538191791647555 ],
 [0.000000000000000, 0.000000000000000, 0.585455889094656, -0.585455889094656, 0.000000000000000, 0.000000000000000, 0.710455889094656, -1.420911778189311, -0.710455889094656, 1.420911778189312, 0.710455889094656, -1.420911778189311, 0.033449749884185, -0.033449749884184, -0.710455889094656, 1.420911778189312, 5.472917917399783, -5.472917917399783, -0.000000000000001, -0.000000000000001 ],
 [0.000000000000000, 0.000000000000000, -0.197856183019786, 0.496069169726312, 0.000000000000000, 0.000000000000000, -0.072856183019786, 0.145712366039573, 0.072856183019786, -1.277500620666952, -0.371069169726312, 0.084648673840851, 0.516781535765885, -0.814994522472410, 0.371069169726312, -0.742138339452624, 1.568925352746098, -1.568925352746098, 1.131788254627380, 0.657489665611773 ],
 [0.000000000000000, 0.000000000000000, -0.197856183019786, 0.496069169726312, 0.000000000000000, 0.000000000000000, -0.371069169726312, 0.084648673840851, 0.371069169726312, -0.742138339452624, -0.072856183019786, 0.145712366039573, 0.516781535765885, -0.814994522472410, 0.072856183019786, -1.277500620666952, 1.568925352746098, -1.568925352746098, 0.657489665611773, 1.131788254627380 ],
 [0.000000000000000, 0.000000000000000, -0.496069169726312, 0.496069169726312, 0.000000000000000, 0.000000000000000, -0.371069169726312, 0.742138339452624, 0.371069169726312, -0.742138339452624, -0.371069169726312, 0.742138339452624, 1.350356803686739, -1.350356803686739, 0.371069169726312, -0.742138339452624, 0.911435687134325, -0.911435687134325, -0.000000000000000, -0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, -0.496069169726312, 0.197856183019787, 0.000000000000000, 0.000000000000000, -0.371069169726312, 0.742138339452624, 0.371069169726312, -0.084648673840851, -0.072856183019786, 1.277500620666953, 0.814994522472410, -0.516781535765885, 0.072856183019786, -0.145712366039573, 1.568925352746098, -1.568925352746098, -0.657489665611773, -1.131788254627380 ],
 [0.000000000000000, 0.000000000000000, -0.496069169726312, 0.197856183019787, 0.000000000000000, 0.000000000000000, -0.072856183019786, 1.277500620666953, 0.072856183019786, -0.145712366039573, -0.371069169726312, 0.742138339452624, 0.814994522472410, -0.516781535765885, 0.371069169726312, -0.084648673840851, 1.568925352746098, -1.568925352746098, -1.131788254627380, -0.657489665611773 ],
 [0.000000000000000, 0.000000000000000, -0.197856183019786, 0.197856183019786, 0.000000000000000, 0.000000000000000, -0.072856183019786, 0.145712366039573, 0.072856183019786, -0.145712366039572, -0.072856183019786, 0.145712366039573, 0.455717843567163, -0.455717843567162, 0.072856183019786, -0.145712366039572, 2.700713607373479, -2.700713607373479, -0.000000000000001, -0.000000000000001 ],
 [0.000000000000000, 0.000000000000000, -0.361105248515956, 0.062858999514142, 0.000000000000000, 0.000000000000000, -0.096039507805726, 0.040365773828541, 0.096039507805726, -0.214477513256047, -0.317550346228776, 0.408454209829626, 1.221605566464057, -0.923359317462244, 0.317550346228776, -2.170260468072067, 0.143495819130218, -0.143495819130218, 0.174111739427505, 1.761806258242441 ],
 [0.000000000000000, 0.000000000000000, -0.361105248515956, 0.062858999514142, 0.000000000000000, 0.000000000000000, -0.317550346228776, 0.408454209829626, 0.317550346228776, -2.170260468072067, -0.096039507805726, 0.040365773828541, 1.221605566464057, -0.923359317462244, 0.096039507805726, -0.214477513256047, 0.143495819130218, -0.143495819130218, 1.761806258242441, 0.174111739427505 ],
 [0.000000000000000, 0.000000000000000, 0.800830446892597, 0.062858999514142, 0.000000000000000, 0.000000000000000, -0.317550346228776, -0.900059083156961, 0.317550346228776, -2.170260468072067, -0.317550346228776, -0.900059083156961, -1.896113083760515, 1.032423637353776, 0.317550346228776, -2.170260468072067, 1.452009112116805, -1.452009112116805, 3.070319551229027, 3.070319551229027 ],
 [0.000000000000000, 0.000000000000000, -0.361105248515956, -0.800830446892597, 0.000000000000000, 0.000000000000000, 1.246901150609823, 0.904055220235281, -1.246901150609823, 1.992152591566243, -0.317550346228776, 0.408454209829626, 0.357916120057318, 0.804019575351235, 0.317550346228776, 0.900059083156961, 3.213815370359246, -3.213815370359246, -2.896207811801523, -1.308513292986587 ],
 [0.000000000000000, 0.000000000000000, -0.361105248515956, -0.800830446892597, 0.000000000000000, 0.000000000000000, -0.317550346228776, 0.408454209829626, 0.317550346228776, 0.900059083156961, 1.246901150609823, 0.904055220235281, 0.357916120057318, 0.804019575351235, -1.246901150609823, 1.992152591566243, 3.213815370359246, -3.213815370359246, -1.308513292986587, -2.896207811801523 ],
 [0.000000000000000, 0.000000000000000, -0.062858999514142, -0.800830446892597, 0.000000000000000, 0.000000000000000, -0.317550346228776, 2.170260468072067, 0.317550346228776, 0.900059083156961, -0.317550346228776, 2.170260468072067, -1.032423637353777, 1.896113083760517, 0.317550346228776, 0.900059083156961, 1.452009112116805, -1.452009112116805, -3.070319551229028, -3.070319551229028 ],
 [0.000000000000000, 0.000000000000000, -0.062858999514142, 0.361105248515956, 0.000000000000000, 0.000000000000000, -0.317550346228776, 2.170260468072067, 0.317550346228776, -0.408454209829626, -0.096039507805726, 0.214477513256047, 0.923359317462244, -1.221605566464058, 0.096039507805726, -0.040365773828542, 0.143495819130218, -0.143495819130218, -1.761806258242441, -0.174111739427505 ],
 [0.000000000000000, 0.000000000000000, -0.361105248515956, 0.361105248515955, 0.000000000000000, 0.000000000000000, -0.096039507805726, 0.040365773828541, 0.096039507805726, -0.040365773828541, 1.246901150609823, 0.904055220235281, 0.726004556058402, -0.726004556058402, -1.246901150609823, -0.904055220235280, 0.317607558557724, -0.317607558557724, -0.000000000000000, -0.000000000000001 ],
 [0.000000000000000, 0.000000000000000, 0.800830446892597, 0.361105248515955, 0.000000000000000, 0.000000000000000, 1.246901150609823, -1.992152591566241, -1.246901150609823, -0.904055220235280, -0.317550346228776, -0.900059083156961, -0.804019575351234, -0.357916120057318, 0.317550346228776, -0.408454209829625, 3.213815370359246, -3.213815370359246, 2.896207811801521, 1.308513292986586 ],
 [0.000000000000000, 0.000000000000000, 0.800830446892597, 0.361105248515955, 0.000000000000000, 0.000000000000000, -0.317550346228776, -0.900059083156961, 0.317550346228776, -0.408454209829625, 1.246901150609823, -1.992152591566241, -0.804019575351234, -0.357916120057318, -1.246901150609823, -0.904055220235280, 3.213815370359246, -3.213815370359246, 1.308513292986586, 2.896207811801521 ],
 [0.000000000000000, 0.000000000000000, -0.062858999514142, 0.361105248515956, 0.000000000000000, 0.000000000000000, -0.096039507805726, 0.214477513256047, 0.096039507805726, -0.040365773828542, -0.317550346228776, 2.170260468072067, 0.923359317462244, -1.221605566464058, 0.317550346228776, -0.408454209829626, 0.143495819130218, -0.143495819130218, -0.174111739427505, -1.761806258242441 ],
 [0.000000000000000, 0.000000000000000, -0.361105248515956, 0.361105248515955, 0.000000000000000, 0.000000000000000, 1.246901150609823, 0.904055220235281, -1.246901150609823, -0.904055220235280, -0.096039507805726, 0.040365773828541, 0.726004556058402, -0.726004556058402, 0.096039507805726, -0.040365773828541, 0.317607558557724, -0.317607558557724, -0.000000000000001, -0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, 0.677732845596408, 0.236959372176209, 0.000000000000000, 0.000000000000000, 3.913918563152031, -2.537699446289976, -3.913918563152031, -0.533972191409811, -0.151401936280363, -0.131938654437497, -0.521829999815295, -0.392862217957323, 0.151401936280363, -0.027761984400733, 0.748710978843947, -0.748710978843947, 3.071671637699787, 0.159700638838231 ],
 [0.000000000000000, 0.000000000000000, 0.677732845596408, 0.236959372176209, 0.000000000000000, 0.000000000000000, -0.151401936280363, -0.131938654437497, 0.151401936280363, -0.027761984400733, 3.913918563152031, -2.537699446289976, -0.521829999815295, -0.392862217957323, -3.913918563152031, -0.533972191409811, 0.748710978843947, -0.748710978843947, 0.159700638838231, 3.071671637699787 ],
 [0.000000000000000, 0.000000000000000, 1.627508138018108, 0.236959372176209, 0.000000000000000, 0.000000000000000, -0.151401936280363, 0.577845760720719, 0.151401936280363, -0.027761984400733, -0.151401936280363, 0.577845760720719, -0.965395085227917, -0.899072424966400, 0.151401936280363, -0.027761984400733, 0.038926563685731, -0.038926563685731, -0.550083776319985, -0.550083776319985 ],
 [0.000000000000000, 0.000000000000000, 0.677732845596408, -1.627508138018108, 0.000000000000000, 0.000000000000000, -0.365100233556590, -0.673231936095658, 0.365100233556590, -2.948523477924114, -0.151401936280363, -0.131938654437497, -2.386297510009612, 3.336072802431313, 0.151401936280363, -0.577845760720719, 0.198627202523962, -0.198627202523962, 3.621755414019772, 0.709784415158216 ],
 [0.000000000000000, 0.000000000000000, 0.677732845596408, -1.627508138018108, 0.000000000000000, 0.000000000000000, -0.151401936280363, -0.131938654437497, 0.151401936280363, -0.577845760720719, -0.365100233556590, -0.673231936095658, -2.386297510009612, 3.336072802431313, 0.365100233556590, -2.948523477924114, 0.198627202523962, -0.198627202523962, 0.709784415158216, 3.621755414019772 ],
 [0.000000000000000, 0.000000000000000, -0.236959372176209, -1.627508138018108, 0.000000000000000, 0.000000000000000, -0.151401936280363, 0.027761984400733, 0.151401936280363, -0.577845760720719, -0.151401936280363, 0.027761984400733, 0.899072424966400, 0.965395085227917, 0.151401936280363, -0.577845760720719, 0.038926563685731, -0.038926563685731, 0.550083776319985, 0.550083776319985 ],
 [0.000000000000000, 0.000000000000000, -0.236959372176209, -0.677732845596408, 0.000000000000000, 0.000000000000000, -0.151401936280363, 0.027761984400733, 0.151401936280363, 0.131938654437497, 3.913918563152031, 0.533972191409811, 0.392862217957323, 0.521829999815295, -3.913918563152031, 2.537699446289975, 0.748710978843947, -0.748710978843947, -0.159700638838231, -3.071671637699786 ],
 [0.000000000000000, 0.000000000000000, 0.677732845596408, -0.677732845596408, 0.000000000000000, 0.000000000000000, 3.913918563152031, -2.537699446289976, -3.913918563152031, 2.537699446289975, -0.365100233556590, -0.673231936095658, 0.019463281842866, -0.019463281842866, 0.365100233556590, 0.673231936095658, 3.820382616543734, -3.820382616543734, -0.000000000000000, -0.000000000000000 ],
 [0.000000000000000, 0.000000000000000, 1.627508138018108, -0.677732845596408, 0.000000000000000, 0.000000000000000, -0.365100233556590, 2.948523477924114, 0.365100233556590, 0.673231936095658, -0.151401936280363, 0.577845760720719, -3.336072802431312, 2.386297510009612, 0.151401936280363, 0.131938654437497, 0.198627202523962, -0.198627202523962, -3.621755414019772, -0.709784415158216 ],
 [0.000000000000000, 0.000000000000000, 1.627508138018108, -0.677732845596408, 0.000000000000000, 0.000000000000000, -0.151401936280363, 0.577845760720719, 0.151401936280363, 0.131938654437497, -0.365100233556590, 2.948523477924114, -3.336072802431312, 2.386297510009612, 0.365100233556590, 0.673231936095658, 0.198627202523962, -0.198627202523962, -0.709784415158216, -3.621755414019772 ],
 [0.000000000000000, 0.000000000000000, -0.236959372176209, -0.677732845596408, 0.000000000000000, 0.000000000000000, 3.913918563152031, 0.533972191409811, -3.913918563152031, 2.537699446289975, -0.151401936280363, 0.027761984400733, 0.392862217957323, 0.521829999815295, 0.151401936280363, 0.131938654437497, 0.748710978843947, -0.748710978843947, -3.071671637699786, -0.159700638838231 ],
 [0.000000000000000, 0.000000000000000, 0.677732845596408, -0.677732845596408, 0.000000000000000, 0.000000000000000, -0.365100233556590, -0.673231936095658, 0.365100233556590, 0.673231936095658, 3.913918563152031, -2.537699446289976, 0.019463281842866, -0.019463281842866, -3.913918563152031, 2.537699446289975, 3.820382616543734, -3.820382616543734, -0.000000000000000, -0.000000000000000 ]
 ])

def compute_element_matrices(v, element_class,
                             constructor_args,
                             use_slip=False):
    element = element_class(*constructor_args)
    if use_slip:
        element.getMatricesSlip(v)
    else:
        element.getMatrices(v)
    return element

def parallel_get_matrices(element_class,
                          constructor_args,
                          v_list, processes=5,
                          chunksize=100,
                          use_slip=False,
                          use_tqdm=False):
    
    if sys.platform == "win32":
        ctx = get_context("spawn")
    else:
        ctx = get_context("fork")
    with ctx.Pool(processes=processes) as pool:
        func = partial(compute_element_matrices,
                       element_class=element_class,
                       constructor_args=constructor_args,
                       use_slip=use_slip)

        it = pool.imap(func, v_list, chunksize=chunksize)

        if use_tqdm:

          init(autoreset=True)  # Garante que a cor reseta automaticamente
          desc = f"{Fore.YELLOW} | Assembly{Style.RESET_ALL}"
          results = list(tqdm(it, total=len(v_list), desc=desc))
        else:
            results = list(it)
    return results
