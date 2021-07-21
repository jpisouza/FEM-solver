## =================================================================== ##
#  this is file testElement.py, created at 09-Jun-2021                #
#  maintained by Gustavo R. Anjos                                       #
#  e-mail: gustavo.rabello@coppe.ufrj.br                                #
## =================================================================== ##

import Elements
import matplotlib.pyplot as plt
import meshio
import numpy as np

plt.close('all')

# elemento linear
X = [0.0,1.0,0.0]
Y = [0.0,0.0,1.0]
lin = Elements.Linear(X,Y)
lin.getM([0,1,2])
# plot triangulo linear

plt.figure(1)
plt.plot(X,Y,'bo')
Xt=X[:3];Xt.append(X[0])
Yt=Y[:3];Yt.append(X[0])
plt.plot(Xt,Yt,'k-')
for i in range(0,3):
 plt.text(X[i]+0.02,Y[i]+0.03,str(i),color='r')
plt.show()
# print (lin.kxx+lin.kyy)
# print (lin.mass)

#--------------------------------------------------
# # loop do elemento MINI
# lin = Elements.Linear(X,Y)
# for e in range(0,ne):
#  v1,v2,v3,v4 = IEN[e]
# 
#  lin.getM([v1,v2,v3,v4])
#  
#  for ilocal in range(0,4):
#   for jlocal in range(0,4):
#    K[iglobal,jglobal] += lin.kxx[ilocal,jlocal] + lin.kyy[ilocal,jlocal]
#    M[iglobal,jglobal] += lin.mass[ilocal,jlocal]
#   for jlocal in range(0,3):
#    G[iglobal,jglobal] += lin.gx[ilocal,jlocal]
#    G[iglobal+numNodes,jglobal] += lin.gy[ilocal,jlocal]
#-------------------------------------------------- 

#print (lin.area)

# elemento quadratico
msh = meshio.read('cilindro_quad.msh')
X_ = np.array([0.48577091, 0.47715651, 0.41354408, 0.48146371, 0.4453503,  0.44965749])
Y_ = np.array([0.19393359, 0.12702769, 0.15530544, 0.16048064, 0.14116656, 0.17461951])
X = list(X_)
Y = list(Y_)

print(msh.cells['triangle6'][16])

quad = Elements.Quad(X,Y)
quad.getM([0,1,2,3,4,5])

# plot triangulo quadratico
plt.figure(2)
plt.plot(X,Y,'bo')
Xt=X[:3];Xt.append(X[0])
Yt=Y[:3];Yt.append(Y[0])
plt.plot(Xt,Yt,'k-')
for i in range(0,6):
  plt.text(X[i]+0.0002,Y[i]+0.0003,str(i),color='r')

plt.show()

print (quad.kxx+quad.kyy)
print (quad.mass)

#print (quad.area)

# elemento quadratico+bolha (quadb)
X = [0.0,1.0,0.0,0.5,0.5,0.0,1/3]
Y = [0.0,0.0,1.0,0.0,0.5,0.5,1/3]
quadb = Elements.QuadBubble(X,Y)
quadb.getM([0,1,2,3,4,5,6])

# plot triangulo quadratico+bolha (quadb)
plt.figure(3)
plt.plot(X,Y,'bo')
Xt=X[:3];Xt.append(X[0])
Yt=Y[:3];Yt.append(X[0])
plt.plot(Xt,Yt,'k-')
for i in range(0,7):
 plt.text(X[i]+0.02,Y[i]+0.03,str(i),color='r')
plt.show()

# print (quadb.area)

# elemento cubico
X = [0.0,1.0,0.0,1/3,2/3,2/3,1/3,0.0,0.0,1/3]
Y = [0.0,0.0,1.0,0.0,0.0,1/3,2/3,2/3,1/3,1/3]
cubic = Elements.Cubic(X,Y)
cubic.getM([0,1,2,3,4,5,6,7,8,9])

# plot triangulo cubico
plt.figure(4)
plt.plot(X,Y,'bo')
Xt=X[:3];Xt.append(X[0])
Yt=Y[:3];Yt.append(X[0])
plt.plot(Xt,Yt,'k-')
for i in range(0,10):
 plt.text(X[i]+0.02,Y[i]+0.03,str(i),color='r')
plt.show()

# print (cubic.mass)
