import numpy as np
import matplotlib.pyplot as plt

x_ = np.linspace(0,1)
y_ = np.linspace(0,1)

x, y = np.meshgrid(x_,y_)


N1 = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N2 = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N3 = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N4 = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')

N1_ = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N2_ = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N3_ = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N4_ = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N5_ = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')
N6_ = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')


z = np.zeros((x.shape[0],x.shape[1]), dtype = 'float')

for i in range (x.shape[0]):
    for j in range (x.shape[1]):
        z[i][j] = 1.0 - x[i][j] - y[i][j]
        if y[i][j] > 1.0 - x[i][j]:
            N1[i][j] = float('NaN')
            N2[i][j] = float('NaN')
            N3[i][j] = float('NaN')
            N4[i][j] = float('NaN')
            
            N1_[i][j] = float('NaN')
            N2_[i][j] = float('NaN')
            N3_[i][j] = float('NaN')
            N4_[i][j] = float('NaN')
            N5_[i][j] = float('NaN')
            N6_[i][j] = float('NaN')
        else:
            N1[i][j] = x[i][j] - 9.0*x[i][j]*y[i][j]*z[i][j]
            N2[i][j] = y[i][j] - 9.0*x[i][j]*y[i][j]*z[i][j]
            N3[i][j] = z[i][j] - 9.0*x[i][j]*y[i][j]*z[i][j]
            N4[i][j] = 27.0*x[i][j]*y[i][j]*z[i][j]
            
            N1_[i][j] = (2*x[i][j] - 1)*x[i][j]
            N2_[i][j] = (2*y[i][j] - 1)*y[i][j]
            N3_[i][j] = (2*z[i][j] - 1)*z[i][j]
            N4_[i][j] = 4.0*x[i][j]*y[i][j]
            N5_[i][j] = 4.0*z[i][j]*y[i][j]
            N6_[i][j] = 4.0*x[i][j]*z[i][j]
            

plt.close('all')
fig = plt.figure(1)
ax = fig.add_subplot(221, projection='3d')
surf = ax.plot_surface(x,y,N1)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(222, projection='3d')
surf = ax.plot_surface(x,y,N2)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(223, projection='3d')
surf = ax.plot_surface(x,y,N3)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(224, projection='3d')
surf = ax.plot_surface(x,y,N4)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')


fig = plt.figure(2)
ax = fig.add_subplot(321, projection='3d')
surf = ax.plot_surface(x,y,N1_)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(322, projection='3d')
surf = ax.plot_surface(x,y,N2_)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(323, projection='3d')
surf = ax.plot_surface(x,y,N3_)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(324, projection='3d')
surf = ax.plot_surface(x,y,N4_)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(325, projection='3d')
surf = ax.plot_surface(x,y,N5_)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')

ax = fig.add_subplot(326, projection='3d')
surf = ax.plot_surface(x,y,N6_)
ax.set_xlabel(r'$L_1$')
ax.set_ylabel(r'$L_2$')
        
            