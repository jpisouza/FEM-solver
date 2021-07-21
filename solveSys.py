import subprocess
import numpy as np

def createMatrixFile(m):
    with open('Matrix.txt', 'w+') as var:
        for i in range (m.shape[0]):
            for j in range (m.shape[1]):
                var.write(str(m[i,j]) + '\t')
            var.write('\n')

def createVectorFile(v):
    with open('Vector.txt', 'w+') as var:
        for i in range (v.shape[0]):
            var.write(str(v[i]) + '\n')
            
def solveSystem(vector):
    createVectorFile(vector)
    
    metodo = subprocess.Popen('Math.exe', \
                        creationflags=subprocess.SW_HIDE, shell=True)
    metodo.wait()
    
    x = np.array(np.loadtxt("result.txt", dtype='float', delimiter='\n'))
    metodo.terminate()
    
    return x
