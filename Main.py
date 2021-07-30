import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
import getopt
import Export
from Mesh import mesh
from FEM_sci import FEM
from Fluid import Fluid
from particleCloud import ParticleCloud
from SetCase import Case


i = 0
end = 100
dt = 0.1
case = os.path.basename(os.path.normpath(os.getcwd()))
# case = './Cases/cylinder/cylinder'

argv = sys.argv[1:]
try:
    options, args = getopt.getopt(argv, "i:e:t:f:",["init=","end=","dt=","file="])

except:
    print("invalid inputs!")

for name, value in options:
    if name in ['-i', '--init']:
        i = int(value)
    elif name in ['-e', '--end']:
        end = int(value)
    elif name in ['-t', '--dt']:
        dt =float(value) 
    elif name in ['-f', '--file']:
        case = os.path.splitext(os.path.abspath(value))[0]

if end <= i:
	end = i + 100


path = os.path.abspath(case) + '.xml'
tree = ET.parse(path)
root = tree.getroot()
if os.path.exists(os.path.dirname(os.path.abspath(case)) + '/' + os.path.splitext(root.attrib['name'])[0] + '.msh'):
    msh = os.path.dirname(os.path.abspath(case)) + '/' + os.path.splitext(root.attrib['name'])[0]
else:
    msh = os.path.splitext(os.path.abspath(root.attrib['name']))[0]


output_dir = os.path.abspath(os.path.dirname(os.path.abspath(case)) + '/Results')

mesh_kind = Case.set_kind(root)
MESH = mesh(msh,mesh_kind)

Case.read(case,MESH)

IC,forces = Case.set_IC(i)
Re,Pr,Ga,Gr,Fr,Da,Fo,particles_flag,two_way,porous = Case.set_parameters()
BC = Case.set_BC()

outflow = Case.set_OutFlow()
   
MESH.set_boundary_prior(BC,outflow)

fluid = Fluid(MESH,Re,Pr,Ga,Gr,Da,Fo,IC)

if particles_flag:
    x_part, v_part, d_part, rho_part, nLoop = Case.set_particles(i)
    
    particleCloud = ParticleCloud(MESH.elem_list,MESH.node_list,x_part,v_part,d_part,rho_part,forces,two_way)


t = np.arange(i,end,dt)

FEM.set_matrices(MESH,fluid,dt,BC,porous)

#----------------------------------------SL Gustavo ----------------------------------------------------------------
# neighborElem = [[]]
# neighborElem = [[] for k in range(MESH.npoints_p)]
# for k in range(0,MESH.IEN_orig.shape[0]):
#  for j in range(0,MESH.IEN_orig.shape[1]):
#   v = MESH.IEN_orig[k][j]
#   neighborElem[v].append(k)
  
# oface = -1*np.ones( (MESH.ne,3),dtype='int' )

# for e in range(len(MESH.IEN_orig)):
#     for iter in range(3):
#         ID = MESH.IEN_orig[e,iter]
#         point = MESH.node_list[ID]
#         for op in range (len(point.opostos)):
#             if point.aresta_correspondente[op][0] in MESH.IEN_orig[e] and  point.aresta_correspondente[op][1] in MESH.IEN_orig[e]:
#                 oface[e,iter] = point.opostos[op]
 #----------------------------------------------------------------------------------------------------------------------------------
print('Output directory ---> ' + output_dir)
if not os.path.isdir(output_dir):
   os.mkdir(output_dir)

if not i == 0:
    i += 1
while i < end:
    
    # fluid = FEM.solve_fields(True,neighborElem,oface)
    if particles_flag:
        fluid = FEM.solve_fields(particleCloud.forces)
        particleCloud.solve(dt,nLoop,fluid.Re,1.0/np.sqrt(fluid.Ga))
    else:
        fluid = FEM.solve_fields(np.zeros((MESH.npoints,2), dtype='float'))
        particleCloud = 0
        
    
    Export.export_data(i,output_dir,fluid,MESH,particleCloud)

    i+=1



