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


# os.add_dll_directory(os.path.abspath('C:\Program Files\FEM simulator\Bin'))
# os.add_dll_directory(os.getcwd())

# dll = ct.cdll.LoadLibrary("MathModule.dll")

i = 0
end = i + 300
dt = 0.1

argv = sys.argv[1:]
try:
    options, args = getopt.getopt(argv, "i:e:t:",["init=","end=","dt="])

except:
    print("invalid inputs!")

for name, value in options:
    if name in ['-i', '--init']:
        i = int(value)
    elif name in ['-e', '--end']:
        end = int(value)
    elif name in ['-t', '--dt']:
        dt =float(value) 



if len(args) >= 1:
    case = os.path.splitext(os.path.abspath(args[0]))[0]
else:   
    case = os.path.basename(os.path.normpath(os.getcwd()))
    # case = './Cases/cylinder/cylinder'

path = os.path.abspath(case) + '.xml'
tree = ET.parse(path)
root = tree.getroot()
if os.path.exists(os.path.dirname(os.path.abspath(case)) + '/' + os.path.splitext(root.attrib['name'])[0] + '.msh'):
    msh = os.path.dirname(os.path.abspath(case)) + '/' + os.path.splitext(root.attrib['name'])[0]
else:
    msh = os.path.splitext(os.path.abspath(root.attrib['name']))[0]

output_dir = os.path.abspath(os.path.dirname(os.path.abspath(case)) + '/Results')

MESH = mesh(msh)

Case.read(case,MESH)
    
outflow = Case.set_OutFlow()
IC = Case.set_IC(i)
Re,Pr,Ga,Gr,Fr,particles_flag = Case.set_parameters()
BC = Case.set_BC()
   
MESH.set_boundary_prior(BC,outflow)

fluid = Fluid(MESH,Re,Pr,Ga,Gr,IC)

if particles_flag:
    x_part, d_part, rho_part, nLoop = Case.set_particles(i)
    
    particleCloud = ParticleCloud(MESH.elem_list,MESH.node_list,x_part,d_part,rho_part)

t = np.arange(i,end,dt)
# dt=t[1]-t[0]

FEM.set_matrices(MESH,fluid,dt,BC)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# if len(sys.argv) > 1:
#     i = int(sys.argv[1])
# else:
#     i = 0

if not i == 0:
    i += 1
while i < end:
    
    fluid = FEM.solve_fields()
    
    x_part=np.array([])
    if particles_flag:
        x_part = particleCloud.solve(dt,nLoop,fluid.Re,1.0/np.sqrt(fluid.Ga))
    
    Export.export_data(i,output_dir,fluid,MESH,x_part)

    i+=1



