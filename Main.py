import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
import getopt
import Export
import ExportSolid
from Mesh import mesh
from FEM_sci import FEM
from SolidFEM import FEM as SolidFEM
from Fluid import Fluid
from particleCloud import ParticleCloud
from SetCase import Case

def main():
    
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
    
    type_ = ''
    output_dir = os.path.abspath(os.path.dirname(os.path.abspath(case)) + '/Results')
    
    mesh_kind = Case.set_kind(root)
    porous_list, limit_name, smooth_value, porosity = Case.set_porous_region(root)
    solid_list = Case.set_solid_region(root)
    MESH = mesh(msh,mesh_kind,porous_list,solid_list, limit_name, smooth_value, porosity)
    
    Case.read(case,MESH)
    
    IC,forces = Case.set_IC(i)
    Re,Pr,Ga,Gr,Fr,Da,Fo,Ma,particles_flag,two_way,COO_flag,porous,turb, SolidProp, mesh_factor, fluid_steps, n_save, compress_output = Case.set_parameters()
    extension = '(VTK)'
    if compress_output:
        extension = '(VTU)'
        
    BC,FSI = Case.set_BC()
    
    outflow = Case.set_OutFlow()
       
    MESH.set_boundary_prior(BC,outflow,FSI)
    
    fluid = Fluid(MESH,Re,Pr,Ga,Gr,IC,Da,Fo,Ma)
    
    if particles_flag:
        x_part, v_part, d_part, rho_part, nLoop, inlet, lims, mean, sigma, factor, type_, freq, dist, rho, max_part, num_method = Case.set_particles(i)
        
        particleCloud = ParticleCloud(MESH.elem_list,MESH.node_list,x_part,v_part,d_part,rho_part,forces,two_way,num_method)
    
        particleCloud.set_distribution(mean, sigma, factor, inlet, type_, freq, dist, rho, lims, max_part)
    
    
    FEM.set_matrices(MESH,fluid,dt,BC,IC,SolidProp,COO_flag,porous,turb)
    neighborElem=[[]]
    SL_matrix=True
    oface=[]
    #----------------------------------------SL Matrix ----------------------------------------------------------------
    if SL_matrix:
        if MESH.IEN.shape[1] > 4: #for quad elements
            neighborElem = [[] for k in range(MESH.npoints)]
            for k in range(0,MESH.IEN.shape[0]):
              for j in range(0,MESH.IEN.shape[1]):
                  v = MESH.IEN[k][j]
                  neighborElem[v].append(k)
        
        else:
            neighborElem = [[] for k in range(MESH.npoints_p)]
            for k in range(0,MESH.IEN_orig.shape[0]):
              for j in range(0,MESH.IEN_orig.shape[1]):
                  v = MESH.IEN_orig[k][j]
                  neighborElem[v].append(k)
          
        oface = -1*np.ones( (MESH.ne,3),dtype='int' )
        
        for e in range(len(MESH.IEN_orig)):
            for iter in range(3):
                ID = MESH.IEN_orig[e,iter]
                point = MESH.node_list[ID]
                for op in range (len(point.opostos)):
                    if point.aresta_correspondente[op][0] in MESH.IEN_orig[e] and  point.aresta_correspondente[op][1] in MESH.IEN_orig[e]:
                        oface[e,iter] = point.opostos[op]
     #----------------------------------------------------------------------------------------------------------------------------------
    print('Output directory ---> ' + output_dir)
    if not os.path.isdir(output_dir):
       os.mkdir(output_dir)
    
    
    if not i == 0:
        i += 1
    while i < end:
        
        # fluid = FEM.solve_fields(True,neighborElem,oface)
        if len(FEM.mesh.FSI) == 0:
            if particles_flag:
                fluid = FEM.solve_fields(i,particleCloud.forces,SL_matrix,neighborElem,oface)
                particleCloud.solve(dt,nLoop,fluid.Re,1.0/np.sqrt(fluid.Ga))
                if type_ == "continuous":
                    if i == 0:
                        f = open(os.path.abspath(os.path.dirname(os.path.abspath(case)) + '/exhaust.txt'), 'w')
                    else:
                        f = open(os.path.abspath(os.path.dirname(os.path.abspath(case)) + '/exhaust.txt'), 'a')
                    f.write(str(particleCloud.count_exit) + '\n')
                    f.close()
                # else:
                #     if i == 0:
                #         f2 = open(os.path.abspath(os.path.dirname(os.path.abspath(case)) + '/position.txt'), 'w')
                #     else:
                #         f2 = open(os.path.abspath(os.path.dirname(os.path.abspath(case)) + '/position.txt'), 'a')
                #     f2.write(str(particleCloud.particle_list[0].pos[0]) + ' ' + str(particleCloud.particle_list[0].pos[1]) + '\n')
                #     f2.close()
            else:
        
                for j in range(fluid_steps):
                    fluid = FEM.solve_fields(i,np.zeros((MESH.npoints,2), dtype='float'), dt/fluid_steps, SL_matrix,neighborElem,oface)
                particleCloud = 0
            
            if n_save != 0 and i%n_save == 0:
                Export.export_data(i,output_dir,fluid,MESH,particleCloud,compress_output)
                print ('--------Time step = ' + str(i) + ' --> saving solution ' + extension + '--------\n')
            else:
                print ('--------Time step = ' + str(i) + ' --------\n')
        
        else:
            n = 0 
            error = 1000.0
            while error >= 1e-5 and n<1:
                # print ('--------Start of convergence iteration ' + str(n) + ' --------\n')
                # u_prime_x_ant = SolidFEM.u_prime_x
                # FSIForces_ant = FEM.fluid.FSIForces[:,0].copy()
                # ux_ant = SolidFEM.ux
                # ux_relax_ant = SolidFEM.ux_relax
                # vx_ant = fluid.vx
                for j in range(fluid_steps):
                    fluid = FEM.solve_fields(i,np.zeros((MESH.npoints,2), dtype='float'), dt/fluid_steps, SL_matrix,neighborElem,oface,n,SolidProp['Fluid_conv'])
                particleCloud = 0
                if i>=SolidProp['Fluid_conv']:
                    k=i-10
                    if SolidProp['HE']:
                        u = SolidFEM.solve_HE(k,mesh_factor,False,n)
                    else:
                        u, u_w = SolidFEM.solve(k,mesh_factor,False,n)
                else:
                    u = SolidFEM.u
                # error = np.sqrt(sum((SolidFEM.u_prime_x - u_prime_x_ant)**2))
                # error = np.linalg.norm(SolidFEM.u_prime_x - u_prime_x_ant)/np.linalg.norm(SolidFEM.u_prime_x)
                # error = max(abs(u - u_ant))
                # error = np.sqrt(sum((FEM.fluid.FSIForces[:,0] - FSIForces_ant)**2))
                # error = max(abs(fluid.FSIForces[:,0] - FSIForces_ant))

                # print ('----------Iteration ' + str(n) + ': Error = ' + str(error) + '---------\n')
                n+=1

            SolidFEM.update_fluidmesh(mesh_factor)
            
            if n_save != 0 and i%n_save == 0:
                ExportSolid.export_data(FEM.solidMesh, output_dir,u,SolidFEM.u_prime, SolidFEM.u_doubleprime, SolidFEM.sigma_x,SolidFEM.sigma_y, SolidFEM.tau_xy, SolidFEM.PK_stress_x, SolidFEM.PK_stress_y, SolidFEM.PK_stress_xy, SolidFEM.sigma_VM, i, compress_output)
                Export.export_data(i,output_dir,fluid,MESH,particleCloud,compress_output)
                print ('--------Time step = ' + str(i) + ' --> saving solution ' + extension + '--------\n')
            else:
                print ('--------Time step = ' + str(i) + ' --------\n')
                            
        
        if len(FEM.mesh.FSI_list) > 0:
            f = open(output_dir + '/FSI_Cd.txt', 'a')    
            f.write(str(i*dt)+ '\t' + str(FEM.Cd) + '\t' + str(FEM.Cl) + '\n')
            f.close()
        
        i+=1
    
    
    # if type_ == "continuous":
    #     f.close()

if __name__ == "__main__":
    main()
