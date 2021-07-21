import numpy as np

def sameSide(p1,p2,a,b):
    cp1 = np.cross(b-a,p1-a)
    cp2 = np.cross(b-a,p2-a)
    if np.dot(cp1,cp2)>=0:
        return True
    else:
        return False

def pointInTriangle(p,a,b,c):
    if sameSide(p,a,b,c) and sameSide(p,b,a,c) and sameSide(p,c,a,b):
        return True
    else:
        return False
    
def pointInElement(p,v1,v2,v3):
    
    a = np.array([v1.x,v1.y])
    b = np.array([v2.x,v2.y])
    c = np.array([v3.x,v3.y])
    if sameSide(p,a,b,c) and sameSide(p,b,a,c) and sameSide(p,c,a,b):
        return True
    else:
        return False

def Area(p1,p2,p3):
    Matriz = np.array([[1,p1[0],p1[1]],
                        [1,p2[0],p2[1]],
                        [1,p3[0],p3[1]]],dtype='float')
    A = 0.5*np.linalg.det(Matriz)
    return np.abs(A)
     
def calc_vd(IEN,IENbound,X,Y,vx,vy,npoints_p,dt):

    Xd = X - dt*vx
    Yd = Y - dt*vy
    vxd = np.zeros( (len(X)),dtype='float' )
    vyd = np.zeros( (len(X)),dtype='float' )
    
    #cálculo para os vértices dos triângulos
    for n in range(len(X[0:npoints_p])):
        if n in IENbound:
            vxd[n] = vx[n]
            vyd[n] = vy[n]
            continue
        for e in range(len(IEN)):
            if n in IEN[e]:
                if pointInTriangle(np.array([Xd[n],Yd[n]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]])):
                    vxd[n] = (Area(np.array([Xd[n],Yd[n]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]))*vx[IEN[e,2]] +\
                          Area(np.array([Xd[n],Yd[n]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vx[IEN[e,1]]+\
                              Area(np.array([Xd[n],Yd[n]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vx[IEN[e,0]])/\
                        Area(np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))
                              
                    vyd[n] = (Area(np.array([Xd[n],Yd[n]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]))*vy[IEN[e,2]] +\
                          Area(np.array([Xd[n],Yd[n]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vy[IEN[e,1]]+\
                              Area(np.array([Xd[n],Yd[n]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vy[IEN[e,0]])/\
                                Area(np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))  

                    break
 
    #cálculo para os centroides
    for e in range(len(IEN)):
        vxd[IEN[e,3]] = (Area(np.array([X[IEN[e,3]],Y[IEN[e,3]]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]))*vx[IEN[e,2]] +\
                          Area(np.array([X[IEN[e,3]],Y[IEN[e,3]]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vx[IEN[e,1]]+\
                              Area(np.array([X[IEN[e,3]],Y[IEN[e,3]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vx[IEN[e,0]])/\
                                Area(np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))  
    
        vyd[IEN[e,3]] = (Area(np.array([X[IEN[e,3]],Y[IEN[e,3]]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]))*vy[IEN[e,2]] +\
                          Area(np.array([X[IEN[e,3]],Y[IEN[e,3]]]),np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vy[IEN[e,1]]+\
                              Area(np.array([X[IEN[e,3]],Y[IEN[e,3]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))*vy[IEN[e,0]])/\
                                Area(np.array([X[IEN[e,0]],Y[IEN[e,0]]]),np.array([X[IEN[e,1]],Y[IEN[e,1]]]),np.array([X[IEN[e,2]],Y[IEN[e,2]]]))  

    return vxd, vyd


def baricentric_v(point,point1, point2, point3,centroide=False):
    p = point
    p1 = np.array([point1.x,point1.y])
    p2 = np.array([point2.x,point2.y])
    p3 = np.array([point3.x,point3.y])
    
    A = Area(p1,p2,p3)
    A1 = Area(p,p2,p3)
    A2 = Area(p,p1,p3)
    A3 = Area(p,p1,p2)
    
    if centroide:
        vx = (A1*point1.vxd + A2*point2.vxd + A3*point3.vxd)/A
        vy = (A1*point1.vyd + A2*point2.vyd + A3*point3.vyd)/A
        T = 0
    else:
        vx = (A1*point1.vx + A2*point2.vx + A3*point3.vx)/A
        vy = (A1*point1.vy + A2*point2.vy + A3*point3.vy)/A
        T = (A1*point1.T + A2*point2.T + A3*point3.T)/A
    return vx, vy, T

def baricentric(point,element):
    p = point
    p1 = np.array([element.nodes[0].x,element.nodes[0].y])
    p2 = np.array([element.nodes[1].x,element.nodes[1].y])
    p3 = np.array([element.nodes[2].x,element.nodes[2].y])
    
    A = Area(p1,p2,p3)
    A1 = Area(p,p2,p3)
    A2 = Area(p,p1,p3)
    A3 = Area(p,p1,p2)

    L1 = A1/A
    L2 = A2/A
    L3 = A3/A

    vx = (L1*element.nodes[0].vx + L2*element.nodes[1].vx + L3*element.nodes[2].vx)
    vy = (L1*element.nodes[0].vy + L2*element.nodes[1].vy + L3*element.nodes[2].vy)
    T = (L1*element.nodes[0].T + L2*element.nodes[1].T + L3*element.nodes[2].T)

    return vx, vy, T

def baricentric_quad(point,element):
    p = point
    p1 = np.array([element.nodes[0].x,element.nodes[0].y])
    p2 = np.array([element.nodes[1].x,element.nodes[1].y])
    p3 = np.array([element.nodes[2].x,element.nodes[2].y])
    
    A = Area(p1,p2,p3)
    A1 = Area(p,p2,p3)
    A2 = Area(p,p1,p3)
    A3 = Area(p,p1,p2)

    L1 = A1/A
    L2 = A2/A
    L3 = A3/A

    N1 = (2*L1-1.0)*L1
    N2 = (2*L2-1.0)*L2
    N3 = (2*L3-1.0)*L3
    N4 = 4*L1*L2
    N5 = 4*L2*L3
    N6 = 4*L1*L3

    edges = element.edges

    vx = N1*element.nodes[0].vx + N2*element.nodes[1].vx + N3*element.nodes[2].vx + N4*edges[0].vx + N5*edges[1].vx + N6*edges[2].vx
    vy = N1*element.nodes[0].vy + N2*element.nodes[1].vy + N3*element.nodes[2].vy + N4*edges[0].vy + N5*edges[1].vy + N6*edges[2].vy
    T = L1*element.nodes[0].T + L2*element.nodes[1].T + L3*element.nodes[2].T 

    return vx, vy, T

def baricentric_v2(point,point1, point2, point3,centroide):
    p = point
    p1 = np.array([point1.x,point1.y])
    p2 = np.array([point2.x,point2.y])
    p3 = np.array([point3.x,point3.y])
    
    A = Area(p1,p2,p3)
    A1 = Area(p,p2,p3)
    A2 = Area(p,p1,p3)
    A3 = Area(p,p1,p2)
    
    L1 = A1/A
    L2 = A2/A
    L3 = A3/A
    
    # N1 = L1 - 9.0*L1*L2*L3
    # N2 = L2 - 9.0*L1*L2*L3
    # N3 = L3 - 9.0*L1*L2*L3
    # N4 = 27.0*L1*L2*L3
    

    # vx = N1*point1.vx + N2*point2.vx + N3*point3.vx + N4*centroide.vx
    # vy = N1*point1.vy + N2*point2.vy + N3*point3.vy + N4*centroide.vy
    
    vx = L1*point1.vx + L2*point2.vx + L3*point3.vx
    vy = L1*point1.vy + L2*point2.vy + L3*point3.vy 
    
    return vx, vy
    

def semi_lagrange(nodes,elements,vx,vy,dt,IENbound):
    vxd = np.zeros( (len(vx)),dtype='float' )
    vyd = np.zeros( (len(vx)),dtype='float' )
    xd = np.zeros( (len(vx)),dtype='float' )
    yd = np.zeros( (len(vx)),dtype='float' )
    
    for point in nodes:
        flag_out = True
        point.vx = vx[point.ID]
        point.vy = vy[point.ID]
        if point.ID in IENbound and not point.out_flow:
            vxd[point.ID] = vx[point.ID]
            vyd[point.ID] = vy[point.ID]
            flag_out = False
            continue
        if not point.centroide:
            xd[point.ID] = point.x - vx[point.ID]*dt
            yd[point.ID] = point.y - vy[point.ID]*dt
            point.xd = xd[point.ID]
            point.yd = yd[point.ID]
    
            for dupla in point.lista_pontos:
                p1 = nodes[dupla[0]]
                p2 = nodes[dupla[1]]
                if pointInElement(np.array([xd[point.ID],yd[point.ID]]),p1,p2,point):
                    vxd[point.ID], vyd[point.ID] = baricentric_v(np.array([xd[point.ID],yd[point.ID]]),point,p1,p2)
                    flag_out = False
        elif point.centroide:
            flag_out = False
        
        if flag_out:
            # print('fora')
            elemento = 0
            elemento_ant = 0
            elemento_2ant = 0
            vetor_dist = []
            vizinhos = np.reshape(point.lista_pontos,2*len(point.lista_pontos))
            for vizinho in vizinhos:
                vetor_dist.append(point.dist_d(nodes[vizinho]))

            vizinho_maislonge = nodes[vizinhos[vetor_dist.index(np.max(vetor_dist))]]
            for elem in vizinho_maislonge.opostos:
                if point in elements[elem].nodes:
                    elemento = elements[elem]
            
            counter = 0
            while (counter < 100):
                if counter >= 90:
                    flag_bound = True
                    # print('counter =' + str(counter))
                    # print('x=' + str(point.x) + ', y=' + str(point.y))
                    for e in elements:
                        elemento = e
                        if pointInElement(np.array([xd[point.ID],yd[point.ID]]), elemento.nodes[0], elemento.nodes[1], elemento.nodes[2]):
                            vxd[point.ID], vyd[point.ID] = baricentric_v(np.array([xd[point.ID],yd[point.ID]]), elemento.nodes[0], elemento.nodes[1], elemento.nodes[2])
                            flag_bound = False
                            break
                    if flag_bound:
                        break
                vetor_dist = []
                if pointInElement(np.array([xd[point.ID],yd[point.ID]]), elemento.nodes[0], elemento.nodes[1], elemento.nodes[2]):
                    vxd[point.ID], vyd[point.ID] = baricentric_v(np.array([xd[point.ID],yd[point.ID]]), elemento.nodes[0], elemento.nodes[1], elemento.nodes[2])
                    break
                for v in elemento.nodes:
                    vetor_dist.append(point.dist_d(v))
                vizinho_maislonge = elemento.nodes[vetor_dist.index(np.max(vetor_dist))]
                # vetor_dist[vetor_dist.index(np.max(vetor_dist))] -= 1000.0
                # segundo_vizinho_maislonge = elemento.nodes[vetor_dist.index(np.max(vetor_dist))]
                for aresta in vizinho_maislonge.lista_pontos:
                    if nodes[aresta[0]] in elemento.nodes and nodes[aresta[1]] in elemento.nodes:
                        flag_bound = True
                        for elem in vizinho_maislonge.opostos:
                            if nodes[aresta[0]] in elements[elem].nodes and nodes[aresta[1]] in elements[elem].nodes:
                                # print (elem)
                                # print('x_viz=' + str(vizinho_maislonge.x) + ', y_viz=' + str(vizinho_maislonge.y))
                                # print('x=' + str(point.x) + ', y=' + str(point.y))
                                # print (np.array([xd[point.ID],yd[point.ID]]))
                                elemento_2ant = elemento_ant
                                elemento_ant = elemento
                                elemento = elements[elem]  
                                if elemento == elemento_2ant:
                                    # if counter > 90:
                                    #     print('e_2ant= ' + str(elemento_2ant.ID))
                                    #     print('e_ant= ' + str(elemento_ant.ID))
                                    #     print('e= ' + str(elemento.ID))
                                    for node in elemento.nodes + elemento_ant.nodes:
                                        # print(elemento.nodes + elemento_ant.nodes)
                                        # print('node = ' + str(node.ID) + ' lista_elem = ' + str(node.lista_elem))
                                        # print('node = ' + str(node))
                                        flag_break = False
                                        for e in node.lista_elem:                                          
                                            if pointInElement(np.array([xd[point.ID],yd[point.ID]]), elements[e].nodes[0], elements[e].nodes[1], elements[e].nodes[2]):
                                                elemento = elements[e]
                                                flag_break = True
                                                break
                                        if flag_break:
                                            break
                                        else:
                                            max_dist = 0
                                            for ponto in elemento.nodes:
                                                if ponto != vizinho_maislonge:
                                                    dist = point.dist_d(ponto)
                                                    if dist > max_dist:
                                                        max_dist = dist
                                                        for el in ponto.opostos:
                                                            for aresta_ in ponto.lista_pontos:
                                                                if nodes[aresta_[0]] in elemento.nodes and nodes[aresta_[1]] in elemento.nodes:
                                                                    if nodes[aresta_[0]] in elements[el].nodes and nodes[aresta_[1]] in elements[el].nodes:                
                                                                        elemento = elements[el]
                                                                        # if counter > 90:
                                                                        #     print('ponto = ' + str(ponto.ID))
                                                                        #     print('ponto mais longe= ' + str(vizinho_maislonge.ID))
                                                                        #     print('elemento = ' + str(elemento.ID))
                                                                        break
                                                                
                                flag_bound = False
                                break
                        break
                    
                if flag_bound:
                    # print('bound')
                    vxd[point.ID] = vx[aresta[0]]
                    vyd[point.ID] = vy[aresta[0]]
                    break
                counter += 1
                
                
                
            
    num = 0
    for e in nodes[0].IEN:
        vxd[e[3]], vyd[e[3]] = baricentric_v(np.array([elements[num].centroide.x,elements[num].centroide.y]),elements[num].nodes[0],elements[num].nodes[1],elements[num].nodes[2])
        num += 1
        
    return vxd, vyd


def semi_lagrange2(nodes,elements,vx,vy,T,dt,IENbound):
    vxd = np.zeros( (len(vx)),dtype='float' )
    vyd = np.zeros( (len(vx)),dtype='float' )
    Td = np.zeros( (len(vx)),dtype='float' )
    xd = np.zeros( (len(vx)),dtype='float' )
    yd = np.zeros( (len(vx)),dtype='float' )
    
    if IENbound.shape[1] > 2:
        interp = lambda point_, element_: baricentric_quad(point_,element_)
    else:
        interp = lambda point_, element_: baricentric(point_,element_)

    for point in nodes:
        point.vx = vx[point.ID]
        point.vy = vy[point.ID]
        if point.ID < len(T):
            point.T = T[point.ID]
    for point in nodes:

        if point.ID in IENbound and not point.out_flow:
            vxd[point.ID] = vx[point.ID]
            vyd[point.ID] = vy[point.ID]
            if not point.edge:
                Td[point.ID] = T[point.ID]
                point.Td = Td[point.ID]
            point.vxd = vxd[point.ID]
            point.vyd = vyd[point.ID]
           
            continue
        if not point.centroide and not point.edge:
            xd[point.ID] = point.x - vx[point.ID]*dt
            yd[point.ID] = point.y - vy[point.ID]*dt
            point.xd = xd[point.ID]
            point.yd = yd[point.ID]
            
            count = 0

            if point.edge:
                p = nodes[elements[point.lista_elem[0]].edges_dict[point.ID]]
            else:
                p = point
            
            p_ant = 0
            p_2ant = 0
            while count < 100:
                count += 1
                flag_break = False
                for e in p.lista_elem:
                    if pointInElement(np.array([xd[point.ID],yd[point.ID]]), elements[e].nodes[0], elements[e].nodes[1], elements[e].nodes[2]):
                        vxd[point.ID], vyd[point.ID], Td[point.ID] = interp(np.array([xd[point.ID],yd[point.ID]]), elements[e])
                        point.vxd = vxd[point.ID]
                        point.vyd = vyd[point.ID]
                        point.Td = Td[point.ID]
                        flag_break = True
                        break
                if flag_break:
                    break
                
                dist_min = 1000000
                
                p_2ant = p_ant
                p_ant = p
                p_aux = p
                for vizinho in np.reshape(p_aux.lista_pontos,2*len(p_aux.lista_pontos)):
                    dist = point.dist_d(nodes[vizinho])
                    if dist < dist_min:
                        dist_min = dist
                        p = nodes[vizinho]
                if p == p_2ant and p.ID in IENbound:
                    # print('caiu aqui ---' + str(point.ID))
                    # print('p = ' + str(p.ID))
                    # print('p_ant = ' + str(p_ant.ID))
                    # print('p_2ant = ' + str(p_2ant.ID))
                    vxd[point.ID] = vx[p.ID]
                    vyd[point.ID] = vy[p.ID]
                    Td[point.ID] = T[p.ID]
                    point.vxd = vxd[point.ID]
                    point.vyd = vyd[point.ID]
                    break

        # if point.ID == 507:
        #     print('x = ' + str(point.x))
        #     print('y = ' + str(point.y))
        #     print('xd = ' + str(point.xd))
        #     print('yd = ' + str(point.yd))
        #     print('vx = ' + str(point.vx))
        #     print('vy = ' + str(point.vy))
        #     print('vxd = ' + str(point.vxd))
        #     print('vyd = ' + str(point.vyd))
        #     print('e = ' + str(e))
        #     print(elements[e].nodes[0].vx)
        #     print(elements[e].nodes[1].vx)
        #     print(elements[e].nodes[2].vx)
                    
                    
                        
    if IENbound.shape[1] == 2:
        vxd_ = vxd[nodes[0].IEN_orig]
        vyd_ = vyd[nodes[0].IEN_orig]
        Td_ = Td[nodes[0].IEN_orig]

        vxd[len(T):] = vxd_.sum(axis=1)/3.0
        vyd[len(T):] = vyd_.sum(axis=1)/3.0
        Td[len(T):] = Td_.sum(axis=1)/3.0

    elif IENbound.shape[1] == 3:

        for i in range (3,6):
            j=i-2
            if i == 5:
                j = 0
            vxd[nodes[0].IEN[:,i]] = (vxd[nodes[0].IEN[:,j]] + vxd[nodes[0].IEN[:,i-3]])/2.0
            vyd[nodes[0].IEN[:,i]] = (vyd[nodes[0].IEN[:,j]] + vyd[nodes[0].IEN[:,i-3]])/2.0
            Td[nodes[0].IEN[:,i]] = (Td[nodes[0].IEN[:,j]] + Td[nodes[0].IEN[:,i-3]])/2.0


    # num = 0
    # for e in nodes[0].IEN:
    #     vxd[e[3]], vyd[e[3]],Td[e[3]] = baricentric_v(np.array([elements[num].centroide.x,elements[num].centroide.y]),elements[num].nodes[0],elements[num].nodes[1],elements[num].nodes[2],True)
    #     num += 1
    
        
    return vxd, vyd, Td
        
def semi_lagrange3(nodes,elements,vx,vy,dt,IENbound):
    vxd = np.zeros( (len(vx)),dtype='float' )
    vyd = np.zeros( (len(vx)),dtype='float' )
    xd = np.zeros( (len(vx)),dtype='float' )
    yd = np.zeros( (len(vx)),dtype='float' )
    
    for point in nodes:
        point.vx = vx[point.ID]
        point.vy = vy[point.ID]
        if point.ID in IENbound and not point.out_flow:
            vxd[point.ID] = vx[point.ID]
            vyd[point.ID] = vy[point.ID]
            point.vxd = vxd[point.ID]
            point.vyd = vyd[point.ID]
            continue
    
        xd[point.ID] = point.x - vx[point.ID]*dt
        yd[point.ID] = point.y - vy[point.ID]*dt
        point.xd = xd[point.ID]
        point.yd = yd[point.ID]
        
        count = 0
        
        if not point.centroide:
            p = point
        else:
            p = nodes[point.lista_pontos[0]]
            # dist_min = 1000000
            # for n in point.lista_pontos:
            #     dist = point.dist_d(nodes[n])
            #     if dist < dist_min:
            #         dist_min = dist
            #         p = nodes[n]
        
        p_ant = 0
        p_2ant = 0
        while count < 100:
            # print(str(point.ID) + ' count =' + str(count))
            count += 1
            flag_break = False
            for e in p.lista_elem:
                if pointInElement(np.array([xd[point.ID],yd[point.ID]]), elements[e].nodes[0], elements[e].nodes[1], elements[e].nodes[2]):
                    vxd[point.ID], vyd[point.ID] = baricentric_v2(np.array([xd[point.ID],yd[point.ID]]), elements[e].nodes[0], elements[e].nodes[1], elements[e].nodes[2], nodes[elements[e].centroide.ID])
                    point.vxd = vxd[point.ID]
                    point.vyd = vyd[point.ID]
                    # print('vx = ' + str(vx[point.ID]))
                    # print('vxd = ' + str(vxd[point.ID]))
                    flag_break = True
                    break
            if flag_break:
                break
            
            dist_min = 1000000
            
            p_2ant = p_ant
            p_ant = p
            p_aux = p
            for vizinho in np.reshape(p_aux.lista_pontos,2*len(p_aux.lista_pontos)):
                dist = point.dist_d(nodes[vizinho])
                if dist < dist_min:
                    dist_min = dist
                    p = nodes[vizinho]
            if p == p_2ant:
                # print('caiu aqui ---' + str(point.ID))
                # print('p = ' + str(p.ID))
                # print('p_ant = ' + str(p_ant.ID))
                # print('p_2ant = ' + str(p_2ant.ID))
                vxd[point.ID] = (vx[p.ID] + vx[p_2ant.ID])/2.0
                vyd[point.ID] = (vy[p.ID] + vy[p_2ant.ID])/2.0
                point.vxd = vxd[point.ID]
                point.vyd = vyd[point.ID]
                break

        # if point.ID == 358:
        #     print('x = ' + str(point.x))
        #     print('y = ' + str(point.y))
        #     print('xd = ' + str(point.xd))
        #     print('yd = ' + str(point.yd))
        #     print('vx = ' + str(point.vx))
        #     print('vy = ' + str(point.vy))
        #     print('vxd = ' + str(point.vxd))
        #     print('vyd = ' + str(point.vyd))
        #     print('nodes = ' + str(np.array([elements[e].nodes[0].ID, elements[e].nodes[1].ID, elements[e].nodes[2].ID])))
        #     print(elements[e].nodes[0].vx)
        #     print(elements[e].nodes[1].vx)
        #     print(elements[e].nodes[2].vx)
                    
    
        
    return vxd, vyd                        
                    
                
            

        
        
        
