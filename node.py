import numpy as np

class Node:
    
    node_list = []
        
    def __init__(self,ID,IEN,IEN_orig,x,y):
        
        self.ID = ID
        self.IEN = IEN
        self.IEN_orig = IEN_orig
        self.x = x
        self.y = y
        self.xd = 0
        self.yd = 0
        self.out_flow = False
        
        self.vx = 0
        self.vy = 0
        self.T = 0
        self.vxd = 0
        self.vyd = 0
        self.Td = 0
        
        if self.ID in IEN_orig:
            self.centroide = False
            self.edge = False
            self.lista_pontos, self.lista_elem = self.vizinhos()
            # self.opostos, self.aresta_correspondente = self.elem_opostos()
        elif self.IEN.shape[1] == 4:
            self.centroide = True
            self.edge = False
            # for e in range(len(IEN)):
            #     if self.ID in IEN[e]:
            #         self.lista_pontos = IEN[e,0:3]
            #         self.elem = e
            #         break
        else:
            self.edge = True
            self.centroide = False
            place = np.where(self.IEN == self.ID)
            self.lista_elem = []
            for i in range (len(place[0])):
                elem = place[0][i]
                self.lista_elem.append(elem)

        Node.node_list.append(self)
        
    def vizinhos(self):
        
        lista_pontos = []
        lista_elem = []
        
        place = np.where(self.IEN_orig==self.ID)
        for i in range (len(place[0])):
            elem = place[0][i]
            lista_elem.append(elem)
            sublista = []
            for n in range (len(self.IEN_orig[elem])):
                if self.IEN_orig[elem,n] != self.ID:
                    sublista.append(self.IEN_orig[elem,n])
            lista_pontos.append(sublista)
            
        # num = 0
        # for elem in self.IEN_orig:           
        #     if self.ID in elem:
        #         lista_elem.append(num)
        #         sublista = []
        #         for n in range (len(elem)):
        #             if elem[n] != self.ID:
        #                 sublista.append(elem[n])
        #         lista_pontos.append(sublista)
        #     num += 1
            
        return lista_pontos, lista_elem
    
    def elem_opostos(self):
        lista_opostos = []
        aresta_correspondente = []
        for lista in self.lista_pontos:
            elems1 = np.where(self.IEN_orig==lista[0])[0]
            elems2 = np.where(self.IEN_orig==lista[1])[0]

            for i in range (len(elems1)):
                for j in range (len(elems2)):
                    if elems1[i] == elems2[j] and elems1[i] not in self.lista_elem:
                        lista_opostos.append(elems1[i])
                        aresta_correspondente.append(lista)

        # num = 0
        # lista_opostos = []
        # aresta_correspondente = []
        # for e in self.IEN_orig:
        #     for lista in self.lista_pontos:
        #         if lista[0] in e and lista[1] in e and self.ID not in e:
        #             lista_opostos.append(num)
        #             aresta_correspondente.append(lista)
        #     num += 1
        
        return lista_opostos, aresta_correspondente
        
        
    def dist(self,ponto):
        d = np.sqrt((self.x - ponto.x)**2 + (self.y - ponto.y)**2)
        return d
    
    def dist_d(self,ponto):
        d = np.sqrt((self.xd - ponto.x)**2 + (self.yd - ponto.y)**2)
        return d