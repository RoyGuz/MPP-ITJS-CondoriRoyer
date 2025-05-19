import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import gmsh


def ResolverSistemaGmesh (Dato,EmpotradoY, EmpotradoX, TraccionadoX, NodoCentro):
    #.........................  SE SACAN LAS MATRICES DE NODOS Y CONECTIVIDADES NECESARIAS ...............................

    # ........................  NODOS ....................................................................................
    #   Se toma la matriz de nodos 
    Nodos = gmsh.model.mesh.get_nodes()         #   Con esto tomo la informacion de los nodos que genera gmesh
    numNodos = Nodos[0].shape[0]                #   Con esto tomo la cantidad de nodos que genero gmesh

    #   Con esto formo mi matriz de nodos original, se forma como [x,y,z]
    MN_original = Nodos[1].reshape(numNodos,3)

    #   Defino mi matriz MN tal cual la voy a utilizar en mis funciones, tomo solamente las dos primeras columans [x,y]
    MN = MN_original[:, :2]

    # ........................  ELEMENTOS ................................................................................

    Elementos = gmsh.model.mesh.get_elements_by_type(2)     #   Con esto tomo la informacion de los elementos generados con gmesh 
    numElementos = Elementos[0].shape[0]                    #   Con esto tomo la cantidad de elementos que genero gmses

    MC_original = Elementos[1].reshape(numElementos,3).astype(int)  #   La notacion con la que se genera esto no es lo indicado. Reformulo para Python

    #   Defino mi matriz de conectividades tal cual la voy a utilizar en mi programa
    MC = MC_original-1

    #.........................................................................................................................

    #   Como ya tengo la matriz de nodos y conectividades, me queda armar la matriz b que es mi matriz que tiene las condiciones de CC

    #.................................................. NODOS EMPOTRADOS .......................................................................................

    nodosEmpotradosY = gmsh.model.mesh.get_nodes_for_physical_group(1,EmpotradoY)     # Tomo la infomacion de los nodos sobe la linea que defini como empotrada.
    #   Tambien se tiene informacion sobre las coordenadas de estos nodos ... pero no me interesa en este momento

    numNodosEmpotrados = nodosEmpotradosY[0].shape[0]

    NEy_original = nodosEmpotradosY[0].reshape(numNodosEmpotrados,1).astype(int)        # Tomo los indices de los nodos que me interesan, pero estos no estan en la notacion de Python ...

    NEy = NEy_original-1

    nodosEmpotradosX = gmsh.model.mesh.get_nodes_for_physical_group(1,EmpotradoX)     # Tomo la infomacion de los nodos sobe la linea que defini como empotrada.
    #   Tambien se tiene informacion sobre las coordenadas de estos nodos ... pero no me interesa en este momento

    numNodosEmpotrados = nodosEmpotradosX[0].shape[0]

    NEx_original = nodosEmpotradosX[0].reshape(numNodosEmpotrados,1).astype(int)        # Tomo los indices de los nodos que me interesan, pero estos no estan en la notacion de Python ...

    NEx = NEx_original-1

    NCentral = gmsh.model.mesh.get_nodes_for_physical_group(0, NodoCentro)        # Tomo la informacion del nodo central ... que me molesta

    NCentral =NCentral[0].astype(int) - 1

    # --------------- Esto toma los nodos que considero van a tener condicion de empotramiento ---------------

    #.................................................. NODOS TRACCIONADOS .......................................................................................

    nodosTraccionados = gmsh.model.mesh.get_nodes_for_physical_group(1,TraccionadoX)     # Tomo la infomacion de los nodos sobe la linea que defini como tracionada.

    #   Tambien se tiene informacion sobre las coordenadas de estos nodos ... pero no me interesa en este momento

    numNodosTraccionados = nodosTraccionados[0].shape[0]

    NT_original = nodosTraccionados[0].reshape(numNodosTraccionados,1).astype(int)        # Tomo los indices de los nodos que me interesan, pero estos no estan en la notacion de Python ...

    NT = NT_original-1
    NT = np.hstack((NT, np.zeros((NT.shape[0], 1))))    #   Esto es para agregar una columna de ceros ---- 

    #print(NCentral)

    #.............................................................................................................................................................

    #   Este es el bloque mas confuso ... pero se sale (creo)
    #   .....................................................................................
    entityTraccionada=gmsh.model.getEntitiesForPhysicalGroup(1,TraccionadoX)
    Tgroup,Ttraccionada,Ltraccionada=gmsh.model.mesh.getElements(1,entityTraccionada[0])
    #   .....................................................................................

    #   Ttraccionada: Me devuelve los elementos de mi sistema que estan sobre la linea EmpotradoX ...   #   Se debe pasar a la notacion de PYTHON 
    #   LTraccionada: Me devuelve una matriz de conectividades ... con los nodos que estan conectados sobre esa linea ... (Es la que sirve)

    Ltraccionada=Ltraccionada[0].reshape(Ttraccionada[0].shape[0],2) #es la matriz de conectividad de los elementos linea del extremo traccionado
    Ltraccionada = Ltraccionada - 1 

    #   Estraigo la matriz de intereses y la agrupo como matriz ... [nodoi nodoj; .... .... ] Son los nodos que estna conectados sobre la linea de interes

    #   .............. MATRIZ QUE GUARDA LA LONGITUD DE LOS ELEMENTOS CONDIDERADOS ..............

    #   Como voy a trabajar con cargas distribuidas ... la fuerza no es la misma en los nodos, varia en funcion de la pocision en la linea de interes 

    Longitudes=np.abs(MN[Ltraccionada[:,0],1]-MN[Ltraccionada[:,1],1])

    #...............................................................................................................................................................

    #   Este bloque es para dar un valor de fuerza a cada nodo .... este valor va a depender de la longitud de cada segmento que une dos nodos de la linea de interes
    FuerzaT = Dato.Tension / ( Dato.ancho * Dato.espesor )     #   Paso el valor de Tension a fuerza ... Divido por el area

    for lin, linea in enumerate(Ltraccionada):        #   Se itera sobre la matriz de conectividades del elemento de interes
        
        #   Tomo los nodos de interes en cada iteracion
        nodo1 = int(linea[0])
        nodo2 = int(linea[1])

        f_nodo = FuerzaT * Longitudes[lin] * Dato.espesor / 2    #   PREGUNTAR POR ESTA DEFINICION ... tiene una deduccion que vimos pero no recuerdo
        
        NT[NT[:,0]==nodo1,1] += f_nodo
        NT[NT[:,0]==nodo2,1] += f_nodo

    #......................................................................................................................................................

    #   Necesito definir a aquellos nodos sobre los cuales no impuse ninguna condicion .... No lo utilizo ... pero me parece util tenerlo

    NSC = np.arange(numNodos)       #   Creo mi vector para los nodos que no tendran ninguna condicion ....

    NSC = NSC[~np.isin(NSC, NEx)]            #   Elimino los nodos empotrados
    NSC = NSC[~np.isin(NSC, NEy)]
    NSC = NSC[~np.isin(NSC, NCentral)]
    NSC = NSC[~np.isin(NSC, NT[:,0])]       #   Elimino los nodos traccionados

    #..........................................................................................................................................

    #   ....................    AHORA SI ... MI MATRIZ b PARA MIS FUNCIONES .....................................

    b = np.zeros([numNodos*2,2])

    #   Pimero los nodos co condicion de empotramiento

    #   La primera columna me indica el tipo de dato que conozco
    #       Si es 0 se conoce la fuerza
    #       Si es 1 se conoce el desplazamiento

    for i,j in enumerate(NEx):
        b[2*j,0] = 1

    for r,t in enumerate(NEy):
        b[2*t+1,0] = 1

    for yy,y in enumerate(NCentral):
        b[2*y,0] = 1
        b[2*y+1,0] = 1    

    for n,m in enumerate(NT):
        b[int(2*m[0]),1] = m[1]

    #print(b)

    #......................................................................................................................................

    return MN, MC, b, Nodos, Elementos