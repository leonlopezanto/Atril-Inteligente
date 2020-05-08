# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:27:17 2019

@author: Antonio
"""

import numpy as np
import os
import Normalizador as norm

def crearDirectorioDataset(directorio):
    
    directorio = "./datasetProcesado/"+directorio
    try:
        os.stat(directorio)
    except:
        os.mkdir(directorio)

def crearDirectorioModelo(directorio):
    
    directorio = "./modelos/"+directorio
    try:
        os.stat(directorio)
    except:
        os.mkdir(directorio)
    
    path = directorio+'/'
    return path
        
#def salvarDatos(data = None, nombreArchivo,  path='./dataset/NotDefined', nombreDirectorio = ' ', crearDirectorio = False):
def salvarDatos(data = None, nombreArchivo=' ', nombreDirectorio = ' ', crearDirectorio = False, tipo='mono'):
    """ Guarda los datos en archivos .bin dentro de archivo separados.
        
    Parámetros:
        data -- valor del archivo a guardar (es una tupla)
        path -- ruta hasta el lugar donde se guardan los valores.
    
    """
    
    if(crearDirectorio):
        crearDirectorioDataset(nombreDirectorio)
    
    
    if data is None:
        print('No se ha introducido información')
    else:   
        path = "./datasetProcesado/"+nombreDirectorio+"/"+str(nombreArchivo) +".bin"
        with open(path,"bw") as dataFile:
            np.savetxt(dataFile, data,fmt='%.20e')


def loadFiles(type='ISO'):
    """Carga los archivos.
    
    Devuelve tres arrays con las rutas a los archivos de audio,
    los nombres de los archivos y las rutas a los archivos matLab 
    del tiempo.
    
    Parámetros:
    initial_dir -- Ruta base donde se encuentran los archivos.
    
    """
    arrayPath = []
    arrayFileName = []
#    arrayPathTxt = []
    arrayPathMidi = []
    if type == 'MUS':
        initial_dir = '../dataset/MusicalPieces'
    elif type == 'ISO':
        initial_dir = '../dataset/Isolated'
    elif type == 'CH':
        initial_dir = '../dataset/Chords'
        
    for root, _, files in os.walk(initial_dir):
        for i in range(len(files)):
#            if(files[i].find('.txt') != -1):
#                arrayPathTxt.append(os.path.join(root, files[i]))
            if(files[i].find('.mid') != -1):
                arrayPathMidi.append(os.path.join(root, files[i]))
            if(files[i].find('.wav') != -1):
                arrayPath.append(os.path.join(root, files[i]))
                arrayFileName.append(files[i])
    return arrayPath, arrayFileName, arrayPathMidi

def miniBatch(it, n=1):
    """
    Separa la matriz de datos en matrices de tamaño n y las va devolviendo por iteraciones
    Parametros:
        -it : Objeto sobre el que se va a iterar.
        -n : tamaño del minibatch
    """
    long = len(it) #Número de elementos de la matriz original.
    #Recorremos la matriz 
    #range(m, n, p)-> lista que empieza en m, acaba antes de n, y aumenta los valores de p en p.
    for i in range(0, long, n): #Devuelve una lista de números entre 0 y longitud, 
        yield it[i:min(i+n, long)]



def cargarDataset(option='completo',  name="", normalize=True, cargarPiezas=True):

    if(option == 'pruebas'):
        directoryTrain = './datasetProcesado/pruebas/train'
        directoryVal = './datasetProcesado/pruebas/val'
    elif(option == 'completo'):
        directoryTrain = './datasetProcesado/completo/train'
        directoryVal = './datasetProcesado/completo/val'
    
    
    X_train, Y_train = list(), list()
    #CARGAMOS LA MATRIZ DE TRAIN
    print("Cargando Train Data")
    numFiles = 1
    for root, d, files in os.walk(directoryTrain):
        print("Cargando {}\n Num: {} / {}".format(root, numFiles, len(d)))
        for i in range(len(files)):
            path = root +'/'+files[i]
            if(cargarPiezas):
                with open(path,"br") as datafiles:
                        #Cargamos los datos
                        if(path.find('_cqt') != -1):
                            data = np.loadtxt(datafiles)
                            X_train.append(data)
                            print(data.shape)
                        #Cargamos el target
                        if(path.find('_target') != -1):
                            target = np.loadtxt(datafiles)
                            Y_train.append(target)
                            print(target.shape)
                            numFiles +=1
            else:
                print(path)
                if(path.find('MUS') == -1):
                    with open(path,"br") as datafiles:
                            #Cargamos los datos
                            if(path.find('_cqt') != -1):
                                data = np.loadtxt(datafiles)
                                X_train.append(data)
                                print(data.shape)
                            #Cargamos el target
                            if(path.find('_target') != -1):
                                target = np.loadtxt(datafiles)
                                Y_train.append(target)
                                print(target.shape)
                                numFiles +=1                
                else:
                    print(path + " HA SIDO RECHAZADO!!")
            
            
    print("Matriz train cargada")
    X_train = np.hstack(X_train).T
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.hstack(Y_train).T
    Y_train = np.array(Y_train, dtype=np.float32)
    
    #Cargamos la matriz de validacion
    X_val, Y_val = list(), list()
    print("Cargando Val Data")
    numFiles = 1
    for root, d, files in os.walk(directoryVal):
        print("Cargando {}\n Num: {}".format(root, numFiles))
        for i in range(len(files)):
            path = root +'/'+files[i]
            if(cargarPiezas):
                with open(path,"br") as datafiles:
                    #Cargamos los datos
                    if(path.find('_cqt') != -1):
                        data = np.loadtxt(datafiles)
                        X_val.append(data)
                        print(data.shape)
                    #Cargamos el target
                    if(path.find('_target') != -1):
                        target = np.loadtxt(datafiles)
                        Y_val.append(target)
                        print(target.shape)
                        numFiles +=1
            else:
                print(path)
                if(path.find('MUS') == -1):
                    with open(path,"br") as datafiles:
                        #Cargamos los datos
                        if(path.find('_cqt') != -1):
                            data = np.loadtxt(datafiles)
                            X_val.append(data)
                            print(data.shape)
                        #Cargamos el target
                        if(path.find('_target') != -1):
                            target = np.loadtxt(datafiles)
                            Y_val.append(target)
                            print(target.shape)
                            numFiles +=1
                else:
                    print(path + " HA SIDO RECHAZADO!!")
            
    print("Matriz val cargada")
    X_val = np.hstack(X_val).T
    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.hstack(Y_val).T
    Y_val = np.array(Y_val, dtype=np.float32)
    
    print('train sequences: {arg_1}  |  train features: {arg_2}'.format(arg_1=X_train.shape[0], arg_2=X_train.shape[1]))
    print('validation sequences: {arg_1}  |  validation features: {arg_2}'.format(arg_1=X_val.shape[0], arg_2=X_val.shape[1]))

    
    #Calculamos los valores de Normalizacion:
    if(normalize):
        print("Calculando valores de Normalizacion...")
        normalizador = norm.Normalizador()
        
        #Diccionario para las caracteristicas del Normalizador.
        stat = dict()
        #El dataset se pasa por trozos para evitar desborde de RAM
        for idx in miniBatch(np.arange(X_train.shape[0]), 1000):
            stat['N'] = len(idx) #Numero de elementos en minibatch
            stat['mean'] = np.mean(X_train[idx, :], axis = 0)
            stat['S1'] = np.sum(X_train[idx,:], axis = 0)
            stat['S2'] = np.sum(X_train[idx,:] ** 2, axis = 0)
            normalizador.accumulate(stat) #Acumulamos.
            
        normalizador.finalize()
    
    print("Normalizando Train...")
    #Normalizamos con los valores acumulados el train y el val dataset.
    #Volvemos a pasar las features por partes.
    for idx in miniBatch(np.arange(X_train.shape[0]), 1000):
        X_train[idx, :] = normalizador.normalize(X_train[idx,:])        
    
    for idx in miniBatch(np.arange(X_val.shape[0]), 1000):
        X_val[idx, :] = normalizador.normalize(X_val[idx,:])
    
    print("Train y Val normalizado con exito")
        

    #Almacenamos los valores de la normalizacion.
    normalizador.saveValues(name)
    
    inputs = [X_train, Y_train, X_val, Y_val]
    
    return inputs


def cargarTest(option='completo',  name="", normalize=True, cargarPiezas=True):
    
    if(option == 'pruebas'):

        directoryTest = './datasetProcesado/pruebas/test'

    elif(option == 'completo'):
        directoryTest = './datasetProcesado/completo/test'

        
    #CARGAMOS LA MATRIZ DE TEST
    X_test, Y_test = list(), list()
    
    print("Cargando Test data")
    numFiles = 1
    for root, d, files in os.walk(directoryTest):
        print("Cargando {}\n Num: {}".format(root, numFiles))
        for i in range(len(files)):
            path = root +'/'+files[i]
            if(cargarPiezas):
                with open(path,"br") as datafiles:
                    #Cargamos los datos
                    if(path.find('_cqt') != -1):
                        data = np.loadtxt(datafiles)
                        X_test.append(data)
                        print(data.shape)
                    #Cargamos el target
                    if(path.find('_target') != -1):
                        target = np.loadtxt(datafiles)
                        Y_test.append(target)
                        print(target.shape)
                numFiles +=1
            else:
                if(path.find('MUS') == -1):
                    with open(path,"br") as datafiles:
                        #Cargamos los datos
                        if(path.find('_cqt') != -1):
                            data = np.loadtxt(datafiles)
                            X_test.append(data)
                            print(data.shape)
                        #Cargamos el target
                        if(path.find('_target') != -1):
                            target = np.loadtxt(datafiles)
                            Y_test.append(target)
                            print(target.shape)
                    numFiles +=1
                else:
                    print(path + " HA SIDO RECHAZADO!!")
    
    print("Matriz test cargada")
    X_test = np.hstack(X_test).T
    Y_test = np.hstack(Y_test).T
    
    
    normalizador = norm.Normalizador()
    
    path = './modelos/Model_'+name+'/'
    
    normalizador.loadValues(path)
    
    print("Normalizando Test...")
    #Normalizamos con los valores acumulados el train y el val dataset.
    #Volvemos a pasar las features por partes.
    for idx in miniBatch(np.arange(X_test.shape[0]), 1000):
        X_test[idx, :] = normalizador.normalize(X_test[idx,:])        
    

    print('Test sequences: {arg_1}  |  Test features: {arg_2}'.format(arg_1=X_test.shape[0], arg_2=X_test.shape[1]))
    print("Test normalizado.")
    
    return X_test, Y_test
    

def GuardarModelo(model, model_name='', path=''):
    """
    Guarda el modelo que se va a utilizar.
    Params:
        model -> Modelo a guardar
        model_name -> Nombre del modelo. Si no hay, se guarda con la fecha y la hora.
    Returns:
        Devuelve el nombre del modelo.
    """
    model_name = model_name + '.h5'
    print("Guardando modelo con nombre: {}".format(model_name))
    
    path = path + model_name
    model.save(path)
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    