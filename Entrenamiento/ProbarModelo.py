# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:36:26 2019

@author: Antonio López León

Se realizan pruebas unitarias con algunos archivos del dataset tras el entrenamiento

"""

from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import Normalizador as norm

def sliding_windows(X, sw = 15):

    padding = int(sw/2)
    X = np.vstack((np.zeros((padding, X.shape[1])), X, np.zeros((padding, X.shape[1]))))
    
    print('Dividiendo matriz en ventanas consecutivas de {}'.format(sw))
#    X = X.T
    Xwin = list()
    for i in range(X.shape[0]-(sw-1)):
        Xwin.append(X[i:i+sw, :])
        
    print('Done')
    
    Xwin = np.array(Xwin)
    Xwin = np.reshape(Xwin, (Xwin.shape[0], 1, Xwin.shape[1],  Xwin.shape[2]))
    return Xwin

def cargarModelo(name):
    """Carga un modelo a partir del nombre"""
    #Busca el modelo en el archivo de modelos disponibles.
    print('Buscando el modelo {}...'.format(name))
    dirModelos = './modelos'
    for root, directory, files in os.walk(dirModelos):
        for i in range(len(directory)):
            if(directory[i].find(name) != -1):
                print('\nCargando modelo...')
                path_model = dirModelos + '/Model_'+ name +'/' + name + '.h5'
                new_model = load_model(path_model)
                print('Modelo cargado. Mostrando Info.')
                new_model.summary()
                print('Cargando pesos...')
                name_weights = dirModelos + '/Model_'+ name +'/' + name + '_weights.h5'
                new_model.load_weights(name_weights)
                
                return new_model
    
    #Si no se carga el modelo, se devuelve -1
    print('Modelo no encontrado')
    return -1
    
def cargarArchivosPruebas():
    """Carga los archivos que se van a utilizar en la prueba"""
    dirPruebas = './archivosPrueba'
    print('Cargando archivos para pruebas...')
    for _, directory, _ in os.walk(dirPruebas):
        if len(directory) > 1:
            print('Archivos cargados correctamente')
            return directory
    
    print('NO HAY ARCHIVOS')
    return -1
    

def cargarDatos(directorio):
    '''Carga el cqt y el target de un archivo'''
    print("Cargando Datos...")
    X, Y = [], []
    numFiles = 1
    for root, directory, files in os.walk(directorio):
        print("Cargando {}\n Num: {} / {}".format(root, numFiles, len(directory)))
        for i in range(len(files)):
            path = root +'/'+files[i]
            with open(path,"br") as datafiles:
                
                
                #Cargamos los datos
                if(path.find('_cqt') != -1):
                    data = np.loadtxt(datafiles)
                    X.append(data)
                    print(data.shape)
                #Cargamos el target
                if(path.find('_target') != -1):
                    target = np.loadtxt(datafiles)
                    Y.append(target)
                    print(target.shape)
        numFiles +=1
        
    print("Matriz train cargada")
    X = np.hstack(X).T
    Y = np.hstack(Y).T
        
    return X, Y

def cargarNormalizador(modelName):
    normalizador = norm.Normalizador()
    path = './modelos/Model_' + modelName + '/'
    normalizador.loadValues(path)
    
    return normalizador


def obtenerTipo(archivo):
    if(archivo.find('MUS')):
        tipo = 'MUS'
    elif(archivo.find('RAND')):
        tipo = 'RAND'
    elif(archivo.find('ISOL')):
        tipo = 'ISOL'
    elif(archivo.find('UCHO')):
        tipo = 'UCHO'
    else:
        tipo = -1
        
    return tipo


def mostrarDatos(modelName, input, tipo='', target = False, title=''):
    '''Crea imagenes para comparar las pruebas del modelo'''
    input.astype(float)
    fig = plt.figure(0, figsize=(10,8))
    plt.imshow(input, aspect="auto")
    plt.title(title, loc='center')
    if target:
        nombre = tipo+'_target'
    else:
        nombre = tipo + '_pred_'
    #crea la carpeta donde guardar las imagenes si no existe
    directorio = './modelos/Model_' + modelName + '/resultadosPrueba/' 
    try:
        os.stat(directorio)
    except:
        os.mkdir(directorio)
    path = directorio + nombre
    fig.savefig(path, bbox_inches='tight')
    plt.show()


def realizarPruebas(modelo, archivos, name, umbral, ws=0):
    '''Se realizan las pruebas del modelo sobre los archivos'''
    
    #Carga normalizador
    normalizador = cargarNormalizador(name)
    n=1
    for i in range(len(archivos)):
        directorio = './archivosPrueba/' + archivos[i]
        cqt, target = cargarDatos(directorio)
        print('Datos cargados')
        cqt = normalizador.normalize(cqt)
        if ws:
            cqt = sliding_windows(cqt, ws)
        pred = modelo.predict(cqt)
        pred = 1.0 * (np.squeeze(pred) > umbral )
        pred = pred.T
        #Obtiene el tipo de archivo:
        tipo = obtenerTipo(archivos[i])
        if(tipo == -1): 
            print('TIPO DESCONOCIDO')
            tipo = 'UNKNOWN'
        
        mostrarDatos(name, pred, tipo+str(n), False, directorio)
        target = target.T
        mostrarDatos(name, target, tipo+str(n), True, directorio)
        n = n+1
        
            
        


