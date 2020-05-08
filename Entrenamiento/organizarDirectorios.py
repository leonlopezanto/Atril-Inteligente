# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 01:36:51 2019

@author: Antonio
"""
import shutil
import random as r
from pathlib import Path
import os

def organizarDirectorios(origen='./datasetProcesado', ubicacion='./datasetProcesado/completo'):
    """Permite reorganizar archivos en directorios para Train y Test
    
    80% de los archivos para Train y validación
    20% de los archivos para Test
    
    """
    
    #Leemos todos los archivos
    directorio = []
    
    for root, directory, files in os.walk(origen):
        for i in range(len(directory)):
            if(directory[i].find('MAPS') != -1):
                directorio.append(os.path.join(root, directory[i]))


    print(len(directorio))
    
    #Escogemos entre el total y sin repetir
    rand = list(range(0, len(directorio)))
    #Shuffle
    rand = r.sample(rand, len(rand))
    
    train = int(round(0.6*len(rand))) #80% train
    test = int(round(0.2*len(rand))) #10% test
    val = int(round(0.2*len(rand))) #10% val
    print("Nº ficheros para train: {}/{}".format(train, len(rand)))
    print("Nº ficheros para val: {}/{}".format(val, len(rand)))
    print("Nº ficheros para test: {}/{}".format(test, len(rand)))
    #Escogemos un 80% de archivos al azar para train y los movemos a ./datasetProcesados/completo/train
    numD = 0
    for i in range(len(directorio)):
        if(numD<train):
            ubiFinal = ubicacion +'/train'
        elif (numD > train) and  (numD < (test+train)):
            ubiFinal = ubicacion +'/test'
        else:
            ubiFinal = ubicacion + '/val'
        
        #Se mueve el directorio a la ubicacion correspondiente
        

        shutil.move(directorio[rand[i]], ubiFinal)
#        else:
#            print("OJO")
        numD +=1

    print("Archivos movidos con éxito. {}".format(len(directorio)))



def crearPruebas(ratio=0.1):
    """ Para crear pruebas cogeremos un 20% de los archivos de train, test y val.
    """
    
    origenTrain = './datasetProcesado/completo/train'
    origenVal = './datasetProcesado/completo/val'
    origenTest = './datasetProcesado/completo/test'
    
    
    directorio = list()
    
    directorio.append(origenTrain)
    directorio.append(origenVal)
    directorio.append(origenTest)
    
    destino = list()
    destinoTrain = './datasetProcesado/pruebas/train'
    destinoVal = './datasetProcesado/pruebas/val'
    destinoTest = './datasetProcesado/pruebas/test'
    destino.append(destinoTrain)
    destino.append(destinoVal)
    destino.append(destinoTest)
    
    
    #Recorremos los directorios y cogemos un 20% de sus archivos de manera aleatoria.
    for i in range(len(directorio)):  
        
        for root, directory, files in os.walk(directorio[i]):
            
            #Escogemos entre el total y sin repetir
            total = list(range(0, round(len(directory))))
            #Shuffle
            totales = round(len(total)*ratio)
            rand = r.sample(total, totales)
            
            
            for k in range(len(rand)):
                if(directory[rand[k]].find('MAPS') != -1):
                    src = root + '/' + directory[rand[k]] 
                    dest = destino[i]+'/'+directory[rand[k]] 
                    shutil.copytree(src, dest)
                    
