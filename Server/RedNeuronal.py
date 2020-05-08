# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:41:50 2019

@author: Antonio
"""
import Normalizador as norm
from keras.models import load_model
import os
import numpy as np


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def Normalizador(modelName, cqt):
    '''
    Carga y normaliza la señal CQT
    :param modelName: Nombre del modelo
    :param cqt: datos CQT
    :return: CQT normalizada
    '''

    normalizador = norm.Normalizador()

    #Carga valores de normalizacion
    normalizador.loadValues('./NeuralNetwork/')
    #Normaliza
    cqt = normalizador.normalize(cqt)
    
    return cqt

def cargarModelo(name):
    '''
    Carga el modelo neuronal y carga los pesos
    :param name: Nombre del modelo
    :return: Modelo cargado
    '''

    dirModelos = './NeuralNetwork/'
    for root, directory, files in os.walk(dirModelos):
        for i in range(len(files)): #len(files)
            if(files[i].find(name) != -1):
                print('\nCargando modelo...')
                path_model = dirModelos + name + '.h5'

                new_model = load_model(path_model)
                # print('Modelo cargado. Mostrando Info.')
                # new_model.summary()

                # print('Cargando pesos...')
                name_weights = dirModelos + name + '_weights.h5'
                new_model.load_weights(name_weights)
                
                # print('Devolvemos el modelo')
                
                return new_model
    
    #Si no se carga el modelo, se devuelve -1
    print('Modelo no encontrado')
    return -1

def predecir(model, cqt, umbral=0.9):
    '''
    Realiza la predicción en base a la CQT
    :param model: Modelo preparado para predecir
    :param cqt: señal cqt normalizada
    :param umbral: límite para limpiar predicción
    :return:
    '''

    #Predicción
    pred = model.predict(cqt)
    pred = 1.0 * (np.squeeze(pred) > umbral )

    return pred.T
    