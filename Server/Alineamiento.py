# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:27:15 2019

@author: Antonio
"""

import librosa
import pretty_midi as pm
import dtw
import gc
from keras import backend as K
import RedNeuronal as rn
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import mposTools
import copy

tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

#Liberamos ram
gc.collect()

#Constantes
FS = 44100
HOP = 512
NOTE_START = 21
N_NOTES = 88
NBINS = 290


def obtenerCQT(wavFiles):
    '''
    Convierte audio a CQT
    :param wavFiles: Archivo de audio a convertir
    :return: CQT transpuesto
    '''

    print("Archivo audio : {}".format(wavFiles))

    X = list()

    #Carga el audio 
    x, sr = librosa.load(wavFiles, sr = FS)

    """Función para extraer el valor CQT del audio"""
    C = librosa.cqt(x, sr=sr, fmin=librosa.midi_to_hz(NOTE_START),
                    n_bins=NBINS, bins_per_octave=36, hop_length = HOP)
    
    cqt = np.abs(C)

    print("Shape data: {}".format(cqt.shape))
    return cqt.T


def sliding_windows(X, sw=21):
    '''
    Crea ventanas deslizantes en caso de red convolucional
    :param X: CQT
    :param sw: Número de ventanas
    :return: Ventanas
    '''
    padding = int(sw / 2)
    X = np.vstack((np.zeros((padding, X.shape[1])), X, np.zeros((padding, X.shape[1]))))

    print('Dividiendo matriz en ventanas consecutivas de {}'.format(sw))
    Xwin = list()
    for i in range(X.shape[0] - (sw - 1)):
        Xwin.append(X[i:i + sw, :])

    X = np.array(Xwin, dtype=np.float32)
    print('Done')

    X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
    print(type(X), type(X[0]))
    return X



def Alinear(pathAudio, pathMidi, mposFile):
    '''
    Se realiza el alineamiento audio - partitura
    :param pathAudio:  Ruta del archivo de audio
    :param pathMidi: Ruta del archivo MIDI
    :param mposFile: Ruta del archivo MPOS
    :return: Vector de tiempos de inicio de compás
    '''

    print('Alineando...')
    K.clear_session()

    #Cargamos el archivo midi
    # print('Cargando MIDI')
    midi = pm.PrettyMIDI(pathMidi)
    #Obtenemos el piano_roll
    piano_roll = midi.get_piano_roll(fs=FS/HOP)[NOTE_START:(NOTE_START+N_NOTES)]

    #Obtenemos la transformada CQT del audio.
    # print('Calculando CQT')
    cqt = obtenerCQT(pathAudio)

    #Carga de modelo neuronal
    #Nombre del modelo entrenado:
    modelName = '28-9_15-20'
    arq = 'mlp'

    #Cargamos el normalizador y normalizamos la señal cqt
    # print('Normalizando CQT')
    cqt = rn.Normalizador(modelName, cqt)

    #cargamos el modelo
    # print('Cargando el Modelo')
    model = rn.cargarModelo(modelName) 

    #Realizamos la prediccion y se preparan datos si es convolucional
    if arq == 'conv':
        # print('Creando ventanas')
        cqt = sliding_windows(cqt)
    
    # print('Realizando predicción')

    pred = rn.predecir(model, cqt)
    # print('Prediccion: ', pred.shape)
    # print('Piano Roll: ', piano_roll.shape)

    #Matriz de distancias
    # print('Calculando matriz de distancias.')
    distance_matrix = 1 - np.dot(piano_roll.T, pred)
    # print('Matriz calculada')

    #Algoritmo DTW
    # print('Calculando camino...')
    penalty = None
    d_m = copy.deepcopy(distance_matrix)

    #Obtener secuencias temporales
    if penalty is None:
        penalty = d_m.mean()
    q,p,_ = dtw.dtw(d_m, gully=.98, additive_penalty=penalty)
    # print('El camino ha sido calculado')
    #Convertimos a segundos
    p = p * (HOP/FS)
    q = q * (HOP/FS)

    #Información archivo mposTools
    bartimes = mposTools.get_mpos_data(mposFile)

    b = copy.deepcopy(bartimes)
    # Valores aislados de P, relacionar con Q y Q comparar con MPOS
    pUnique, pIndex = np.unique(p, return_index=True)

    # Q asociados a pAislada
    qUnique = []
    for i in pIndex:
        qUnique.append(q[i])

    iniCompasP = []
    retardo = 0.5
    i = 0
    while len(bartimes) > 0 and i < len(qUnique):
        if i == len(qUnique):
            print(i)
        if (qUnique[i] >= bartimes[0]):
            # Tiempo inicio en P
            iniCompasP.append(pUnique[i] - retardo)
            # print("Compas ", len(bartimes), "(", bartimes[0], "):", qUnique[i])
            bartimes.pop(0)

        i += 1

    # for i in range(len(iniCompasP)):
    #     iniCompasP[i] = max(0.0, iniCompasP[i] - 0.6)
    #     # print("Compas ", i + 1, "(", b[i], "):", iniCompasQ[i], " ---> ", iniCompasP[i])
    print(len(iniCompasP))
    iniCompasP.append(iniCompasP[-1]+10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    iniCompasP.append(iniCompasP[-1] + 10)
    return iniCompasP

