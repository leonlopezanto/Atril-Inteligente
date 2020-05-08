# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:08:00 2019

@author: Antonio López León
@Description: Obtiene el valor CQT de todos los archivos del Dataset.

"""
import librosa
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

#Librerías propias
import Midi as m
import Acordes as ac
import Visualizaciones as v
import CargarGuardar as cg
import organizarDirectorios as od

#constantes Parámetros CQT
sr = 44100
hop_length = 512
n_bins = 290
num_notas = 88

def extraer_CQT(audio_data, sr, hop_length):
    """Función para extraer el valor CQT del audio"""
    C = librosa.cqt(audio_data, sr=sr, fmin=librosa.midi_to_hz(21),
                    n_bins=n_bins, bins_per_octave=36, hop_length=hop_length)
    
    amplitudNota = np.abs(C)
    
    return amplitudNota
    
def guardarImagenes(i, fileName, X, Y, tipo):
    tituloCQT = str(i)+"_"+fileName+"_cqt"
    v.visualizarDataset(X, tituloCQT, tipo=tipo, guardar=True, mostrar=False)
    tituloTarget = str(i)+"_"+fileName+"_target"
    v.visualizarDataset(Y, tituloTarget, tipo=tipo, guardar=True, mostrar=False)
        
def checkFin(a , i):
    if a.shape[0] != n_bins:
        print("Fallo en posicion ", i)
        sys.exit()
    
def CQTIsolate(path, fileName, midiFile):
    """Función para extraer el valor CQT de los archivos ISO"""
    
    
    #Recorremos cada archivo
    for i in range(len(path)):
        print("Archivo audio nº: {}/{}\nName:{}".format(i+1, len(path), fileName[i]))
        
        X = list()
        Y = list()
        
        #Cargar audio
        x, _ = librosa.load(path[i], sr=sr)
        
        #Cargar midi
        midi = m.Midi(midiFile[i])
        
        
        #Obtener datos midi
        onSet, offSet = midi.OnsetOffsetMidi()
        pitches = midi.getPitchesTarget()
        
        #Comprobar si las notas son acordes
        acordes = ac.Acordes(onSet, offSet, pitches, hop_length, sr)
        
        #Procesado de audio y creación del target para cada archivo
        for nota in np.arange(len(midi)):
            #Se recorta y procesa por notas individuales
            xnota = x[int(round(onSet[nota]*sr)) : int(round(offSet[nota]*sr))]
            #Procesado cqt
            amplitudNota = extraer_CQT(xnota, sr, hop_length)
            checkFin(amplitudNota , i)
            #Target
            target = np.zeros((num_notas, len(amplitudNota[0])), dtype = np.ndarray)
            for j in range(len(acordes[nota])):
                target[acordes[nota], :] = 1
                
            X.append(amplitudNota)
            Y.append(target)
        
        #Almacena X e Y en memoria
        nombreArchivo = fileName[i] + "_cqt"
    
        X = np.hstack(X)
        print("Shape data: {}".format(X.shape))
        cg.salvarDatos(X, nombreArchivo, fileName[i], True)
        nombreArchivo = fileName[i] + "_target"
        Y = np.hstack(Y)
        print("Shape target: {}".format(Y.shape))
        cg.salvarDatos(Y, nombreArchivo, fileName[i], False)
        
        #Se guardan datos para visualizar cada 50 archivos.
        if(i % 50 == 0):
            guardarImagenes(i, fileName[i], X, Y, 'isolate')

def CQTChord(path, fileName, midiFile):
    """Funcion para extraer el valor CQT de los archivos Chords"""

    #Recorremos cada archivo
    for i in range(len(path)):
        print("Archivo audio nº: {}/{}\nName:{}".format(i+1, len(path), fileName[i]))
        
        X = list()
        Y = list()
        
        #Cargar audio
        x, _ = librosa.load(path[i], sr=sr)
        #Cargar midi
        midi = m.Midi(midiFile[i])
        #Obtener datos midi
        onSet, offSet = midi.OnsetOffsetMidi()
        pitches = midi.getPitchesTarget()
        #Comprobar y agrupar en acordes
        acordes = ac.Acordes(onSet, offSet, pitches, hop_length, sr)
        
        #Para acordes cojemos el offset más largo y el onset más corto.
        xnota = x[int(round(min(onSet)*sr)) : int(round(max(offSet)*sr))]
        
        amplitudNota = extraer_CQT(xnota, sr, hop_length)   
        checkFin(amplitudNota , i)
        #Creamos el target:
        target = np.zeros((num_notas, len(amplitudNota[0])), dtype = np.ndarray)
        for j in range(len(acordes)):
            target[acordes[j],:] = 1
        
        X.append(amplitudNota)
        Y.append(target)
                   
        #Almacenamos X e Y 
        nombreArchivo = fileName[i] + "_cqt"
        X = amplitudNota
        print("Shape data: {}".format(X.shape))
        cg.salvarDatos(X, nombreArchivo, fileName[i], True)
        nombreArchivo = fileName[i] + "_target"
        Y = target
        print("Shape target: {}".format(Y.shape))
        cg.salvarDatos(Y, nombreArchivo, fileName[i], False)
        #Guardamos imagenes del target y de CQT
        if(i % 50 == 0):
            guardarImagenes(i, fileName[i], X, Y, 'chords')
    
def CQTPieces(path, fileName, midiFile):
    """Funcion para extraer el valor CQT de los archivos MUS"""

    for i in range(len(path)):
        print("Archivo audio nº: {}/{}\nName:{}".format(i+1, len(path), fileName[i]))
           
        X = list()
        Y = list()
        x, _ = librosa.load(path[i], sr=sr)

        midi = m.Midi(midiFile[i])
        onSet, offSet = midi.OnsetOffsetMidi()
        pitches = midi.getPitchesTarget()
        
        amplitudPieza = extraer_CQT(x, sr, hop_length)
        checkFin(amplitudPieza , i)
        #Creamos el target
        longPieza = x.shape[-1]
        target = np.zeros((num_notas, math.ceil(longPieza/hop_length)))
        
        for nota in range(len(pitches)):
            tini = int(round((onSet[nota]*sr)/hop_length))
            tfin = int(round((offSet[nota]*sr)/hop_length))
            target[pitches[nota], tini:tfin] = 1
        
        #Almacenamos
        nombreArchivo = fileName[i] + "_cqt"
        X = amplitudPieza
        print("Shape data: {}".format(X.shape))
        cg.salvarDatos(X, nombreArchivo, fileName[i], True)
        nombreArchivo = fileName[i] + "_target"
        Y = target
        print("Shape target: {}".format(Y.shape))
        cg.salvarDatos(Y, nombreArchivo, fileName[i], False)
                
        #Guardamos imagenes del target y de CQT
        if(i % 10 == 0):
            guardarImagenes(i, fileName[i], X, Y, 'mus')
    
        
################################### MAIN ###########################################

##Notas aisladas           
tipo = 'ISO'
path, fileName, midiFile = cg.loadFiles(tipo)

if tipo == 'ISO':
    CQTIsolate(path, fileName, midiFile)

###Acordes
tipo = 'CH'
path, fileName, midiFile = cg.loadFiles(tipo) 

if tipo == 'CH':
    CQTChord(path,fileName,midiFile)

#Piezas musicales completas
tipo = 'MUS'
path, fileName, midiFile = cg.loadFiles(tipo) 
if tipo == 'MUS':
    CQTPieces(path, fileName, midiFile)


#Organizar los directorios para poder utilizarlos.
od.organizarDirectorios()
od.crearPruebas()














