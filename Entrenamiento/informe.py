# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:50:04 2019

@author: Antonio
"""
from datetime import datetime
import os.path as path
def datosModelo(name, arq, dim, epochs, batch_size, piezas, tipo):
    """Guarda los datos iniciales del modelo"""
    directorio = './modelos/Model_' + name + '/info' + name + '.txt'
    
    if path.exists(directorio):
        f = open(directorio, 'a')
    else:
        f = open(directorio, 'w')
    f.write('\nArq: ' + arq)
    f.write('\nFecha: '+str(name))
    f.write('\nTipo de entrenamiento: '+str(tipo))
    if(piezas):
        f.write('\nDataset completo')
    else:
        f.write('\nDataset sin piezas musicales')
    f.write('\nCapas Internas: '+str(dim))
    f.write('\nTam Batch: '+str(batch_size))
    f.write('\nEpochs predefinidos: ' + str(epochs))
    
    f.close()
    
def actualizarDatosModelo(name, f1, error_tot, tProcesamiento, num_epochs):
    """Guarda los datos finales del modelo tras el procesamiento"""
    
    directorio = './modelos/Model_' + name + '/info' + name + '.txt'
    if path.exists(directorio):
        f = open(directorio, 'a')
    else:
        f = open(directorio, 'w')
        
    f.write('\n******Datos finales******')
    f.write('\nMejor F1: ' + str(f1))
    f.write('\nError Total: ' + str(error_tot))
    f.write('\nNum Epochs: '+str(num_epochs))
    f.write('\nTiempo Procesamiento: '+str(tProcesamiento))
    
def informeProgreso(name, best_val_score, error_total, train_loss, val_loss):
    
    directorio = './modelos/Model_' + name + '/desarrollo' + name + '.txt'
    if path.exists(directorio):
        f = open(directorio, 'a')
    else:
        f = open(directorio, 'w')
        
        
    now = datetime.now()
    hora = ('{}:{}:{}'.format(now.hour,now.minute, now.second ))
    f.write('\n***** UPDATE '+str(hora)+' *****')
    f.write('\nFecha: '+str(name))
    f.write('\nMejor Score F1: '+str(best_val_score))
    f.write('\nMejor error Total: '+str(error_total))
    f.write('\nMejor perdida train: '+str(train_loss))
    f.write('\nMejor perdida val: '+str(val_loss)+'\n')
    

    f.close()

def informeTest(name, evaluate, numeroAciertos, acc, f1, err, longTarget):
    
    directorio = './modelos/Model_' + name + '/Test' + name + '.txt'
    
    f = open(directorio, 'w')
    f.write('****Test**** ')
    f.write('\nCon metodo evaluate:')
    f.write('\nLoss: '+str(evaluate[0]))
    f.write('\nAccuraccy: '+ str(evaluate[1]))
    f.write('\n\nMetodo manual: ')
    f.write('\nNumero aciertos: ' +  str(numeroAciertos) + '/' + str(longTarget) + '(' + str(acc) + '%)')
    f.write('\nF1 Frame: ' + str(f1))
    f.write('\nError Total: ' + str(err))
    f.close()


    