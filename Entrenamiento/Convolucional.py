# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:39:35 2020

@author: Antonio

Entrenamiento de la red Convolucional
"""


from datetime import datetime
import numpy as np
import time as t

import math

# KERAS
from keras.layers import Dense, BatchNormalization,Flatten, Conv2D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D

#from keras.layers.advanced_activations import loglayer
import gc
import time

#Librerias propias
from f1_scores_func import f1_framewise, error_tot
from manageModels import save_scores_in_fig
import ProbarModelo as pm
import CargarGuardar as cg
import informe 


def prepararDirectorio():
    name = ('{}-{}_{}-{}'.format(now.day,now.month,now.hour,now.minute ))
    nameDir = 'Model_'+name
    
    #En el directorio guardamos el modelo, informe e imagenes.
    path = cg.crearDirectorioModelo(nameDir)
    
    return name, path



def sliding_windows(X, sw = 7):
    
    print('Dividiendo matriz en ventanas consecutivas de {}'.format(sw))
#    X = X.T
    Xwin = list()
    for i in range(X.shape[0]-(sw-1)):
        Xwin.append(X[i:i+sw, :])
        
    print('Done')
        
    return np.array(Xwin, dtype=np.float16)

def TrainGenerator(X_train, Y_train, sw, batch):
    
    print('Generando..')
    padding = int(sw/2)
    X_train = np.vstack((np.zeros((padding, X_train.shape[1]), dtype=np.float16), X_train, np.zeros((padding, X_train.shape[1]), dtype=np.float16 )))
    #range(m, n, p)-> lista que empieza en m, acaba antes de n, y aumenta los valores de p en p.
#    for i in range(0, long, batch): #Devuelve una lista de números entre 0 y longitud, 
    while True:
        Xwin = list()
        Ywin = list()
        for i in range(X_train.shape[0]-(sw-1)):
            Xwin.append(X_train[i:i+sw, :])
            Ywin.append(Y_train[i])
            if len(Xwin) == batch:
                Xwin = np.array(Xwin, dtype=np.float16)
                Xwin = np.reshape(Xwin, (Xwin.shape[0], 1, Xwin.shape[1],  Xwin.shape[2]))
                yield Xwin, np.array(Ywin, dtype=np.float16)
#                print('enviado')
                Xwin = list()
                Ywin = list()
    
                

def training(parametros, X_train, Y_train, X_val, Y_val, name, path):
    
    #Parametros 
    inp_dim = parametros['inp_dim']
    hidden_dim = parametros['hidden_dim']
    out_dim = parametros['out_dim']
    batch_size = parametros['batch_size']
    nb_epoch = parametros['nb_epoch']
    patience = parametros['patience']
    sw = parametros['w_s']
    
    """Entrenamiento del perceptron multicapa"""
    ini = t.time()
    
    #Creamos ventanas
    padding = int(sw/2)
    X_val= np.vstack((np.zeros((padding, X_val.shape[1])), X_val, np.zeros((padding, X_val.shape[1]))))
    X_val = sliding_windows(X_val, sw)
    X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1],  X_val.shape[2]))
    
    #INFORME A CONVOLUCIONAL
    informe.datosModelo(name, 'convolucional', hidden_dim, nb_epoch, batch_size, parametros['piezas'], parametros['tipo'])
    
    #Red neuronal convolucional
    #Red neuronal SOTA
    model = Sequential()
    model.add(Conv2D(filters=50, kernel_size=(5, 25), input_shape=(1, sw, X_train.shape[1]), data_format='channels_first', activation='tanh', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3,5), activation='tanh', padding='same'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(BatchNormalization())

    
    model.add(Flatten())
    for i in range(len(hidden_dim)):
        model.add(Dense(hidden_dim[i], activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(out_dim, activation='sigmoid'))
    
    #Info del modelo
    model.summary()
    
    #Compilamos el modelo creado.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Nb epochs: {arg_1}  |  Batch size: {arg_2}'.format(arg_1=nb_epoch, arg_2=batch_size))
    
    #Guardamos el modelo
    cg.GuardarModelo(model, name, path)
    #Nombre archivo de pesos
    keras_weight_file = path + name + '_weights.h5'
    
    #Variables de entrenamiento
    score_epoch_sampling = 1
    val_scores_list, val_loss_list, train_loss_list = [], [], []
    epc, best_val_score, patience_count, best_error_score = 0, 0, 0, 0
    
    #ENTRENAMIENTO
    while epc < nb_epoch:
        gc.collect()
        hist = model.fit_generator(TrainGenerator(X_train, Y_train, sw, batch_size), validation_data=(X_val, Y_val), verbose=1, shuffle=False, steps_per_epoch=math.ceil(X_train.shape[0]/batch_size),)
        gc.collect()
        val_loss_list.append(hist.history.get('val_loss'))
        train_loss_list.append(hist.history.get('loss'))
        
        #Validacion y paciencia 
        if not epc % score_epoch_sampling:
            #Selecciona ejemplos del validation y se visualizan errores
            gc.collect()
            preds = model.predict(X_val, verbose=0)
            
            val_scores = dict()
            val_scores['f1_framewise'] = np.mean(f1_framewise(1.0 * (np.squeeze(preds) > 0.5), Y_val))
            val_scores['error_tot'] = error_tot(1.0 * (np.squeeze(preds) > 0.5), Y_val)
            val_scores_list.append(val_scores)
            print("* * validation scores: ", val_scores)
            
            #Guarda la figura con pérdidas y scores
            save_scores_in_fig(val_loss_list, train_loss_list, val_scores_list, name)
            
            #Si se mejora el F1
            if val_scores['f1_framewise'] > best_val_score:
                best_train_loss = train_loss_list[-1]
                best_val_loss = val_loss_list[-1]
                best_error_score = val_scores['error_tot']
                best_val_score= val_scores['f1_framewise']
                patience_count = 0
                #Guarda los pesos actualizados
                model.save_weights(keras_weight_file, overwrite=True)
                #Actualizacion del informe
                informe.informeProgreso(name, best_val_score, best_error_score, best_train_loss, best_val_loss)
            else:
                patience_count += score_epoch_sampling
                if patience_count > patience:
                    print("Entrenamiento interrumpido, f1 no ha mejorado.\nMejor: ", best_val_score)
                    model.load_weights(keras_weight_file)
                    break
        
        epc += 1
        print('********\nNumero de epochs: ', epc,'\nPaciencia: ', patience_count, 
              '\nMejor valor F1: ', best_val_score, '\n********')
    
    model.load_weights(keras_weight_file)
    print('Modelo guardado: ', keras_weight_file)
    
    #Mueve el archivo de los pesos a la carpeta del modelo.
#    if os.path.exists(keras_weight_file):
#        print('Archivo pesos movido')
#        path = './modelos/Model_'+name
#        shutil.move(keras_weight_file, path)
    
    #Informe final.
    fin = t.time()
    tiempoProcesamiento = (fin - ini)/60
    print("Tiempo entrenar modelo: {} minutos".format(tiempoProcesamiento/60))
    informe.actualizarDatosModelo(name, best_val_score, best_error_score, tiempoProcesamiento, epc)

    return model

    
def test(name, model, umbral, ws, tipo):
    
    #Obtenemos los datos
    
    X_test, Y_test = cg.cargarTest(tipo,name, True, True)
    
    """Test sobre el modelo obtenido"""
    padding = int(ws/2)
    X_test = np.vstack((np.zeros((padding, X_test.shape[1])), X_test, np.zeros((padding, X_test.shape[1]))))
    X_test = sliding_windows(X_test, ws)
    
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1],  X_test.shape[2]))
    #Evaluacion del modelo
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('--EVALUACION DEL TEST AUTOMATICO--')
    print('Perdida: ', score[0])
    print('Precision: ', score[1]*100)
    
    print('--EVALUACION DEL TEST MANUAL--')
    prediccion = model.predict(X_test, verbose=2)
    pred = 1.0 * (np.squeeze(prediccion) > umbral )
    
    #Prediccion Manual
    numeroAciertos = sum([np.argmax(Y_test[i])==np.argmax(pred[i]) for i in range(len(Y_test))])
    acc = numeroAciertos/len(Y_test)*100
    print('Accuracy de la red: {}/{} {}% '.format(numeroAciertos, len(Y_test), acc))
    

    f1 = np.mean(f1_framewise(1.0 * (np.squeeze(prediccion) > 0.5), Y_test))
    err = error_tot(1.0 * (np.squeeze(prediccion) > 0.5), Y_test)

    #Informe
    informe.informeTest(name, score, numeroAciertos, acc, f1, err, len(Y_test))

    
def procesarModelo(parametros, X_train, Y_train, X_val, Y_val, name, path):
    
#    #ENTRENAMIENTO
    gc.collect()
    ini = t.time()
    model = training(parametros, X_train, Y_train, X_val, Y_val, name, path)
    fin = t.time()
    print("Tiempo entrenar modelo: {} minutos".format((fin-ini)/60))   
    
    #Prueba de archivos individuales con modelo entrenado
    model = pm.cargarModelo(name)
    
    #TEST
    umbral = 0.5
    test(name, model, umbral, parametros['w_s'], parametros['tipo'])    
    
    if(model == -1):
        print('El modelo no ha sido cargado correctamente')
    
    #Pruebas post entrenamiento
    archivos = pm.cargarArchivosPruebas()
    pm.realizarPruebas(model, archivos, name, umbral, parametros['w_s'])


######################### MAIN ###################################

gc.collect()
#tipo "pruebas" para comprobar funcionamiento, "Completo" para entrenamiento
parametros = dict(inp_dim=290, hidden_dim=(1000, 200), out_dim=88,
                  batch_size=512, nb_epoch= 1, patience=6, 
                  tipo='pruebas', piezas=True, w_s=21)

#Nombre para organizar cada modelo
now = datetime.now()
name, path = prepararDirectorio()

#Control del tiempo con ini y fin
ini = t.time()

#Carga de los datasets
X_train, Y_train, X_val, Y_val= cg.cargarDataset(parametros['tipo'], name,True, parametros['piezas'])
fin = t.time()

procesarModelo(parametros, X_train, Y_train, X_val, Y_val, name, path ) 