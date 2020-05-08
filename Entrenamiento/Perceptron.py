# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:57:45 2020

Entrenamiento de perceptrón 
"""

from datetime import datetime
import numpy as np
import time as t
import shutil
import os

#import CQT
#Librerias Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

#Librerias propias
from f1_scores_func import f1_framewise, error_tot
from manageModels import save_scores_in_fig
import ProbarModelo as pm
import CargarGuardar as cg
import informe 


def training(parametros, X_train, Y_train, X_val, Y_val, name, path):
    """Entrenamiento del perceptron multicapa"""
    ini = t.time()
    #Parametros del perceptron
    inp_dim = parametros['inp_dim']
    hidden_dim = parametros['hidden_dim']
    out_dim = parametros['out_dim']
    batch_size = parametros['batch_size']
    nb_epoch = parametros['nb_epoch']
    patience = parametros['patience']
    
    informe.datosModelo(name, 'DNN', hidden_dim, nb_epoch, batch_size, parametros['piezas'], parametros['tipo'])
    
    #Red neuronal completamente conectada con 3 capas
    model = Sequential()
     
    for i in range(len(hidden_dim)):
        if i == 0:
            model.add(Dense(hidden_dim[0], input_dim = inp_dim, activation='relu'))
        else:
            model.add(Dense(hidden_dim[i], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
    model.add(Dense(out_dim, activation='sigmoid'))
    
    #Info del modelo
    model.summary()
    
    #Compilamos el modelo creado.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Nb epochs: {arg_1}  |  Batch size: {arg_2}'.format(arg_1=nb_epoch, arg_2=batch_size))
    
    #Guardamos el modelo
    cg.GuardarModelo(model, name, path)
    #Nombre archivo de pesos
    keras_weight_file = name + '_weights.h5'
    
    #Variables de entrenamiento
    score_epoch_sampling = 1
    val_scores_list, val_loss_list, train_loss_list = [], [], []
    epc, best_val_score, patience_count, best_error_score = 0, 0, 0, 0
    
    #ENTRENAMIENTO
    while epc < nb_epoch:
        hist = model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, 
                         validation_data=(X_val, Y_val), verbose=1)
        val_loss_list.append(hist.history.get('val_loss'))
        train_loss_list.append(hist.history.get('loss'))
        
        #Validacion y paciencia 
        if not epc % score_epoch_sampling:
            #Selecciona ejemplos del validation y se visualizan errores
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
#    weight_path = './'+keras_weight_file
    if os.path.exists(keras_weight_file):
        print('Archivo pesos movido')
        path = './modelos/Model_'+name
        shutil.move(keras_weight_file, path)
    

    #Informe
        #Informe final.
    fin = t.time()
    tiempoProcesamiento = (fin - ini)/60
    print("Tiempo entrenar modelo: {} minutos".format(tiempoProcesamiento/60))
    informe.actualizarDatosModelo(name, best_val_score, best_error_score, tiempoProcesamiento, epc)

    return model
    
def test(name, model, umbral, tipo):
    """Test sobre el modelo obtenido"""
    
    X_test, Y_test = cg.cargarTest(tipo, name, True, True)
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
    
def procesarModelo(parametros):
    
    #Nombre para organizar cada modelo
    now = datetime.now()
    name = ('{}-{}_{}-{}'.format(now.day,now.month,now.hour,now.minute ))
    nameDir = 'Model_'+name
    
    #En el directorio guardamos el modelo, informe e imagenes.
    path = cg.crearDirectorioModelo(nameDir)
    
    #Control del tiempo con ini y fin
    ini = t.time()
    #Carga de los datasets
    X_train, Y_train, X_val, Y_val = cg.cargarDataset(parametros['tipo'], name,True, parametros['piezas'])
    
    fin = t.time()
    print("Tiempo carga Dataset {} minutos".format((fin-ini)/60))
    #ENTRENAMIENTO
    ini = t.time()
    model = training(parametros, X_train, Y_train, X_val, Y_val, name, path)
    fin = t.time()
    print("Tiempo entrenar modelo: {} minutos".format((fin-ini)/60))   

    #TEST
    umbral = 0.5
    test(name, model, umbral, parametros['tipo'])    

    #Prueba de archivos individuales con modelo entrenado
    #Carga el modelo 
    model = pm.cargarModelo(name)    
    if(model == -1):
        print('El modelo no ha sido cargado correctamente')
    
    #Rutas a los archivos de prueba
    archivos = pm.cargarArchivosPruebas()
    
    pm.realizarPruebas(model, archivos, name, umbral)
    
    
######################### MAIN ###################################


parametros = dict(inp_dim=290, hidden_dim=(5000, 2500, 1000, 500, 250), out_dim=88,
          batch_size=32768, nb_epoch= 1, patience=25, 
          tipo='pruebas', piezas=True)

procesarModelo(parametros)



