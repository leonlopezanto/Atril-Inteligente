# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:29:09 2019

@author: Antonio
"""
import matplotlib.pyplot as plt
import numpy as np


def visualizarDataset(datos, titulo = '', tipo='', guardar = True, mostrar = False):
    
    datos = np.array(datos, dtype=float)              #Para mostrar los targets
    fig = plt.figure(0, figsize=(10,8))
    plt.title(titulo, loc='center')
    plt.imshow(datos, aspect="auto")
    
    if guardar:
        ruta = './imagenes/' + tipo +'/'+ titulo + '.jpg'
        fig.savefig(ruta, bbox_inches = 'tight')
    
    if mostrar:
        plt.show()
        
        
#Visualizamos la perdida del modelo entrenado.
def visualizarDatos(train_loss, valid_loss,accuracy_train, accuracy_val, hidden_dim, name, path, nb_epochs, guardar, his):
    #Gráfica de pérdidas
    plt.ion()
    fig = plt.figure(0, figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    
    #Posicion mas baja de la perdida en la validacion
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r', label='Parada')
    
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    #información de la gráfica:
    title = ('Epochs: {}, Neuronas/Capa, {}, Train Loss: {:0.5f}, Val Loss: {:0.5f}\n'.format(nb_epochs, hidden_dim, np.mean(train_loss), np.mean(valid_loss)))
    plt.title(title, loc='center')
    plt.show()
    if guardar:
        #Guardamos con la fecha y la hora.     
        name1 = name + '_loss'
        p = path + name1
        fig.savefig(p, bbox_inches='tight')
    
    
    #Gráfica de precisiones
    fig = plt.figure(1, figsize=(10,8))
    plt.plot(range(1,len(accuracy_train)+1),accuracy_train, label='Train Acc')
    plt.plot(range(1,len(accuracy_val)+1),accuracy_val,label='Val Acc')
    
    #Posicion mas baja de la perdida en la validacion
    plt.axvline(minposs, linestyle='--', color='r', label='Parada')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    #información de la gráfica:
    title = ('Epochs: {}, Neuronas/Capa, {}, Train Acc: {:0.5f}, Val Acc: {:0.5f}\n'.format(nb_epochs, hidden_dim, np.mean(accuracy_train), np.mean(accuracy_val)))
    plt.title(title, loc='center')
    plt.show()
    if guardar:
        #Guardamos con la fecha y la hora.     
        name2 = name + '_acc'
        p = path + name2
        fig.savefig(p, bbox_inches='tight')
        
        
    
def compararMatrices(pred, target, tipo, name='sinmas', path='', guardar=True):
    print('Target')
    fig1 = plt.figure(3, figsize=(10,8))
    nombre1 = "Target Matrix " + tipo
    plt.title(nombre1)
    plt.imshow(target.astype(float), aspect="auto")
    #fig1.show()
    if guardar:
        #Guardamos con la fecha y la hora.     
        name1 = name + '_matrix_target_'+tipo 
        p = path + name1
        fig1.savefig(p, bbox_inches='tight')
    print('Prediccion')
    fig2 = plt.figure(3, figsize=(10,8))
    nombre2 = "Prediction Matrix " + tipo
    plt.title(nombre2)
    plt.imshow(pred, aspect="auto")
   # fig2.show()
    if guardar:
        #Guardamos con la fecha y la hora.     
        name2 = name + '_matrix_pred_'+tipo 
        p = path + name2
        fig2.savefig(p, bbox_inches='tight')