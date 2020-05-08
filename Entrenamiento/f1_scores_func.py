"""
Created on Sat Feb 15 23:39:35 2020

@author: Antonio

Medidas de evaluación del entrenamiento
"""


import os
import numpy as np


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0]*A.shape[1], A.shape[2])


def reshape_2Dto3D(A, sec_dim):
    return A.reshape(A.shape[0]/sec_dim, sec_dim, A.shape[1])


def f1_framewise(O, T):
    """
    Fast version of F-measure computation
    """
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2*T-O) == 1).sum(axis=1)
    FP = ((O-T) == 1).sum(axis=1)
    FN = ((T-O) == 1).sum(axis=1)
    eps = np.finfo(np.float).eps

    prec = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)   
    f1_score = 2*prec*recall/(prec+recall)
    return f1_score


def error_tot(O, T):
    """
    Total error computation: see:
    A Discriminative Model for Polyphonic PianoTranscription Graham E. Poliner and Daniel P.W. Ellis
    """
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2*T-O) == 1).sum(axis=1)+0.0
    nb_O, nb_T = O.sum(axis=1), T.sum(axis=1)    
    err_tot = np.sum(np.maximum(nb_O,nb_T)-TP)/nb_T.sum()

    return err_tot        
    
