# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:21:17 2019

@author: Antonio
"""
import Midi as m
def comprobarAcordes(on, off, inst):
    """Comprueba si dos acordes se tocan al mismo tiempo"""
    if(off - on) < inst:
        return True
    else:
        return False

    
def Acordes(onSet, offSet, pitches, hop_length, sr):
    
    instantaneo = hop_length / sr
    acordes = []

    i = 0
    j = 1
    while i < len(onSet):
        ac = []
        primerOn = onSet[i]
        ac.append(pitches[i])
        while j < len(onSet):
            if(comprobarAcordes(primerOn, onSet[j], instantaneo)):
                ac.append(pitches[j])
                j += 1
            else:
                break
        i = j
        j += 1

        acordes.append(ac)

                    
    return acordes
#
#midi = m.Midi('MAPS_ISOL_RE_F_S0_M25_ENSTDkAm.mid')
#onSet, offSet = midi.OnsetOffsetMidi()
#pitches = midi.getPitches()
#midi.piano_roll()
#sr = 44100
#hop = 512
##onSet.append(0.624989)
##onSet.append(0.724898)
##onSet.append(0.7249)
##
##pitches.append(20)
##pitches.append(30)
##pitches.append(40)
#a = []
#
#a = acordes(onSet, offSet, pitches, hop, sr)
#
#print(a)






