# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:17:09 2019

@author: Antonio
"""

#PRUEBAS MIDI

import pretty_midi as pm

import numpy as np
# For plotting

import librosa.display
import matplotlib.pyplot as plt

# For putting audio in the notebook



class Midi:
    def __init__(self, path):
        self.sr = 44100
        self.hop_length = 512
        pm.pretty_midi.MAX_TICK = 1e11
        self.midi_path = path
        self.midi = pm.PrettyMIDI(path)
        #Para evitar fallos al cargar archivos largos.
        
   
    def __len__(self):
        return len(self.midi.instruments[0].notes)
    
    def __getitem__(self):
        return self.midi
    def item(self):
        return self.midi
    
#    def piano_matrix(self):
#        return = self.midi.get_piano_roll(fs=(1/(self.hop_length/self.sr)))
#        
#    
    def piano_roll(self, start_pitch=21, end_pitch=109, fs=20):
        """Muestra el espectrograma del midi"""
#        librosa.display.specshow(self.midi.get_piano_roll(fs=44100/512)[start_pitch:end_pitch],
#                          x_axis='time', y_axis='cqt_note',
#                         fmin=pm.note_number_to_hz(start_pitch))
#        plt.show()
#        print(self.midi.get_piano_roll(fs=44100/512)[start_pitch:end_pitch].shape())
        
        return self.midi.get_piano_roll(fs=44100/512)[start_pitch:end_pitch]
        
    def midi_info(self):
    # Let's look at what's in this MIDI file
        print('There are {} time signature changes'.format(len(self.midi.time_signature_changes)))
        print('There are {} instruments'.format(len(self.midi.instruments)))
        for i in range( len(self.midi.instruments)):
            print('Instrument {} has {} notes'.format(i, len(self.midi.instruments[0].notes)))
        print('Tiempo final: {} / {}'.format(self.midi.get_end_time(), self.midi.estimate_tempo()))
        
    def OnsetOffsetMidi(self):
        onset = []
        offset = []
        
        for i in range(len(self.midi.instruments)):
            for j in range(len(self.midi.instruments[i].notes)):
#                print('Tiempo inicio nota {}: {}'.format(j, self.midi.instruments[i].notes[j].start))
                onset.append(self.midi.instruments[i].notes[j].start)
#                print('Tiempo final nota {}: {}'.format(j, self.midi.instruments[i].notes[j].end))
                offset.append(self.midi.instruments[i].notes[j].end)
        return onset, offset

    def OnsetOffsetTick(self):
        onTick = []
        offTick = []
    
        for i in range(len(self.midi.instruments)):
            for j in range(len(self.midi.instruments[i].notes)):
#                print('Tick inicio nota {}: {}'.format(j,self.midi.time_to_tick(self.midi.instruments[i].notes[j].start)))
                onTick.append(self.midi.time_to_tick(self.midi.instruments[i].notes[j].start))
#                print('Tick fin nota {}: {}'.format(j, self.midi.time_to_tick(self.midi.instruments[i].notes[j].end)))
                offTick.append(self.midi.time_to_tick(self.midi.instruments[i].notes[j].end))
                
        return onTick, offTick
    
    def adjustTimes(self, original_times, new_times):
        self.midi.adjust_times(original_times, new_times)
    
    def getPitches(self):
        pitches = []
        for i in range(len(self.midi.instruments)):
            for j in range(len(self.midi.instruments[i].notes)):
                pitches.append(self.midi.instruments[i].notes[j].pitch)
        return pitches
    
    def getPitchesTarget(self):
        pitches = []
        for i in range(len(self.midi.instruments)):
            for j in range(len(self.midi.instruments[i].notes)):
                pitches.append(self.midi.instruments[i].notes[j].pitch - 21)
        return pitches
        
    def getPitch(self, idx):
        return self.midi.instruments[0].notes[idx].pitch
    
    def ajustar_tiempo(self, original, p, new, q, direct):
        self.midi.adjust_times(original[p], new[q])
#        librosa.output.write_wav(direct+'midiAlineado.mid', self.midi, 44100)
        self.midi.write(direct + 'midiAlign.mid')
#    def createTarget(self):
#        
#        
        
        


#fp = open('MAPS_ISOL_CH0.05_F_ENSTDkAm.txt')
#m = pm.PrettyMIDI('MAPS_ISOL_CH0.05_F_ENSTDkAm.mid')
#x, sr = librosa.load('MAPS_ISOL_CH0.05_F_ENSTDkAm.wav' , sr=44100)
#
#print('E')

#def plot_piano_roll(pm, start_pitch=21, end_pitch=109, sr=44100):
#    # Use librosa's specshow function for displaying the piano roll
#    a = librosa.display.specshow(pm.get_piano_roll(sr)[start_pitch:end_pitch],
#                             hop_length=1, sr=sr, x_axis='time', y_axis='cqt_note',
#                             fmin=pm.note_number_to_hz(start_pitch))
#    return a
#
#a = plot_piano_roll(m)
##plt.figure(figsize=(8, 4))
#plot_piano_roll(pm, 0, 10000)
## Note the blurry section between 1.5s and 2.3s - that's the pitch bending up!
#    
#
## Synthesis frequency
#fs = 16000
#IPython.display.Audio(pm.synthesize(fs=16000), rate=16000)
## Sounds like sine waves... 

#Parsing midi files.
#
#pm = pretty_midi.PrettyMIDI('MAPS_MUS-bk_xmas1_ENSTDkAm.mid')
#plt.figure(figsize=(12,4))
#plot_piano_roll(pm, 0, 120)

#def midi_info(pm):
#    # Let's look at what's in this MIDI file
#    print('There are {} time signature changes'.format(len(pm.time_signature_changes)))
#    print('There are {} instruments'.format(len(pm.instruments)))
#    for i in range( len(pm.instruments)):
#        print('Instrument {} has {} notes'.format(i, len(pm.instruments[0].notes)))
#    
#        
#    
#midi_info(pm)

#def OnsetOffsetMidi(pm):
#    onset = []
#    offset = []
#    
#    for i in range(len(pm.instruments)):
#        for j in range(len(pm.instruments[i].notes)):
#            print('Tiempo inicio nota {}: {}'.format(j, pm.instruments[i].notes[j].start))
#            onset.append(pm.instruments[i].notes[j].start)
#            print('Tiempo final nota {}: {}'.format(j, pm.instruments[i].notes[j].end))
#            offset.append(pm.instruments[i].notes[j].end)
##            print('Tick inicio nota {}: {}'.format(j, pm.time_to_tick(pm.instruments[j].notes[j].start)))
##            print('Tick fin nota {}: {}'.format(j, pm.time_to_tick(pm.instruments[j].notes[j].end)))
#    
#    return onset, offset
#
#def OnsetOffsetTick(pm):
#    onTick = []
#    offTick = []
#    
#    for i in range(len(pm.instruments)):
#        for j in range(len(pm.instruments[i].notes)):
#            print('Tick inicio nota {}: {}'.format(j, pm.time_to_tick(pm.instruments[i].notes[j].start)))
#            onTick.append(pm.instruments[i].notes[j].start)
#            print('Tick fin nota {}: {}'.format(j, pm.time_to_tick(pm.instruments[i].notes[j].end)))
#            offTick.append(pm.instruments[i].notes[j].end)
#    
#
#OnsetOffsetTick(pm)
#
## What's the start time of the 10th note on the 3rd instrument?
#print(pm.instruments[0].notes[0].start)
## What's that in ticks?
#tick = pm.time_to_tick(pm.instruments[0].notes[0].end)
#print(tick)
## Note we can also go in the opposite direction
#print(pm.tick_to_time(int(tick)))



# Plot the tempo changes over time
# Many MIDI files won't have more than one tempo change event,
# but this particular file was transcribed to somewhat closely match the original song.
#times, tempo_changes = pm.get_tempo_changes()
#plt.plot(times, tempo_changes, '.')
#plt.xlabel('Time')
#plt.ylabel('Tempo');
#plt.show()



## Get and downbeat times
#beats = pm.get_beats()
#downbeats = pm.get_downbeats()
## Plot piano roll
#plt.figure(figsize=(12, 4))
#plot_piano_roll(pm)
#ymin, ymax = plt.ylim()
## Plot beats as grey lines, downbeats as white lines
#mir_eval.display.events(beats, base=ymin, height=ymax, color='#AAAAAA')
#mir_eval.display.events(downbeats, base=ymin, height=ymax, color='#FFFFFF', lw=2)
## Only display 20 seconds for clarity
#plt.xlim(0, 420);
#
## Plot a pitch class distribution - sort of a proxy for key
#plt.bar(np.arange(12), pm.get_pitch_class_histogram());
#plt.xticks(np.arange(12), ['C', '', 'D', '', 'E', 'F', '', 'G', '', 'A', '', 'B'])
#plt.xlabel('Note')
#plt.ylabel('Proportion')