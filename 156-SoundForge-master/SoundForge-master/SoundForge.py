import sys
import random
import math
import os
import pyaudio
import pygame
from pygame.locals import *
from random import * 
import numpy
from numpy import sqrt, log

from recorder import SwhRecorder #http://www.swharden.com/blog/2013-05-09-realtime-fft-audio-visualization-with-python/
from frequency_estimator import freq_from_autocorr as freq_from_autocorr #https://github.com/endolith/waveform-analyzer

class SoundForge(object):
    def run(self, screen):
        screen_size = (1024,600)
        screen_color = (0, 0 ,0)
        
        #build frequency to note dictionary
        notes_dictionary = self.build_range()

        #sort the keys and turn into a numpy array for logical indexing
        frequencies = numpy.array(sorted(notes_dictionary.keys()))

        top_note = 24 #set it two octaves higher
        bottom_note = 0
        
        #other variables
        screen = pygame.display.set_mode(screen_size)
        screen.fill(screen_color)
        input_note = 1 #the y value on the plot
        old_position = (0,0) #the last position of note played
        show_notes = True #show the note
        signal_level = 0 #volume level
        trys = 1
        line = True
        sound_gate = 15 #zero is loudest possible input level
        target_note = 0 #closest note in the dictionary
        sound_recorder = SwhRecorder() #microphone
        
        while trys <> 0:
            trys += 1

            for n in range(0, screen_size[0], 40):
                sound_recorder.setup()
                raw_data_signal = sound_recorder.getAudio()
                signal_level = round(abs(self.loudness(raw_data_signal)),2) #find the volume level from raw data

                try: 
                    input_note = round(freq_from_autocorr(raw_data_signal,sound_recorder.RATE),2) #find the freq from the audio sample
                except:
                    input_note == 0
                sound_recorder.close()

                if input_note > frequencies[len(notes_dictionary)-1] or input_note < frequencies[0]: #if the frequency is too high, do nothing
                    continue
                if signal_level > sound_gate: #noise gate to prevent ambient noises
                    continue

                target_note = self.closest_value_index(frequencies, round(input_note, 2)) #find the closest note in the note dictionary
                position = ((n), (screen_size[1]-(int(screen_size[1]/(frequencies[top_note]-frequencies[bottom_note]) * (input_note - frequencies[bottom_note])))) ) 

                #user interface
                for event in pygame.event.get():
                    #quit program
                    if event.type ==  QUIT:
                        sound_recorder.close()
                        return
                    elif event.type == KEYDOWN:
                        #increase top_note range
                        if event.key == K_s:    
                            if top_note <= len(notes_dictionary)-7:
                                top_note += 6
                        #decrease top note range no lower than an octave higher than bottom note
                        if event.key == K_x:    
                            if top_note >= 6 and top_note >= bottom_note + 6:
                                top_note -= 6
                        #increase bottom note range no higher than an octave lower than top note
                        if event.key == K_a:    
                            if bottom_note < top_note:
                                bottom_note += 6
                        #decrease bottom note range
                        if event.key == K_z:    
                            if bottom_note >= 6:
                                bottom_note -= 6 

                        #quit program
                        if event.key == K_q:    
                            sound_recorder.close()
                            return

                if n == 0 or n == screen_size[0]:
                    old_position = position
                
                #draw info box on top
                meter = "###################################"
                info_font = pygame.font.Font(None, 18)
                top_info = info_font.render(
                   "Bottom : " + str(notes_dictionary[frequencies[bottom_note]]) + " :  dec(z), inc(a)     "
                   + "Top : " + str(notes_dictionary[frequencies[top_note]]) + " :  dec(x), inc(s)     "
                   #+ "         Show Notes(n):" + str(show_notes) + "         Lines(l):" + str(line)
                   + "         Loudness: " + meter[1:int(20-signal_level)]
                   , 1, (255,255,255))
                pygame.draw.rect(screen, (0,0,0), (0,0,screen_size[0],20))
                pygame.draw.line(screen, (200,200,200),(0,20),(1024,20), 1)
                screen.blit(top_info, (5,5))

                #draw lines
                if input_note < frequencies[len(notes_dictionary)-1]:
                    if old_position < position:
                        random_color = (randint(20,255),randint(20,255),randint(20,255))
                        pygame.draw.line(screen, random_color, old_position, position, 2)
                    old_position = position

                #draw notes name
                font = pygame.font.Font(None, 30)
                text = font.render(str(notes_dictionary[frequencies[target_note]]), 1, (0,255,0))
                screen.blit(text, (position))

                #update the display
                pygame.display.flip()
                pygame.display.update()

            #clear screen at the end of every loop run
            screen.blit(screen, (0, 0))
            screen.fill(screen_color)

    def build_range(self):
        return {  65.41:'C2',   69.30:'C2#',   73.42:'D2',   77.78:'E2b',   82.41:'E2',   87.31:'F2',   92.50:'F2#',   98.00:'G2',  103.80:'G2#',  110.00:'A2',  116.50:'B2b',  123.50:'B2',
                 130.80:'C3',  138.60:'C3#',  146.80:'D3',  155.60:'E3b',  164.80:'E3',  174.60:'F3',  185.00:'F3#',  196.00:'G3',  207.70:'G3#',  220.00:'A3',  233.10:'B3b',  246.90:'B3',
                 261.60:'C4',  277.20:'C4#',  293.70:'D4',  311.10:'E4b',  329.60:'E4',  349.20:'F4',  370.00:'F4#',  392.00:'G4',  415.30:'G4#',  440.00:'A4',  466.20:'B4b',  493.90:'B4',
                 523.30:'C5',  554.40:'C5#',  587.30:'D5',  622.30:'E5b',  659.30:'E5',  698.50:'F5',  740.00:'F5#',  784.00:'G5',  830.60:'G5#',  880.00:'A5',  932.30:'B5b',  987.80:'B5',
                1047.00:'C6', 1109.00:'C6#', 1175.00:'D6', 1245.00:'E6b', 1319.00:'E6', 1397.00:'F6', 1480.00:'F6#', 1568.00:'G6', 1661.00:'G6#', 1760.00:'A6', 1865.00:'B6b', 1976.00:'B6',
                2093.00:'C7', 2217.40:'C7#', 2349.00:'D7', 2489.00:'E7b', 2637.00:'E7', 2793.80:'F7', 2959.90:'F7#', 3135.90:'G7', 3322.40:'G7#', 3520.00:'A7', 3729.00:'B7b', 3951.00:'B7',
                4186.00:'C8'
                }

    #http://www.algebra.com/algebra/homework/logarithm/logarithm.faq.question.251335.html
    def loudness(self, chunk):
        data = numpy.array(chunk, dtype=float) / 2**12
        ms = math.sqrt(numpy.sum(data ** 2.0) / len(data))
        if ms < 10e-8: ms = 10e-8
        return 10.0 * math.log(ms, 10.0)

    def find_nearest(self, array, value):
        index = (numpy.abs(array - value)).argmin()
        return array[index]

    def closest_value_index(self, array, guessValue):
        # Find closest element in the array, value wise
        closestValue = self.find_nearest(array, guessValue) # Find indices of closestValue
        indexArray = numpy.where(array==closestValue) # Numpys 'where' returns a 2D array with the element index as the value
        return indexArray[0][0]

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1024,600))
    SoundForge().run(screen)