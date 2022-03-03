
## Mocking Bot - Task 2 : Instrument Classification

#  Instructions
#  ------------
#
#  This file contains Main function and Instrument_identify function. Main Function helps you to check your output
#  for practice audio files provided. Do not make any changes in the Main Function.
#  You have to complete only the Instrument_identify function. You can add helper functions but make sure
#  that these functions are called from Instrument_identify function. The final output should be returned
#  from the Instrument_identify function.
#
#  Note: While evaluation we will use only the onset_detect function. Hence the format of input, output
#  or returned arguments should be as per the given format.
#  
#  Recommended Python version is 2.7.
#  The submitted Python file must be 2.7 compatible as the evaluation will be done on Python 2.7.
#  
#  Warning: The error due to compatibility will not be entertained.
#  -------------


## Library initialisation

# Import Modules
# DO NOT import any library/module
# related to Audio Processing here
import numpy as np
import math
import wave
import os
import librosa
from sklearn import svm
import pickle
import struct
from scipy import signal
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from essentia.standard import *
import essentia as es
from f_Suprt import*
from sklearn import preprocessing

# Teams can add helper functions
# Add all helper functions here

############################### Your Code Here #############################################
def onset_detect(sound):
	
	sound = np.divide(sound,float(2**15))
	f, t, Zxx = signal.stft(sound,44100,window='hann',nperseg=441,noverlap=0)
	#plt.pcolormesh(t, f, np.abs(Zxx))
	'''plt.title('STFT Magnitude')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')'''
	#plt.show()
	#print np.size(f),np.size(t),np.shape(Zxx),Zxx[:,[0]]
		
	mag=np.zeros(np.size(t))							   
	for i in range(np.size(t)):
		absX = abs(Zxx[:,[i]])                                  
		absX[absX<np.finfo(float).eps] = np.finfo(float).eps    
		magX = 20 * np.log10(absX)
		te=np.amax(magX)
		mag[i]=te
	#print mag
	#plt.plot(mag)
	#plt.show()
	smag=np.zeros(np.size(t))
	smag1=np.zeros(np.size(t))
	for i in range(np.size(mag)-1):
		smag[i]=(mag[i+1]-mag[i])/0.01
		smag1[i]=(mag[i+1]-mag[i])/0.01
		if smag1[i]<0:
			smag1[i]=0
	#plt.plot(smag)
	#plt.show()
	'''plt.plot(smag1)
	plt.show()'''
	fsmag=np.zeros(np.size(t))
	#fsmag1=np.zeros(np.size(t))
	'''plt.plot(smag)
	plt.show()'''
	'''k=signal.find_peaks(smag)
	print k'''
	win=np.hanning(20)
	#plt.plot(win)
	#plt.show()
	#j=np.size(smag)//20
	'''for i in range(j):
		ll=i*20
		hl=(20*i+20)
		fsmag=smag[ll:hl]*win
		plt.plot(fsmag)
		plt.show()'''
	fsmag=np.convolve(smag,win,mode='same')
	#fsmag1=np.convolve(smag1,win,mode='same')
	th=np.amax(fsmag)
	#print th-4500
	#plt.plot(fsmag)
	#plt.show()
	'''plt.plot(fsmag1)
	plt.show()'''
	#print np.argsort(fsmag)[:101]
	for i in range(np.size(fsmag)):
		if fsmag[i]<0:
			fsmag[i]=0
	#sa=np.argsort(fsmag)
	#print np.amax(fsmag)
	#k=np.size(fsmag)
	#print sa[(k-100):]	

	#plt.plot(fsmag)
	#plt.show()		
	os=np.argwhere(fsmag>(th-4500))
	#print os
	flsmag=np.zeros(np.size(os))
	for i in range(np.size(os)):
		flsmag[i]=mag[os[i][0]]
	j=0
	index=np.zeros(100,dtype=int)	
	for i in range(np.size(os)-1):
		if(os[i+1][0]-os[i][0]>30):
			index[j]=i
			j=j+1
	index=np.trim_zeros(index)		
	flsmag=np.around(flsmag,decimals=0)
	flsmag=np.floor(flsmag)
	index=np.insert(index,0,0)
	#print flsmag,
	#print index	
	

	#plt.plot(flsmag)
	#plt.show()
	loc=np.zeros(np.size(index),dtype=int)
	j=0
	for i in range(np.size(index)):
		if i<np.size(index)-1:
			if i==0:
				#plt.plot(flsmag[index[i]:index[i+1]+1])
				#plt.show()
				temp=np.amin(flsmag[index[i]:index[i+1]+1])
				#print i,temp
				temp_indx=np.argwhere(flsmag==temp)
				#print np.argwhere(mag==temp_indx)
				#print temp_indx
				if np.size(temp_indx)>1:
					for k in range(np.size(temp_indx)):
						if temp_indx[k][0]>=index[i] and temp_indx[k][0]<=index[i+1]:
							loc[j]=temp_indx[k][0]
							#print loc[j]

				else:
					loc[j]=temp_indx[0][0]
					#print loc[j]
			else:
				#plt.plot(flsmag[index[i]+1:index[i+1]+1])
				#plt.show()
				temp=np.amin(flsmag[index[i]+1:index[i+1]+1])
				#print i,temp
				temp_indx=np.argwhere(flsmag==temp)
				#print temp_indx
				if np.size(temp_indx)>1:
					for k in range(np.size(temp_indx)):
						if temp_indx[k][0]>=index[i]+1 and temp_indx[k][0]<=index[i+1]:
							loc[j]=temp_indx[k][0]
							#print loc[j]

				else:
					loc[j]=temp_indx[0][0]
					#print loc[j]

			
				

		else:
			#plt.plot(flsmag[index[i]:])
			#plt.show()
			temp=np.amin(flsmag[index[i]+1:])
			temp_indx=np.argwhere(flsmag==temp)
			if np.size(temp_indx)>1:
				for k in range(np.size(temp_indx)):
					if temp_indx[k][0]>index[i]:
						loc[j]=temp_indx[k][0]
						#print loc[j]

			else:
				loc[j]=temp_indx[0][0]
				#print loc[j]

		j=j+1
	#print loc

	o=[]
	if loc[0]!=0:
		o.append(0.00)
	for i in range(np.size(loc)):
		#print loc[i],os[loc[i]][0]
		o.append(round(int(os[loc[i]][0])*0.01,2))
	#print o	

	'''plt.plot(flsmag)
	plt.show()	'''		


	'''o=[]
	#o[0]=1
	#j=1
	o.append(0.00)
	for i in range(np.size(os)-1):
		if(flsmag[i]<=-100 and flsmag[i+1]>flsmag[i]):
			o.append(round(int(os[i][0])*0.01,2))'''
			
	#Onsets=o
	return o

def Instrument_identify(audio_file):
	
	#   Instructions
	#   ------------
	#   Input 	:	audio_file -- a single test audio_file as input argument
	#   Output	:	1. Instruments -- List of string corresponding to the Instrument
	#			2. Detected_Notes -- List of string corresponding to the Detected Notes
		#                       3. Onsets -- List of Float numbers corresponding
	#			        to the Note Onsets (up to Two decimal places)
	#   Example	:	For Audio_1.wav file,
	# 				Instruments = ["Piano","Violin","Piano","Trumpet"]
	#				Detected_Notes = ["C3","B5","A6","C5"]
	#                               Onsets = [0.00, 0.99, 1.32, 2.04]
		
	# Add your code here
	Instruments = []
	Detected_Notes = []
	Onsets = []
	notes=[]
	inst=[]
	#instrument_clasifier_pkl= open("Instrument_classifier.pkl", 'rb')
	#classifier= pickle.load(instrument_clasifier_pkl)

	models = []
	with open('Instrument_classifier_mlp_1_3.pkl', "rb") as f:
		while True:
			try:
				models.append(pickle.load(f))
				#print 1
			except EOFError:
				break
	'''classifier1=models[0]
	classifier2=models[1]
	classifier3=models[2]
	classifier4=models[3]
	classifier5=models[4]'''
	classifier6=models[5]
	'''classifier7=models[6]
	classifier8=models[7]
	classifier9=models[8]'''

	i=['Violin','Flute','Piano','Trumpet']
	le = preprocessing.LabelEncoder()
	y = le.fit_transform(i)
	#print y
				
	
	#print "Loaded Decision tree model :: ", mlp
	file_length = audio_file.getnframes()
	#print file_length
	samplerate=44100
	sound = np.zeros(file_length)
	
	for i in range(file_length):
		data = audio_file.readframes(1)
		data = struct.unpack("<h",data)
		sound[i]= int(data[0])
		
	sound = np.divide(sound,float(2**15))
	#print sound
	#energy=np.square(sound)
	#os=[]
	#os=librosa.onset.onset_detect(y=sound,sr=44100,hop_length=256,backtrack=True,energy=energy,units='time')
	#print os
	Onsets=onset_detect(sound)



	#print Onsets
	sound_slice = np.zeros(file_length)
	for i in range(np.size(Onsets)):
		if i==np.size(Onsets)-1:

			sound_slice=sound[int(Onsets[i]*samplerate):]
			#sound_slice=sound[0:43218]
			#sound_slice=sound[43218:58653]
			#sound_slice=sound[58653:89964]
			#sound_slice=sound[89964:]
			X1,X2,X3=F_find(sound_slice,44100)
			'''f = open("dataset101.csv", 'a') 
			X1.to_csv(f, header = False,index=False)
			f.close()
			f = open("dataset102.csv", 'a') 
			X2.to_csv(f, header = False,index=False)
			f.close()'''
			f = open("dataset103.csv", 'a') 
			X3.to_csv(f, header = False,index=False)
			f.close()
			
			#pqr=pqr.tolist()
			#print pqr[0]
			#inst.append(pqr[0])
			'''r = librosa.autocorrelate(y=sound_slice, max_size=5000)
			midi_hi = 120.0
			midi_lo = 12.0
			f_hi = librosa.midi_to_hz(midi_hi)
			f_lo = librosa.midi_to_hz(midi_lo)
			t_lo = samplerate/f_hi
			t_hi = samplerate/f_lo
			r[:int(t_lo)] = 0
			r[int(t_hi):] = 0
			t_max = r.argmax()
			freq= float(samplerate)/t_max
			note=librosa.core.hz_to_note(freq)
			notes.append(note)'''
			spec=Spectrum()
			w=Windowing(type='hann')
			sound_slice=es.array(sound_slice)
			if np.size(sound_slice)%2!=0:
				sound_slice=sound_slice[:-1]
				#print np.size(sound_slice)
			s=spec(w(sound_slice))
			#print s
			p=PitchYinFFT(interpolate=False)
			#i=interpolate()
			f,c=p(s)
			#print f,librosa.core.hz_to_note(f)
			f=librosa.core.hz_to_note(f)
			notes.append(f)




		else:
			#print i
			sound_slice=sound[int(Onsets[i]*samplerate):int(Onsets[i+1]*samplerate)]
			#sound_slice=sound[0:43218]
			#sound_slice=sound[43218:58653]
			#sound_slice=sound[58653:89964]
			#sound_slice=sound[89964:]
			#print int(Onsets[i]*samplerate),int(Onsets[i+1]*samplerate)
			#sound_slice=es.array(sound_slice)
			X1,X2,X3=F_find(sound_slice,44100)
			'''f = open("dataset101.csv", 'a') 
			X1.to_csv(f, header = False,index=False)
			f.close()
			f = open("dataset102.csv", 'a') 
			X2.to_csv(f, header = False,index=False)
			f.close()'''
			f = open("dataset103.csv", 'a') 
			X3.to_csv(f, header = False,index=False)
			f.close()



			#X=X.iloc[:,:].values[0]
			#print X
			'''scaler = StandardScaler()
			print np.reshape(X,(-1,1))
			X=scaler.fit_transform(X)'''
			#X=X.format(array)
			'''X1=X1.values
			X2=X2.values
			X3=X3.values
			
			pqr1=classifier1.predict(X1)
			pqr2=classifier2.predict(X2)
			pqr3=classifier3.predict(X3)
			#le = preprocessing.LabelEncoder()
			#pqr=le.inverse_transform(pqr)
			print pqr1,pqr2,pqr3'''
			#pqr=pqr.tolist()
			#print pqr[0]
			#inst.append(pqr[0])
			
			'''X=df.iloc[:,:]
			predict_instrument=mlp.predict(X)
			print predict_instrument'''
			'''r = librosa.autocorrelate(y=sound_slice, max_size=5000)
			midi_hi = 120.0
			midi_lo = 12.0
			f_hi = librosa.midi_to_hz(midi_hi)
			f_lo = librosa.midi_to_hz(midi_lo)
			t_lo = samplerate/f_hi
			t_hi = samplerate/f_lo
			r[:int(t_lo)] = 0
			r[int(t_hi):] = 0
			t_max = r.argmax()
			freq= float(samplerate)/t_max
			note=librosa.core.hz_to_note(freq)'''
			#print note
			#print sound_slice
			spec=Spectrum()
			w=Windowing(type='hann')
			sound_slice=es.array(sound_slice)
			if np.size(sound_slice)%2!=0:
				sound_slice=sound_slice[:-1]
				#print np.size(sound_slice)
			s=spec(w(sound_slice))
			#print s
			p=PitchYinFFT(interpolate=False)
			#i=interpolate()
			f,c=p(s)
			#print f,librosa.core.hz_to_note(f)
			f=librosa.core.hz_to_note(f)
			notes.append(f)




			#notes.append(note)
	'''colmn_nms1=['LogAttackTime','HD','FluxD','spectral_bandwidthM','mfcc_1D','mfcc_3D','RMSM','spectral_bandwidthD',
		'mfcc_4M','mfcc_11D','ZCRD','spectral_centroidD','mfcc_8D','mfcc_6D','mfcc_7D','mfcc_4D','spectral_centroidM','mfcc_10M','mfcc_10D']
	colmn_nms2=['spectral_centroidM','mfcc_2M','HC','ZCRM','mfcc_3M','HD','ZCRD','HS','mfcc_4M','mfcc_1M','mfcc_10M','FluxM','FluxD',
			'mfcc_5M','mfcc_7M','spectral_bandwidthM','spectral_centroidD','mfcc_6M']'''
	colmn_nms3=['LogAttackTime','RMSM','mfcc_1D','mfcc_3D','mfcc_4D','mfcc_6D','mfcc_7D','mfcc_8D','mfcc_10D','mfcc_11D',
			'spectral_centroidM','spectral_centroidD','mfcc_1M','mfcc_2M','mfcc_3M','mfcc_4M','mfcc_5M','mfcc_6M','mfcc_7M','mfcc_10M',
			'ZCRM','ZCRD','HC','HD','HS','FluxM','FluxD','spectral_bandwidthM','spectral_bandwidthD']
		

	#instrdata_1 = pd.read_csv("dataset101.csv", names=colmn_nms1)
	#instrdata_2 = pd.read_csv("dataset102.csv", names=colmn_nms2)
	instrdata_3 = pd.read_csv("dataset103.csv", names=colmn_nms3)
	'''instrdata_1.head()		
	X1 = instrdata_1.iloc[:,:9].values
	#print X1
	#print y1
	instrdata_2.head()		
	X2 = instrdata_2.iloc[:,:9].values
	#print X2	
	#print y2'''
	instrdata_3.head()		
	X3 = instrdata_3.values
	#print X3
	scaler = StandardScaler()
	#print np.reshape(X,(-1,1))
	#X1=scaler.fit_transform(X1)
	#X2=scaler.fit_transform(X2)
	X3=scaler.fit_transform(X3)
	#print X2
	'''pqr1=classifier1.predict(X1)
	pqr2=classifier2.predict(X2)
	pqr3=classifier3.predict(X3)
	pqr4=classifier4.predict(X1)
	pqr5=classifier5.predict(X2)'''
	pqr6=classifier6.predict(X3)
	'''pqr7=classifier7.predict(X1)
	pqr8=classifier8.predict(X2)
	pqr9=classifier9.predict(X3)'''

	pqr6=le.inverse_transform(pqr6)
	#print pqr1,pqr2,pqr3
	#print pqr6
	#print pqr7,pqr8,pqr8
	#os.remove("dataset101.csv")
	#os.remove("dataset102.csv")
	#os.remove("dataset103.csv")

	
	Detected_Notes=notes
	Instruments=pqr6.tolist()
	
	return Instruments, Detected_Notes, Onsets


############################### Main Function #############################################

if __name__ == "__main__":

	#   Instructions
	#   ------------
	#   Do not edit this function.

	# code for checking output for single audio file
	path = os.getcwd()
	
	file_name = path + "/Task_2_Audio_files/Audio.wav"
	audio_file = wave.open(file_name)
	
	Instruments, Detected_Notes, Onsets = Instrument_identify(audio_file)

	print("\n\tInstruments = "  + str(Instruments))
	print("\n\tDetected Notes = " + str(Detected_Notes))
	print("\n\tOnsets = " + str(Onsets))
	os.remove("dataset103.csv")

	# code for checking output for all audio files
	
	'''x = raw_input("\n\tWant to check output for all Audio Files - Y/N: ")
		
	if x == 'Y':

		Instruments_list = []
		Detected_Notes_list = []
				Onsets_list = []
				
		file_count = len(os.listdir(path + "/Task_2_Audio_files"))

		for file_number in range(1, file_count):

			file_name = path + "/Task_2_Audio_files/Audio_"+str(file_number)+".wav"
			audio_file = wave.open(file_name)

			Instruments, Detected_Notes,Onsets = Instrument_identify(audio_file)
			
			Instruments_list.append(Instruments)
			Detected_Notes_list.append(Detected_Notes)
						Onsets_list.append(Onsets)
		print("\n\tInstruments = " + str(Instruments_list))
		print("\n\tDetected Notes = " + str(Detected_Notes_list))
				print("\n\tOnsets = " + str(Onsets_list))'''

