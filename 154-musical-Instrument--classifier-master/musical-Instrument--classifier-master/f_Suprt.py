# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:54:02 2018

@author: adwaithmk
"""
import numpy as np
import librosa
import os
import pandas as pd
import essentia as es
from essentia.standard import *
from pyAudioAnalysis import audioFeatureExtraction as pya
from scipy import signal
import warnings
import matplotlib.pyplot as plt

def spectralEnvelope(h_ampl):
	"""
		given the harmonic amplitude, estimate the spectral envelope, moving average of window_size=3
	"""
	env = np.zeros(len(h_ampl))
	kk = 0
	env[kk] = (h_ampl[kk] + h_ampl[kk+1]) / 2
	for kk in xrange(1,len(h_ampl)-1):
		env[kk] = (h_ampl[kk-1] + h_ampl[kk] + h_ampl[kk+1])/3
	kk = len(h_ampl)
	env[kk-1] = (env[kk-2] + env[kk-1]) / 2
	return env


def stft_(wavedata, fs, window_size, hopsize, mode='psd'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec, f, t, p = plt.specgram(wavedata, Fs=fs,NFFT=window_size, noverlap=(window_size-hopsize), mode=mode, scale='dB')
    return spec, f, t

def spectrum(wavedata,window_size,mode='psd'):
    spec, f, t = stft_(wavedata, 44100, window_size,window_size, mode=mode)
    # to transform to dB scale: 10*log(psd)
    spectrum = np.mean(spec, axis=1)
    return spectrum, f

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    source: https://github.com/endolith/waveform-analyzer/blob/master/common.py
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def estimated_f0(wavedata, window_size):
    """using the peak of the spectrum to find the estimated fundamental frequency
    sometime the harmonics may be stronger than the f0
    """
    spec, f = spectrum(wavedata, window_size)
    ipeak = np.argmax(spec)
    if ipeak:
        ipeak2 = np.argmax(spec[:ipeak])
        if spec[ipeak] - spec[ipeak2] < 0.2 * abs(spec[ipeak]):
            ipeak = ipeak2
    i_interp = parabolic(spec, ipeak)[0]
    return i_interp * 44100.0 / window_size

def harmonics(wavedata,window_size,c=0.1):
    f0 = estimated_f0(wavedata,window_size)
    spec, f = spectrum(wavedata, window_size)
    num_h = int(0.5*44100/f0)
    harmo = np.array([i*f0 for i in xrange(1,num_h+1)])
    harmonics = []

    for h in xrange(num_h):
        possible_hz = (max(0, harmo[h]-c*f0), min(harmo[h]+c*f0, 44100.0/2-44100.0/window_size))
        start = int(possible_hz[0]/44100*window_size)
        end = int(possible_hz[1]/44100*window_size)+1
        if start < end:
            local_max_pos = np.argmax(spec[start:end]) + start
            fq = parabolic(spec, local_max_pos)[0] * 44100.0 / window_size
            mag = spec[local_max_pos]
            harmonics.append((fq, mag))
    return np.array(harmonics)

def harmonicCentroid(harmo):
    """
        harmonics is n*2 array, first column is the hamonic frequency, second col is the magnitude
    """
    return np.sum(np.product(harmo,axis=1)) / np.sum(harmo[:,1])

def harmonicDeviation(harmo):
    """the absolute deviation between the amplitude and the envelope"""
    freq = harmo[:,0]
    h_ampl = harmo[:,1]
    env = spectralEnvelope(h_ampl)
    hd = np.sum(np.abs(h_ampl - env)) / len(env)
    return hd

def harmonicSpead(harmo):
    hc = harmonicCentroid(harmo)
    freq = harmo[:,0]
    h_ampl = harmo[:,1]
    num = np.sum(h_ampl * (freq - hc) ** 2)
    denum = np.sum(h_ampl)
    return np.sqrt(num/denum)



def F_find(x,sr):
	#x, sr=librosa.load(file_name1,sr=44100,mono=True)
	#print sr
	sound=x
	samplerate=sr
	'''spec=Spectrum()
	w=Windowing(type='hann')
	if np.size(sound)%2!=0:
		x=sound[:-1]
	spc=spec(w(x))'''
	#print spc
	y=es.array(sound)
	l=LogAttackTime()
	lat,astrt,astp=l(y)
	#print lat
	#flu=Flux()
	#s_flux=flu(spc)
	#print s_flux
	spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
	mspec_bw=np.mean(spec_bw)
	#print mspec_bw
	sdspec_bw=np.std(spec_bw)
	#print sdspec_bw
	#print mspec_bw,sdspec_bw
	mfcc=librosa.feature.mfcc(y=sound, sr=samplerate,n_mfcc=11)
	#print mfcc
	mfcc_1=mfcc[0]
	mfcc_2=mfcc[1]
	mfcc_3=mfcc[2]
	mfcc_4=mfcc[3]
	mfcc_5=mfcc[4]
	mfcc_6=mfcc[5]
	mfcc_7=mfcc[6]
	mfcc_8=mfcc[7]
	mfcc_10=mfcc[9]
	mfcc_11=mfcc[10]

	mmfcc_1=np.mean(mfcc_1)
	mmfcc_2=np.mean(mfcc_2)
	mmfcc_3=np.mean(mfcc_3)
	mmfcc_4=np.mean(mfcc_4)
	mmfcc_5=np.mean(mfcc_5)
	mmfcc_6=np.mean(mfcc_6)
	mmfcc_7=np.mean(mfcc_7)
	mmfcc_10=np.mean(mfcc_10)

	sdmfcc_1=np.std(mfcc_1)
	
	sdmfcc_3=np.std(mfcc_3)
	
	sdmfcc_4=np.std(mfcc_4)
	
	sdmfcc_6=np.std(mfcc_6)
	
	sdmfcc_7=np.std(mfcc_7)
	
	sdmfcc_8=np.std(mfcc_8)
	
	sdmfcc_10=np.std(mfcc_10)
	
	sdmfcc_11=np.std(mfcc_11)
	#print sdmfcc_1,sdmfcc_3,sdmfcc_4,sdmfcc_6,sdmfcc_7,sdmfcc_8,sdmfcc_10,sdmfcc_11	
	
	#print mmfcc_1,mmfcc_2,mmfcc_3,mmfcc_4,mmfcc_5,mmfcc_6,mmfcc_7,mmfcc_10	
	#print mmfcc_4,mmfcc_10
	'''rms=librosa.feature.rmse(y=sound)
	mrms=np.mean(rms)
	print mrms'''
	r=RMS()
	rms_e=r(y)
	mrms= np.mean(rms_e)
	
	zcr = librosa.feature.zero_crossing_rate(y=sound)
	mzcr=np.mean(zcr)
	sdzcr=np.std(zcr)
	#print  mzcr,sdzcr
	
	cent = librosa.feature.spectral_centroid(y=sound, sr=samplerate)
	mcent=np.mean(cent)	
	sdcent=np.std(cent)
	#print mcent,sdcent


	f, t, Zxx = signal.stft(sound,44100,window='hann',nperseg=441,noverlap=0)		
	stf=np.zeros(np.size(t))							   
	for i in range(1,np.size(t)):
		X=abs(Zxx[:,[i]])
		X_prev=abs(Zxx[:,[i-1]])
		stf[i-1]=pya.stSpectralFlux(X,X_prev)
	sdsf=np.std(stf)
	msf=np.mean(stf)
	#print sdsf

	harmo=harmonics(sound,441)
	HC=harmonicCentroid(harmo)
	#print HC,'-'
	HD=harmonicDeviation(harmo)
	#print HD,'--'
	HS=harmonicSpead(harmo)
	#print HS   	
	rawdata={'LogAttackTime':lat,		 			 
			 'RMSM':mrms,		 
			 'mfcc_1D':sdmfcc_1,
			 'mfcc_3D':sdmfcc_3,			 
			 'mfcc_4D':sdmfcc_4,
			 'mfcc_6D':sdmfcc_6,
			 'mfcc_7D':sdmfcc_7,
			 'mfcc_8D':sdmfcc_8,			 			 
			 'mfcc_10D':sdmfcc_10,
			 'mfcc_11D':sdmfcc_11,
			 'spectral_centroidM':mcent,
			 'spectral_centroidD':sdcent,
			 'mfcc_1M':mmfcc_1,
			 'mfcc_2M':mmfcc_2,
			 'mfcc_3M':mmfcc_3,
			 'mfcc_4M':mmfcc_4,
			 'mfcc_5M':mmfcc_5,
			 'mfcc_6M':mmfcc_6,
			 'mfcc_7M':mmfcc_7,
			 'mfcc_10M':mmfcc_10,
			 'ZCRM':mzcr,
			 'ZCRD':sdzcr,
			 'HC':HC,			 
			 'HD':HD,			 
			 'HS':HS,			 
			 'FluxM':msf,
			 'FluxD':sdsf,			 
			 'spectral_bandwidthM':mspec_bw,
			 'spectral_bandwidthD':sdspec_bw,			 
			 }
	df3 = pd.DataFrame(data=rawdata, columns = ['LogAttackTime','RMSM','mfcc_1D','mfcc_3D','mfcc_4D','mfcc_6D','mfcc_7D','mfcc_8D','mfcc_10D','mfcc_11D',
			'spectral_centroidM','spectral_centroidD','mfcc_1M','mfcc_2M','mfcc_3M','mfcc_4M','mfcc_5M','mfcc_6M','mfcc_7M','mfcc_10M',
			'ZCRM','ZCRD','HC','HD','HS','FluxM','FluxD','spectral_bandwidthM','spectral_bandwidthD'],index=[0])
	'''print df
	'''
	rawdata={'LogAttackTime':lat,
			 'HD':HD,
			 'FluxD':sdsf,
			 'spectral_bandwidthM':mspec_bw,   
			 'mfcc_1D':sdmfcc_1,
			 'mfcc_3D':sdmfcc_3,
			 'RMSM':mrms,
			 'spectral_bandwidthD':sdspec_bw,
			 'mfcc_4M':mmfcc_4,
			 'mfcc_11D':sdmfcc_11,
			 'ZCRD':sdzcr,
			 'spectral_centroidD':sdcent,
			 'mfcc_8D':sdmfcc_8,
			 'mfcc_6D':sdmfcc_6,
			 'mfcc_7D':sdmfcc_7,
			 'mfcc_4D':sdmfcc_4,
			 'spectral_centroidM':mcent,
			 'mfcc_10M':mmfcc_10,
			 'mfcc_10D':sdmfcc_10,
			 }
	df1 = pd.DataFrame(data=rawdata, columns = ['LogAttackTime','HD','FluxD','spectral_bandwidthM','mfcc_1D','mfcc_3D','RMSM','spectral_bandwidthD',
		'mfcc_4M','mfcc_11D','ZCRD','spectral_centroidD','mfcc_8D','mfcc_6D','mfcc_7D','mfcc_4D','spectral_centroidM','mfcc_10M','mfcc_10D'],index=[0])
	rawdata={'spectral_centroidM':mcent,
			 'mfcc_2M':mmfcc_2,
			 'HC':HC,
			 'ZCRM':mzcr,   
			 'mfcc_3M':mmfcc_3,
			 'HD':HD,
			 'ZCRD':sdzcr,
			 'HS':HS,
			 'mfcc_4M':mmfcc_4,
			 'mfcc_1M':mmfcc_1,
			 'mfcc_10M':mmfcc_10,
			 'FluxM':msf,
			 'FluxD':sdsf,
			 'mfcc_5M':mmfcc_5,
			 'mfcc_7M':mmfcc_7,
			 'spectral_bandwidthM':mspec_bw,
			 'spectral_centroidD':sdspec_bw,
			 'mfcc_6M':mmfcc_6,
			 }
	df2 = pd.DataFrame(data=rawdata, columns = ['spectral_centroidM','mfcc_2M','HC','ZCRM','mfcc_3M','HD','ZCRD','HS','mfcc_4M','mfcc_1M','mfcc_10M','FluxM','FluxD',
			'mfcc_5M','mfcc_7M','spectral_bandwidthM','spectral_centroidD','mfcc_6M'],index=[0])
	#print df1,df2,df3














	'''
	f = open("dataset153.csv", 'a') 
	df.to_csv(f, header = False,index=False)
	f.close()'''
	return df1,df2,df3

	


	   