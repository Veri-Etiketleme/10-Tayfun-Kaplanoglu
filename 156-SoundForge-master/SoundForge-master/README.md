#SoundForge
==========

helps you tune the piano, the guitar, or the human vocal chords

###Version 0.0.1 pre-alpha

##About
==========

To figure out what notes resonate on what frequency, I followed these charts
http://www.phy.mtu.edu/~suits/notefreqs.html <br />
http://en.wikipedia.org/wiki/Piano_key_frequencies (for piano)

There are 3 parts to this project. <br />
1) Get raw signal input from Microphone <br />
2) Get Frequency from raw signal input <br />
3) Display output (aka Musical Notes resonating on what frequency)

###1) Get input from Microphone
For this I used the recorder class after a random google for 'python record microphone' which yields this site http://www.swharden.com/blog/2013-05-09-realtime-fft-audio-visualization-with-python/

It's capturing raw signal works pretty well so yeah. Thanks Mr Harden.

###2) Get Frequency from raw signal
Rather than writing my own frequency estimation code (aka write my own implementation of Quadratic Interpolation of Sinusoidal Spectrum-Analysis peaks through Quadratic Polynomial, Fast Fourier Transform, and Gaussian probability density function) I decided to use an already prebuilt library available here:

https://gist.github.com/endolith/255291

All hail open source! :D

There are four methods to estimate frequency, 

1) by counting zero crossings and divide average period by time to get frequency
2) using Fast Fourier Transform to find peak
3) using autocorrelation to find peak
4) Calculate Harmonic Product Spectrum (HPS) and find the peak

Any of the four methods work, although depending on source, some method works better than the other. For example, tuning human vocal chords works well with autocorrelation, while tuning a piano works better using HPS. Read http://cnx.org/content/m11714/latest/

Anyways, thanks to endolith for the heavy lifting

###3) Display output (aka Musical Notes resonating on what frequency)

Initially I wanted to do a Frets on Fire clone or something but in the end I decided to scale it down so that all it does is display the notes played.

#How it works
==========

Initially I was writing tons of stuff about maths and science and crap so screw that TLDR stuff. How this app works? Download the app, stick a microphone into your pc, and play. To change the note range, press 'A key' to increase the lowest note or 'Z key' to lower it. For the highest note, press 'S key' to increase it or 'X key' to lower it. Thats it.
