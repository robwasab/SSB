import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, firwin, convolve
from Hilbert import Hilbert
import numpy as np
from numpy.fft import fft


n = 50 
pi = np.pi
hilbert = Hilbert()

a_coefs, b_coefs = hilbert.calc(n)

H_a = np.fft.fft(b_coefs)
H_b = np.fft.fft(a_coefs)
H_diff = H_a - H_b

plt.subplot(121)
plt.plot(a_coefs, color='b')
plt.plot(b_coefs, color='m')

plt.subplot(222)
freqs = np.arange(n) / float(n) * 44.1E3
plt.plot(freqs, 10.0*np.log10(np.abs(H_diff)))

plt.subplot(224)
plt.plot(freqs, (np.unwrap(np.angle(H_a)) - np.unwrap(np.angle(H_b)))*180.0/pi)

cycles = 128
fc = 1.5E3
fs = 44.1E3
F = fc/fs
pts = int(round(fs/fc * cycles))
sig = np.cos(2.0*pi*F*np.arange(pts))

sig_a = convolve(sig, a_coefs)[:pts]
sig_b = convolve(sig, b_coefs)[:pts]

sig   *=  np.cos(2.0*pi*10E3/fs*np.arange(pts))
sig_a *=  np.cos(2.0*pi*10E3/fs*np.arange(pts))
sig_b *= -np.sin(2.0*pi*10E3/fs*np.arange(pts))
#sig_a *= np.sin(2.0*pi*10E3/fs*np.arange(pts))
#sig_b *= np.cos(2.0*pi*10E3/fs*np.arange(pts))

ssb = 2.0*(sig_a + sig_b)
plt.figure(2)
plt.subplot(121)
plt.plot(ssb)

ssb_fft = fft(ssb)/float(pts)
dsb_fft = fft(sig)/float(pts)
freqs = np.arange(pts)/float(pts)*fs
plt.subplot(122)
plt.plot(freqs, 10.0 * np.log10(np.abs(ssb_fft)))
plt.plot(freqs, 10.0 * np.log10(np.abs(dsb_fft)), alpha=0.25)

plt.show()
