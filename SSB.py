from scipy.signal import convolve
from Hilbert import Hilbert
import numpy as np

pi = np.pi

class SSB(object):
	def __init__(self, *args, **kwargs):
		# number of fir samples
		self.n = 50
		self.fc = 10E3
		self.fs = 44.1E3
		# mix in phase or quadrature phase
		self.in_phase = True

		for key in kwargs:
			if key == 'fc':
				self.fc = float(kwargs[key])
			elif key == 'fs':
				self.fs = float(kwargs[key])
			elif key == 'in_phase':
				self.in_phase = bool(kwargs[key])
			elif key == 'n':
				self.n = int(kwargs[key])

		hilbert = Hilbert()
		self.a_coefs, self.b_coefs = hilbert.calc(self.n)

	def calc(self, m):
		# not meant to run in real time
		sig_a = convolve(m, self.a_coefs)[:len(m)]
		sig_b = convolve(m, self.b_coefs)[:len(m)]

		w = 2.0*pi*self.fc/self.fs
		if self.in_phase == True:
			sig_a *=  np.cos(w*np.arange(pts))
			sig_b *= -np.sin(w*np.arange(pts))
		else:
			sig_a *=  np.sin(w*np.arange(pts))
			sig_b *=  np.cos(w*np.arange(pts))

		ssb = 2.0*(sig_a + sig_b)
		return ssb

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
	from numpy.fft import fft

	gen = SSB(n=50, fc=10E3, fs=44.1E3, in_phase=True)
	H_a = np.fft.fft(gen.a_coefs)
	H_b = np.fft.fft(gen.b_coefs)
	H_diff = H_a - H_b

	plt.subplot(121)
	plt.plot(gen.a_coefs, color='b')
	plt.plot(gen.b_coefs, color='m')

	plt.subplot(222)
	freqs = np.arange(gen.n) / float(gen.n) * 44.1E3
	plt.plot(freqs, 10.0*np.log10(np.abs(H_diff)))

	plt.subplot(224)
	plt.plot(freqs, (np.unwrap(np.angle(H_a)) - np.unwrap(np.angle(H_b)))*180.0/pi)

	cycles = 128
	fc = 1.5E3
	fs = 44.1E3
	F = fc/fs
	pts = int(round(fs/fc * cycles))
	sig = np.cos(2.0*pi*F*np.arange(pts))

	ssb = gen.calc(sig)
	dsb = sig * np.cos(2.0*pi*gen.fc/gen.fs*np.arange(pts))

	plt.figure(2)

	ssb_fft = fft(ssb)/float(pts)
	dsb_fft = fft(dsb)/float(pts)
	freqs = np.arange(pts)/float(pts)*fs
	plt.plot(freqs, 10.0 * np.log10(np.abs(ssb_fft)))
	plt.plot(freqs, 10.0 * np.log10(np.abs(dsb_fft)), color='orange', alpha=0.75)
	plt.show()
