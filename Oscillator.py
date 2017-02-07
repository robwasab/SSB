import numpy as np

class Oscillator(object):
	def __init__(self, *args, **kwargs):
		self.fc = 10E3
		self.fs = 44.1E3
		for key in kwargs:
			if key == 'fc':
				self.fc = float(kwargs[key])
			elif key == 'fs':
				self.fs = float(kwargs[key])

		self.omega = 2.0 * np.pi * self.fc / self.fs
		self.x = 1.0
		self.y = 0.0
		self.cos = np.cos(self.omega)
		self.sin = np.sin(self.omega)

	def calc(self):
		x = self.cos*self.x - self.sin*self.y
		y = self.sin*self.x + self.cos*self.y
		self.x = x
		self.y = y
		return (x,y)

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
	osc = Oscillator(fc=1E3, fs=44.1E3)
	pts = 500
	a = np.zeros(pts)
	b = np.zeros(pts)
	tim = (np.arange(pts)+1) / 44.1E3
	for k in range(pts):
		a[k],b[k] = osc.calc()
	plt.plot(tim, a)
	plt.plot(tim, b)
	plt.plot(tim, np.cos(2.0*np.pi*1E3*tim), color='m', alpha=0.5)
	plt.show()
