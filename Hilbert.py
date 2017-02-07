import numpy as np
from numpy import sin
from numpy import cos

pi = np.pi

def A(t, w1, w2, a):
	if t == 0.0:
		return np.sqrt(2.0) * (w2 - w1)
	elif t == pi/2.0/a:
		return a * ( sin( pi/4.0 * ( ( a + 2.0*w2 )/a ) ) - sin( pi/4.0 * ( a + 2.0*w1 )/a ) )
	elif t == -pi/2.0/a:
		return a * ( sin( pi/4.0 * ( ( a - 2.0*w1 )/a ) ) - sin( pi/4.0 * ( a - 2.0*w2 )/a ) )
	else:
		return 2.0*pi*pi*cos(a*t)/(t*(4.0*a*a*t*t-pi*pi))*(sin(w1*t+pi/4.0)-sin(w2*t + pi/4.0))

def A_k(k, n, w1, w2, a):
	return A(2.0*pi*(k-(n-1)/2.0), w1, w2, a)

class Hilbert(object):
	def __init__(self, *args, **kwargs):
		self.f1 = 0.05
		self.f2 = 0.45
		self.a = 0.05
		for key in kwargs:
			if key == 'f1':
				self.f1 = float(kwargs[key])
			elif key == 'f2':
				self.f2 = float(kwargs[key])
			elif key == 'a':
				self.a = float(kwargs[key])
		print 'f1: %.3f'%((self.f1+self.a) * 44.1E3)
		print 'f2: %.3f'%((self.f2-self.a) * 44.1E3)

	def calc(self, n):
		a_coefs = np.zeros(n)

		for k in np.arange(n):
			coef = A_k(k, n, self.f1, self.f2, self.a)
			a_coefs[k] = coef

		b_coefs = a_coefs[np.arange(n-1, -1, -1)]

		return (a_coefs, b_coefs)
