import pickle
import numpy as np

from ae import Autoencoder
import matplotlib.pyplot as plt

class SolidAutoencoder():
	def __init__(self, nvis, nhid, data):
		self.nvis = nvis
		self.nhid = nhid

		self.W = data.W.get_value()
		self.b = data.b.get_value()
		self.c = data.c.get_value()

	def reconstruct(self, X):
		h = np.tanh(np.dot(X, self.W) + self.b)
		o = np.dot(h, np.transpose(self.W)) + self.c
		return 1.0 / (1.0 + np.exp(-o))