import pickle
import numpy as np
import matplotlib.pyplot as plt

class solidMLP:
	def __init__(self, params):
		self.h0_W = params[0].get_value()
		self.h0_b = params[1].get_value()
		self.softmax_b = params[2].get_value()
		self.softmax_W = params[3].get_value()

	def output(self, X):
		def sigmoid(x):	
			return 1.0 / (1.0 + np.exp(-x))
		def softmax(x):
			return x 

		h = sigmoid(np.dot(X, self.h0_W) + self.h0_b)
		o = np.dot(h, self.softmax_W) + self.softmax_b
		return o

if __name__ == "__main__":
	f = open('mlp_best.pkl', 'r')
	p = pickle.load(f)
	s = solidMLP(p.get_params())
	f.close()

	f  = open('data/mnist_valid_X.pkl', 'r')
	data = pickle.load(f)

	for i in range(10):
		print np.argmax(s.output(data[i]))
		plt.imshow(data[i].reshape((28,28)))
		plt.show()