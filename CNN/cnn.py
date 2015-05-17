import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

class CNNCost(DefaultDataSpecsMixin, Cost):
	supervised = True

	def expr(self, model, data, **kwargs):
		space, source = self.get_data_specs(model)
		space.validate(data)

		inputs, targets = data
		outputs = model.cnn_output(inputs)
		loss = -(targets * T.log(outputs)).sum(axis=1)

		return loss.mean()

class CNN(Model):
	"""
	W1: [nvis * nhid]
	b: 	[nhid]
	W2: [nhid * 1]
	c: 	[1]
	"""
	def __init__(self, nvis, nhid, nclasses):
		super(CNN, self).__init__()

		self.nvis = nvis
		self.nhid = nhid
		self.nclasses = nclasses

		W1_value = numpy.random.uniform(size=(self.nvis, self.nhid))
		b_value = numpy.random.uniform(size=(self.nhid))
		W2_value = numpy.random.uniform(size=(self.nhid, nclasses))
		c_value = numpy.random.uniform(size=(nclasses))

		self.W1 = sharedX(W1_value, 'W1')
		self.W2 = sharedX(W2_value, 'W2')
		self.b = sharedX(b_value, 'b')
		self.c = sharedX(c_value, 'c')

		self._params = [self.W1, self.W2, self.b, self.c]
		self.input_space = VectorSpace(dim=self.nvis)
		self.output_space = VectorSpace(dim=self.nclasses)

	def cnn_output(self, X):
		h = T.tanh(T.dot(X, self.W1) + self.b)
		o = T.tanh(T.dot(h, self.W2) + self.c)
		return T.nnet.softmax(o)
		
