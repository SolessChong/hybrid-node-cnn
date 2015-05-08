import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class AutoencoderCost(DefaultDataSpecsMixin, Cost):
    supervised = False

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        
        X = data
        X_hat = model.reconstruct(X)
        loss = -(X * T.log(X_hat) + (1 - X) * T.log(1 - X_hat)).sum(axis=1)
        return loss.mean()

class Autoencoder(Model):
    def __init__(self, nvis, nhid):
        super(Autoencoder, self).__init__()

        self.nvis = nvis
        self.nhid = nhid

        W_value = numpy.random.uniform(size=(self.nvis, self.nhid))
        self.W = sharedX(W_value, 'W')
        b_value = numpy.zeros(self.nhid)
        self.b = sharedX(b_value, 'b')
        c_value = numpy.zeros(self.nvis)
        self.c = sharedX(c_value, 'c')
        self._params = [self.W, self.b, self.c]

        self.input_space = VectorSpace(dim=self.nvis)

    def reconstruct(self, X):
        h = T.tanh(T.dot(X, self.W) + self.b)
        return T.nnet.sigmoid(T.dot(h, self.W.T) + self.c)