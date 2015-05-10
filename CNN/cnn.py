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
		outputs = model.output(inputs)
		loss = -(targets * T.log(outputs)).sum(axis=1)

		return loss.mean()

