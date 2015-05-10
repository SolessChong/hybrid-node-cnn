import theano
import theano.tensor as T
import pickle
from ae import Autoencoder
from solid_ae import SolidAutoencoder
import matplotlib.pyplot as plt

# Load data
f = open('data/mnist_valid_X.pkl', 'rb')
d = pickle.load(f)
f.close()

# Read params
f = open('ae_train.pkl', 'rb')
p = pickle.load(f)
f.close()

# Construct model
ae = Autoencoder(784, 200)
ae.W.set_value(p.W.get_value())
ae.b.set_value(p.b.get_value())
ae.c.set_value(p.c.get_value())

# Calculate
x = T.vector()
func = theano.function([x], ae.reconstruct(x))

r = func(d[0])

# Visualize
plt.imshow(r.reshape((28,28)))
plt.show()

# Construct solid model
sae = SolidAutoencoder(784, 200, p)
rst = sae.reconstruct(d[0])
plt.imshow(rst.reshape((28,28)))
plt.show()