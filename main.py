import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import decomposition
import matplotlib.pyplot as plt

def load_data(train_fn, test_fn):
	print "Loading data..."
	content = np.loadtxt(open(train_fn, 'rb'), delimiter=",", skiprows=1)
	data = content[:,1:]
	label = content[:,0]
	content = np.loadtxt(open(test_fn, 'rb'), delimiter=",", skiprows=1)
	test = content

	return data, label, test

def write_data(fn, pred):
	print "Writing predictions..."
	with open(fn, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=",")
		writer.writerow(["ImageId", "Label"])
		for i in range(len(pred)):
			writer.writerow([i+1, pred[i]])

def preprocess(data, label, test, lite=None):
	print "Preprocessing data..."

	# Scale
	data = data / 255 * 2 - 1
	test = test / 255 * 2 - 1
	label = label.astype(int)

	# Shuffle
	shuffle = np.random.permutation(len(data))
	data = data[shuffle,]
	label = label[shuffle,]

	# Lite set
	if lite:
		data = data[1:lite, :]
		test = test[1:lite, :]
		label = label[1:lite]

	# PCA
	pca = decomposition.PCA(n_components=80)
	pca.fit(data)

	data = pca.transform(data)
	test = pca.transform(test)

	return data, label, test

def search_params(data, label):
	print "Searching for best params..."

	# Grid search
	""" Iter #1
	tuned_params = [{
			'kernel': ['rbf'], 
			'gamma': [1e-3, 1e-4],
			'C': [1, 10, 100, 1000]
		}]
	"""
	
	tuned_params = [{
		'kernel': ['rbf'],
		'gamma': [1e-1,1e-2,1e-3],
		'C': [30, 100, 300]
	}]

	clf = grid_search.GridSearchCV(
		svm.SVC(C=1), tuned_params, cv=5
		)
	clf.fit(data, label)

	print "Best params are: ", clf.best_params_

	return clf

def solve(clf, data, label):
	print "Solving problem..."

	clf.fit(data, label)

	return clf

def evaluate(clf, data, label):
	print "Evaluating the classifier..."

	pred = clf.predict(data)
	mse = np.sqrt(((pred - label) ** 2).mean())

	print "MSE is ", mse

	return mse

def vis_img(img):
	if img.ndim == 2:
		plt.imshow(img.reshape((28,28)))
	plt.show()

if __name__ == "__main__":
	#LITE = 10000
	LITE = None

	data_o, label_o, test_o = load_data('data/train.csv', 'data/test.csv')
	data, label, test = preprocess(data_o, label_o, test_o, LITE)

	# Validation
	data_t, data_v, label_t, label_v = \
		cross_validation.train_test_split(data, label, test_size=0.2)
	clf = search_params(data_t, label_t)
	performance = evaluate(clf, data_v, label_v)

	write_data('output.csv', pred)