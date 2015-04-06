import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
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

def preprocess(data, label, test, lite=False):
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
		data = data[1:10000, :]
		test = test[1:10000, :]
		label = label[1:10000]

	return data, label, test

def solve(data, label):
	print "Solving problem..."

	# Grid search
	tuned_params = [
		{
			'kernel': ['rbf'], 
			'gamma': [1e-3, 1e-4],
			'C': [1, 10, 100, 1000]
		}]

	clf = grid_search.GridSearchCV(
		svm.SVC(C=1), tuned_params, cv=5
		)
	clf.fit(data, label)

	print "Best params are: ", solver.best_params_

	return clf

def evaluate(clf, data, label):
	print "Evaluating the classifier..."

	pred = clf.predict(data)
	mse = ((pred - label) ** 2).mean()

	print "MSE is ", mse

	return mse

if __name__ == "__main__":
	LITE = True

	data, label, test = load_data('data/train.csv', 'data/test.csv')
	data, label, test = preprocess(data, label, test, LITE)

	# Validation
	data_t, data_v, label_t, label_v = \
		cross_validation.train_test_split(data, label, test_size=0.2)
	clf = solve(data_t, label_t)
	performance = evaluate(clf, data_v, label_v)

	write_data('output.csv', pred)