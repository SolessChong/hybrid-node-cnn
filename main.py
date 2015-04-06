import numpy as np
from sklearn import svm
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

def split_valid(data, label, fold):
	split = int(data.shape[0] / fold)

	data_t = data[0:split,]
	label_t = label[0:split,]
	data_v = data[split:,]
	label_v = label[split:,]

	return data_t, label_t, data_v, label_v

def solve(data, label):
	print "Solving problem..."
	clf = svm.SVC()
	clf.fit(data, label)

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
	data_t, label_t, data_v, label_v = split_valid(data, label, 5)
	clf = solve(data_t, label_t)
	rst = evaluate(clf, data_v, label_v)

	write_data('output.csv', pred)