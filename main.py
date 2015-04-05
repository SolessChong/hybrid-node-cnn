import numpy as np
from sklearn import svm

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

def solve(data, label, test):
	print "Solving problem..."
	clf = svm.SVC()
	clf.fit(data, label)
	rst = clf.predict(test)

	return rst

if __name__ == "__main__":
	data, label, test = load_data('data/train.csv', 'data/test.csv')
	pred = solve(data, label, test)
	write_data('output.csv', pred)