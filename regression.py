# from sklearn.linear_model import LinearRegression #import statement
# clf = LinearRegression() # we created a classifier from an object named    LinearRegression.
# clf.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2]) #fitting a classifier on a data
# clf.coef_  # calculated the slope


from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
digits = datasets.load_digits()

# setup classifier
clf = svm.SVC(gamma=0.001, C=100.)
# train classifier
clf.fit(digits.data[:-1], digits.target[:-1])

#  predict last value
predicted = clf.predict(digits.data[-1:])
print(predicted)
