#%% [markdown]
# Import libraries and print versions.

#%%
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))

#%% [markdown]
# Read Iris dataset

#%%
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv('./iris.data', names = columns)
dataset.sample(5)

#%% [markdown]
# Summarize dataset

#%%
dataset.describe()

#%% [markdown]
# Boxplot

#%%
import matplotlib.pyplot as plt
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#%% [markdown]
# Histogram

#%%
dataset.hist()
plt.show()

#%% [markdown]
# Scatter plot

#%%
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

#%% [markdown]
# Split the dataset. Use some for training and some for validation.

#%%
from sklearn import model_selection

array = dataset.values
X = array[:,0:4] # select only flower attributes (the input data)
Y = array[:,4] # select only flower classes (the result)

# what is this for??
seed = 10 # some number...

# this splits the data. using 20% for validation, thus 80% for training
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = seed)

#%% [markdown]
# Initialize models we are using

#%%
# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('k-neighbors', KNeighborsClassifier()))
models.append(('Decision tree', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))

#%% [markdown]
# Run the models

#%%
results = []
names = []

for modelname, model in models:
    kfold = model_selection.KFold(n_splits = 20, random_state = seed) # 10-fold cross-validation
    res = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
    results.append(res)
    names.append(modelname)
    res.mean()
    print("%s %f, std=%f" % (modelname, res.mean(), res.std()))

#%% [markdown]
# Compare all algorithms by drawing a chart.

#%%
fig = plt.figure()
fig.suptitle('Compare algorithms')
ax = fig.add_subplot(111)
plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()

#%% [markdown]
# Discover SVM results. Print some sklearn metrics.

#%%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# train support-vector-machine
svm = SVC(gamma = 'auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)

# accuracy score
accuracy_score(Y_validation, predictions)

#%% [markdown]
# Confusion matrix. For 150 samples total, 0.2*150 = 30 predictions total. For each of 3 classes.

#%%
confusion_matrix(Y_validation, predictions)

#%% [markdown]
# Classification report

#%%
classification_report(Y_validation, predictions)

#%% [markdown]
# 
#%% [markdown]
# code inspired by https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

