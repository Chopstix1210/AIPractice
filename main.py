# Projected created by Kevin Segovia on 12-26-2021

# Imports
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# # shape
# print(dataset.shape)
# # shows that it is a 150 instace with 5 attributes
#
# # head
# print(dataset.head(20))
# # shows the first 20 rows of the dataset
#
# # descriptions
# print(dataset.describe())
# # we are able to determine that there is a rnage of 0-8 cm
#
# # class distribution
# print(dataset.groupby('class').size())
# # can see that there are 4 attributes with 3 of them splitting 50 rows

# Now we visualize the data using box and whiskers plot
# Box and Whiskers info: https://datavizcatalogue.com/methods/box_plot.html
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

# Visualize using histograms()
dataset.hist()
pyplot.show()

# Visualize using Scatterplot
scatter_matrix(dataset)
pyplot.show()

# Create a validation dataset
# short desc of what is working we will hold back some data from the algo to test for accuracy.
# about 80% to the algo to train, eval, and select and 20% to validate
array = dataset.values
X = array[:, 0:4]
# array = dataset.values
# X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# more on indexing: https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
# testing harness
# going to use 10-fold cross validation to estimate model accuracy
# train on 9 sets test on 1
# learn more k-fold: https://machinelearningmastery.com/k-fold-cross-validation/
# useing a random_set arg on a fixed_var to know if the algo is evaluated on the same split
# https://machinelearningmastery.com/introduction-to-random-number-generators-for-machine-learning/

# Build the models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Cart', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))

# have to eval every model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results =cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# From the Console we are able to see that the SVM algo has best accuracy but we can also use box and
# whiskers to compare as well

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# SVM is still most accurate so we will use SVM
# We can now make predictions on the data as well
# https://machinelearningmastery.com/make-predictions-scikit-learn/
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Eval the predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

