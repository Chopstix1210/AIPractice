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

# shape
print(dataset.shape)
# shows that it is a 150 instace with 5 attributes

# head
print(dataset.head(20))
# shows the first 20 rows of the dataset

# descriptions
print(dataset.describe())
# we are able to determine that there is a rnage of 0-8 cm

# class distribution
print(dataset.groupby('class').size())
# can see that there are 4 attributes with 3 of them splitting 50 rows

# Now we visualize the data using box and whiskers plot
# Box and Whiskers info: https://datavizcatalogue.com/methods/box_plot.html
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
