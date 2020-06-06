# Import Libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# Load Datasets - Uncomment 2/3 per run to test each dataset!
#dataset = datasets.load_wine()
#dataset = datasets.load_digits()
dataset = datasets.load_breast_cancer()

# Fit Naive Bayes Model to the chosen Dataset
model = GaussianNB()
model.fit(dataset.data, dataset.target)

# Make the Predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# Print Results
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
