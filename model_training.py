import numpy as np
from sklearn import datasets, model_selection
from decision_tree import DecisionTree

iris = datasets.load_iris()

X = np.array(iris.data)
Y = np.array(iris.target)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

my_tree = DecisionTree()
my_tree.train(X_train, Y_train)
my_tree.print_tree()

train_preds = my_tree.predict(X_set=X_train)
print("Train size", len(Y_train))
print("True preds", sum(train_preds == Y_train))
print("Accuracy", sum(train_preds == Y_train) / len(Y_train))