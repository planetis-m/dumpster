# import necessary modules
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import export_text

# load the iris dataset
iris = load_iris()

# separate the features and target
X = iris.data
y = iris.target

# set the scoring metrics to be used for evaluating the models
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Create a decision tree classifier
clf = tree.DecisionTreeClassifier(max_depth=None, criterion='gini', max_leaf_nodes=5, min_samples_split=10, min_samples_leaf=1)
# Train the classifier on the training data
clf.fit(X, y)

# perform cross-validation on the KNN classifier
scores = cross_validate(clf, X, y, cv=10, scoring=scoring)

# print the mean scores for each scoring metric
print("DecisionTreeClassifier:")
print("Accuracy:", scores['test_accuracy'].mean())
print("Precision:", scores['test_precision_macro'].mean())
print("Recall:", scores['test_recall_macro'].mean())
print("F1-score:", scores['test_f1_macro'].mean())

# DecisionTreeClassifier:
# Accuracy: 0.9666666666666666
# Precision: 0.9722222222222223
# Recall: 0.9666666666666666
# F1-score: 0.9663299663299663

tree_rules = export_text(clf, class_names=iris['target_names'])
print(tree_rules)

from sklearn.ensemble import RandomForestClassifier

# # Create a RandomForestClassifier with the given hyperparameters
rfc = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=10)
# Train the classifier on the training data
rfc.fit(X, y)

# perform cross-validation on the KNN classifier
scores = cross_validate(rfc, X, y, cv=10, scoring=scoring)

# print the mean scores for each scoring metric
print("RandomForestClassifier:")
print("Accuracy:", scores['test_accuracy'].mean())
print("Precision:", scores['test_precision_macro'].mean())
print("Recall:", scores['test_recall_macro'].mean())
print("F1-score:", scores['test_f1_macro'].mean())

# RandomForestClassifier:
# Accuracy: 0.9666666666666666
# Precision: 0.9722222222222223
# Recall: 0.9666666666666666
# F1-score: 0.9663299663299663

for i, tree in enumerate(rfc.estimators_):
  tree_rules = export_text(tree, class_names=iris['target_names'])
  print("Estimator ", i)
  print(tree_rules)

# from sklearn.model_selection import GridSearchCV
#
# # Create a Random Forest classifier
# rfc = RandomForestClassifier()
#
# # Define the parameter grid for hyperparameter tuning
# param_grid = {
#     'n_estimators': [5, 10, 20, 30, 50],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # Perform grid search with cross-validation
# grid_search = GridSearchCV(rfc, param_grid, cv=5)
# grid_search.fit(X, y)
#
# # Get the best parameters from the grid search
# best_params = grid_search.best_params_
#
# print(f"Best Parameters: {best_params}")
