import pandas as pd
import numpy as np

df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Ensemble Techniques/Diabeted_Ensemble.csv")

df.isnull().sum()
df.dropna()
df.columns

# Dummy variables
df.head()
df.info()

# Input and Output Split
predictors = df.loc[:, df.columns!=" Class variable"]
type(predictors)

target = df[" Class variable"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42) # n_jobs is basically for parallel processing(parallely the tree have to be grown), bootstrap is whether the samples are replacement or not.

bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test)) # test - 81%

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train)) # training - 100%

# overfitting case

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test)) # 79%

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train)) # 84%

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test)) # 80%

accuracy_score(y_train, boost_clf.predict(x_train)) # 91%

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

# Hyperparameters - to improve the accuracy
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf2.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, boost_clf2.predict(x_test))
accuracy_score(y_test, boost_clf2.predict(x_test)) # 79%

# Evaluation on Training Data
accuracy_score(y_train, boost_clf2.predict(x_train)) # 80%

import xgboost as xgb # pip install xgboost

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)


xgb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test)) # 77%

# Evaluation on Training Data
accuracy_score(y_train, xgb_clf.predict(x_train)) # 100%

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, cv_xg_clf.predict(x_test))
accuracy_score(y_test, cv_xg_clf.predict(x_test)) # 78%
grid_search.best_params_


"{'colsample_bytree': 0.8,
 'gamma': 0.3,
 'max_depth': 9,
 'rag_alpha': 0.01,
 'subsample': 0.9}"







