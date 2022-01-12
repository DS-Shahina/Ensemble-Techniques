import pandas as pd

df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Ensemble Techniques/movies_classification.csv")

# Dummy variables
df.head()
df.info()

# n-1 dummy variables will be created for n categories
df = pd.get_dummies(df, columns = ["3D_available", "Genre"], drop_first = True)

df.head()


# Input and Output Split
predictors = df.loc[:, df.columns!="Start_Tech_Oscar"]
type(predictors)

target = df["Start_Tech_Oscar"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42) # n_jobs is basically for parallel processing(parallely the tree have to be grown), n_estimators=500- 500 trees

rf_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, rf_clf.predict(x_test))
accuracy_score(y_test, rf_clf.predict(x_test))


######
# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

# create dictionary
param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]} # min 2,3 & 10 records should be there to split otherwise not split, max_features -4,5,6,7,8,9,10 columns

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy') 
# n_jobs = -1, -1 means parallel processing, use all the processor that you have, cv(cross validation = 5) taking the training dataset and split it into 5 parts, it is testing in 1 part 1st by training the 4 parts.

grid_search.fit(x_train, y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, cv_rf_clf_grid.predict(x_test))
accuracy_score(y_test, cv_rf_clf_grid.predict(x_test))
