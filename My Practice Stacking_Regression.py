# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

# Load the dataset
diabetes = load_diabetes() # 442 is the dataset size

train_x, train_y = diabetes.data[:400], diabetes.target[:400] # start till 400
test_x, test_y = diabetes.data[400:], diabetes.target[400:] # after 400 till last - 42
 

# Create the ensemble's base learners and meta-learner
# Append base learners to a list
base_learners = []

# KNN regression model
knn = KNeighborsRegressor(n_neighbors=5)
base_learners.append(knn)

# Decision Tree regressor model
dtr = DecisionTreeRegressor(max_depth=4, random_state=123456)
base_learners.append(dtr)

# Ridge regression
ridge = Ridge()
base_learners.append(ridge)

# Meta model using linear regerssion model
meta_learner = LinearRegression()


# Create the training metadata
# Create variables to store metadata and the targets

meta_data = np.zeros((len(base_learners), len(train_x)))
meta_targets = np.zeros(len(train_x))

# Create the cross-validation folds
KF = KFold(n_splits = 5) # split it into 5 parts
meta_index = 0

for train_indices, test_indices in KF.split(train_x): # train_indices = 320, test_indices = 80 out of 400 and repeats into 5 times
  # Train each learner on the K-1 folds 
  # and create metadata for the Kth fold
  for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x[train_indices], train_y[train_indices])
    predictions = learner.predict(train_x[test_indices])
    meta_data[i][meta_index:meta_index + len(test_indices)] = predictions

  meta_targets[meta_index:meta_index + len(test_indices)] = train_y[test_indices]
  meta_index += len(test_indices)

# Transpose the metadata to be fed into the meta-learner
meta_data = meta_data.transpose()


# Create the metadata for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_errors = []
base_r2 = [] #r2_score- r square score - goodness of fit, how well the data is getting fit with that model 
# r2_score -  always lie between 0 to 1 , 0 means it's not at all fitting, 1 is fitting perfectly.
for i in range(len(base_learners)):
  learner = base_learners[i]
  learner.fit(train_x, train_y)
  predictions = learner.predict(test_x)
  test_meta_data[i] = predictions

  err = metrics.mean_squared_error(test_y, predictions)
  r2 = metrics.r2_score(test_y, predictions)

  base_errors.append(err)
  base_r2.append(r2)

test_meta_data = test_meta_data.transpose()

# Fit the meta-learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

err = metrics.mean_squared_error(test_y, ensemble_predictions)
r2 = metrics.r2_score(test_y, ensemble_predictions)

# Print the results 
for i in range(len(base_learners)):
  learner = base_learners[i]
  print(f'{base_errors[i]:.1f} {base_r2[i]:.2f} {learner.__class__.__name__}')
print(f'{err:.1f} {r2:.2f} Ensemble')

