"It is a multiclass classification problem. Given dataset has two features one is characters and the other is dependent feature i.e. characters_strength. This characters_strength is classifed as:

"0-character is weak

"1-character is strong

"Let us start with the characters strength classification problem.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes,tree , neural_network , linear_model
from sklearn.model_selection import train_test_split , KFold , RepeatedStratifiedKFold ,cross_val_score
from sklearn.ensemble import VotingClassifier ,StackingClassifier, BaggingClassifier ,  AdaBoostClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

df = pd.read_excel("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Ensemble Techniques/Ensemble_Password_Strength.xlsx")

df.isnull().sum()
df.dropna()
df.columns

# Dummy variables
df.head()
df.info()
df['characters'].unique()

df = pd.get_dummies(df, columns = ["characters" ], drop_first = True)
X=df.iloc[:,1:2000]
Y=df.iloc[:,0]
test_samples = 300
Xtrain, Ytrain = X[:-test_samples], Y[:-test_samples]
Xtest, Ytest = X[-test_samples:], Y[-test_samples:]

"VOTING"

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=2)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

#Hard voting
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(Xtrain, Ytrain)

# Predict the most voted class
hard_predictions = voting.predict(Xtest)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(Ytest, hard_predictions)) # 0.853

#soft voting
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(Xtrain, Ytrain)
learner_4.fit(Xtrain, Ytrain)
learner_5.fit(Xtrain, Ytrain)
learner_6.fit(Xtrain, Ytrain)

# Predict the most probable class
soft_predictions = voting.predict(Xtest)


# Accuracy of hard voting
print('Soft Voting:', accuracy_score(Ytest, soft_predictions)) #0.853

# Get the base learner predictions
predictions_4 = learner_4.predict(Xtest)
predictions_5 = learner_5.predict(Xtest)
predictions_6 = learner_6.predict(Xtest)

# Accuracies of base learners
print('L4:', accuracy_score(Ytest, predictions_4)) #0.853
print('L5:', accuracy_score(Ytest, predictions_5)) #0.146
print('L6:', accuracy_score(Ytest, predictions_6)) #0.853

"STACKING"

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', linear_model.LogisticRegression()))
    level0.append(('knn', neighbors.KNeighborsClassifier()))
    level0.append(('cart', tree.DecisionTreeClassifier()))
    level0.append(('svm', svm.SVC()))
    level0.append(('bayes', naive_bayes.GaussianNB()))
    # define meta learner model
    level1 = linear_model.LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# get a list of models to evaluate
def get_models():
    models = dict()
    models['lr'] = linear_model.LogisticRegression()
    models['knn'] = neighbors.KNeighborsClassifier()
    models['cart'] = tree.DecisionTreeClassifier()
    models['svm'] = svm.SVC()
    models['bayes'] = naive_bayes.GaussianNB()
    models['stacking'] = get_stacking()
    return models

# evaluate a give model using cross-validation
def evaluate_model(model, x, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, Xtrain , Ytrain)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

"lr 0.859 (0.000)
"knn 0.859 (0.000)
"cart 0.859 (0.000)
"svm 0.859 (0.000)
"bayes 0.141 (0.000)
"stacking 0.859 (0.000)"
 
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

# define the base models
level0 = list()
level0.append(('lr', linear_model.LogisticRegression()))
level0.append(('knn', neighbors.KNeighborsClassifier()))
level0.append(('cart', tree.DecisionTreeClassifier()))
level0.append(('svm', svm.SVC()))
level0.append(('bayes', naive_bayes.GaussianNB()))

# define meta learner model
level1 = linear_model.LogisticRegression()
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(Xtrain, Ytrain)

predictions = model.predict(Xtest)

acc = accuracy_score(Ytest, predictions)

acc #0.853

"Bagging"

clftree = tree.DecisionTreeClassifier()

bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(Xtrain, Ytrain)

# Evaluation on Testing Data
print(confusion_matrix(Ytest, bag_clf.predict(Xtest)))
accuracy_score(Ytest, bag_clf.predict(Xtest)) #0.853

print(accuracy_score(Ytrain, bag_clf.predict(Xtrain))) #1.0

"Ada Boost"

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(Xtrain, Ytrain)

# Evaluation on Testing Data
print(confusion_matrix(Ytest, ada_clf.predict(Xtest)))
print(accuracy_score(Ytest, ada_clf.predict(Xtest))) #0.853

# Evaluation on Training Data
accuracy_score(Ytrain, ada_clf.predict(Xtrain)) # 1.0

"Gradient Boosting"

# Hyperparameters
boost_clf = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)

boost_clf.fit(Xtrain, Ytrain)

print(confusion_matrix(Ytest, boost_clf.predict(Xtest)))
accuracy_score(Ytest, boost_clf.predict(Xtest)) #0.853

# Evaluation on Training Data
accuracy_score(Ytrain, boost_clf.predict(Xtrain)) #0.858

"XGBoosting"
import re

regex = re.compile(r"\[|\]|<", re.IGNORECASE)

Xtrain.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in Xtrain.columns.values]
Ytrain.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in Xtrain.columns.values]

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

xgb_clf.fit(Xtrain, Ytrain)

#Evaluation on Testing Data
print(confusion_matrix(Ytest, xgb_clf.predict(Xtest)))
accuracy_score(Ytest, xgb_clf.predict(Xtest)) #0.853

# Evaluation on Training Data
accuracy_score(Ytrain, xgb_clf.predict(Xtrain)) #0.858

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

from sklearn.model_selection import GridSearchCV
# Grid Search
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(Xtrain, Ytrain)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
print(accuracy_score(Ytest, cv_xg_clf.predict(Xtest)))
grid_search.best_params_

help(GridSearchCV)










