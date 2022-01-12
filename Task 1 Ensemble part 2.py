import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Ensemble Techniques/Diabeted_Ensemble.csv")    # loading the dataset
df.head()                        # viewing top 5 rows of dataset
df.info()

# Creating X and y for training
X = df.drop(' Class variable', axis = 1)
y = df[' Class variable']

# 20 % training dataset is considered for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# initializing sc object
sc = StandardScaler()  
# variables that needed to be transformed
var_transform = [' Plasma glucose concentration', ' Diastolic blood pressure', ' Body mass index', ' Diabetes pedigree function', ' Age (years)']
X_train[var_transform] = sc.fit_transform(X_train[var_transform])   # standardising training data 
X_test[var_transform] = sc.transform(X_test[var_transform])            # standardising test data
print(X_train.head())

# Building First Layer Estimators

KNC = KNeighborsClassifier()   # initialising KNeighbors Classifier
NB = GaussianNB()              # initialising Naive Bayes

# Training KNeighborsClassifier

model_kNeighborsClassifier = KNC.fit(X_train, y_train)   # fitting Training Set
pred_knc = model_kNeighborsClassifier.predict(X_test)   # Predicting on test dataset

#  Evaluation of KNeighborsClassifier

acc_knc = accuracy_score(y_test, pred_knc)  # evaluating accuracy score
print('accuracy score of KNeighbors Classifier is:', acc_knc * 100) # 68%

# Training Naive Bayes Classifier

model_NaiveBayes = NB.fit(X_train, y_train)
pred_nb = model_NaiveBayes.predict(X_test)

# Evaluation of Naive Bayes Classifier

acc_nb = accuracy_score(y_test, pred_nb)
print('Accuracy of Naive Bayes Classifier:', acc_nb * 100) #76%

# Implementing Stacking Classifier

lr = LogisticRegression()  # defining meta-classifier
clf_stack = StackingClassifier(classifiers =[KNC, NB], meta_classifier = lr, use_probas = True, use_features_in_secondary = True)

# Training Stacking Classifier

model_stack = clf_stack.fit(X_train, y_train)   # training of stacked model
pred_stack = model_stack.predict(X_test)       # predictions on test data using stacked model

# Evaluating Stacking Classifier

acc_stack = accuracy_score(y_test, pred_stack)  # evaluating accuracy
print('accuray score of Stacked model:', acc_stack * 100) # 72%

#Our both individual models scores an accuracy of 68%, 76% and our Stacked model got an accuracy of nearly 72%.
#By Combining two individual models we got a significant performance improvement.

"Voting"

from sklearn import datasets, linear_model, svm, neighbors, naive_bayes 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier - bydefault it is hard voting
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(X_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(X_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions)) # 68% of accuracy on hard voting

#################

# Soft Voting # 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(X_train, y_train)
learner_4.fit(X_train, y_train)
learner_5.fit(X_train, y_train)
learner_6.fit(X_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(X_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(X_test)
predictions_5 = learner_5.predict(X_test)
predictions_6 = learner_6.predict(X_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4)) # 66%
print('L5:', accuracy_score(y_test, predictions_5)) # 76%
print('L6:', accuracy_score(y_test, predictions_6)) # 70%

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions)) # 72% of accuracy on Soft voting




