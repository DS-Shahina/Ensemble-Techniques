# Import the required libraries
#Voting talks about different algorithm, they don't need to have same algorithm
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (inbuild dataset)
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

# Split the train and test samples
# this is sequential partioning
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples] # [:-test_sample] -[:-1] removing that last character , ignore 100 and take remaining dataset
x_test, y_test = x[-test_samples:], y[-test_samples:] # [-4:] - (-4 to till the end of data) - 100 records in the test sample (-100,-99,-98----)

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier - bydefault it is hard voting
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(x_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(x_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions)) # 90% of accuracy on hard voting

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
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4)) # 96%
print('L5:', accuracy_score(y_test, predictions_5)) # 88%
print('L6:', accuracy_score(y_test, predictions_6)) # 94%

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions)) # 94% of accuracy on hard voting

