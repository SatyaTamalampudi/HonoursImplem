import numpy as np
import os
from numpy import load

#evaltion
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# or we can use a heatmap from the seaborn library
import seaborn as sn
import matplotlib.pyplot as plt
#export the models from the sklearn library
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
get_ipython().run_line_magic('matplotlib', 'inline')


print(os.getcwd()) 
path = "C:\\Users\\1609653\\AnomalyDetection\\FeaturesData"
os.chdir(path)


train_data = load('training_data.npy',allow_pickle=True)
test_data = load('testing_data.npy',allow_pickle=True)
print(train_data.shape)
print(test_data.shape)

labels_X = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
labels_Y = [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
X_train = train_data
Y_train = labels_X
X_test  = test_data
Y_test = labels_Y


accuracies = []

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf', gamma=0.7, C=1.0)))

#print(type(models[2][1]))


classifier = models[5][1]
print(classifier)
classifier.fit(X_train, Y_train)
pred2 = classifier.predict(X_test)

print("Classifier %s [Accuracy]:" % (models[5][0]))
accuracy = np.sum(pred2 == Y_test) / float(len(Y_test))
accuracies.append(accuracy)
print(accuracy)


cm = confusion_matrix(Y_test, pred2)
print(cm)
print(np.sum(pred2 == Y_test) / float(len(Y_test)))


# Setting up Gridsearch for SVM to find best parameters
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, Y_train) 
# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
grid_predictions = grid.predict(X_test) 


# Setting gridsearch for logistic regression
# Create logistic regression
logistic =LogisticRegression()
print(logistic)

# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = clf.fit(X_train, Y_train)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
# Predict target vector
prediction = best_model.predict(X_test)



#Plotting model accuracies
models = ["SVM","L.R","N.B"]
results = [57.1,71.4,52.3]


objects = models
y_pos = np.arange(len(objects))
performance = results
 
plot = plt.bar(y_pos, performance, align='center', alpha=0.5)
plot[0].set_color('blue')
plot[1].set_color('b')
plot[2].set_color('b')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Models')
 
plt.show()


#Plotting ROC vurve

fpr1, tpr1, threshold1 = roc_curve(Y_test, pred1) 
fpr2, tpr2, threshold2 = roc_curve(Y_test, pred2) 
fpr3, tpr3, threshold3 = roc_curve(Y_test, pred3)
 
# This is the AUC
auc1 = auc(fpr1, tpr1)
print(auc1)
auc2 = auc(fpr2, tpr2)
print(auc2)
auc3 = auc(fpr3, tpr3)
print(auc2)


# This is the ROC curve
plt.plot(fpr1,tpr1, label='SVM (area = %0.2f)' % (auc1))
plt.plot(fpr2,tpr2, label='L.R (area = %0.2f)' % (auc2))
plt.plot(fpr3,tpr3, label='N.B (area = %0.2f)' % (auc3))

plt.plot([0, 1], [0, 1], 'k--') # diagonal

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show() 