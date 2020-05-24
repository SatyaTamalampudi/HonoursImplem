import numpy as np
import os
import cv2
from numpy import load

# # Importing Models
#export the models from the sklearn library
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# or we can use a heatmap from the seaborn library
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.getcwd()) 
path = "C:\\Users\\1609653\\AnomalyDetection\\FeaturesData"
os.chdir(path)

train_data = load('training_data.npy',allow_pickle=True)
test_data = load('testing_data.npy',allow_pickle=True)

train_data1 = cv2.normalize(train_data, 0, 1, norm_type=cv2.NORM_MINMAX)
test_data1 = cv2.normalize(test_data,0,1,norm_type=cv2.NORM_MINMAX)

print(train_data1.shape)
print(test_data1.shape)


labels_X = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
labels_Y = [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
X_train = train_data1
Y_train = labels_X
X_test  = test_data1
Y_test = labels_Y




# # SVM
model = SVC(C = 1.0)
print(model)
#svclassifier = SVC(kernel='sigmoid')
model.fit(X_train, Y_train)
pred1 = model.predict(X_test)
print(pred1)

# Confusion Matrix
cm1 = confusion_matrix(Y_test, pred1)
print(cm1)
print(np.sum(pred1 == Y_test) / float(len(Y_test)))

df_cm = pd.DataFrame(cm1, range(2), range(2))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="YlGnBu", annot=True, annot_kws={"size": 16},fmt="d")# font size
plt.title("SVM")

#Classification report
cr1 = classification_report(Y_test, pred1)
print(cr1)


# # Logistic.Regression

models = []
models.append(('LR', LogisticRegression()))
models.append(('NB', BernoulliNB()))
#models.append(('SVM', SVC(kernel='rbf', gamma=0.7, C=1.0)))
print(type(models[1][1]))


classifier = models[0][1]
print(classifier)
classifier.fit(X_train, Y_train)
pred2 = classifier.predict(X_test)

print("Classifier %s [Accuracy]:" % (models[0][1]))
print(np.sum(pred2 == Y_test) / float(len(Y_test)))


cm2 = confusion_matrix(Y_test, pred2)
print(cm2)
print(np.sum(pred2 == Y_test) / float(len(Y_test)))


df_cm = pd.DataFrame(cm, range(2), range(2))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sn.set(font_scale=1.4)#for label size
#sn.title("L.R")
sn.heatmap(df_cm, cmap="YlGnBu", annot=True, annot_kws={"size": 16},fmt="d")# font size
plt.title("L.R")
#Classification report
cr2 = classification_report(Y_test, pred2)
print(cr2)


# # Naive Bayes

classifier = GaussianNB()
print(classifier)
classifier.fit(X_train, Y_train)
pred3 = classifier.predict(X_test)

print("Classifier %s [Accuracy]:" % GaussianNB())
print(np.sum(pred3 == Y_test) / float(len(Y_test)))

cm3 = confusion_matrix(Y_test, pred3)
print(cm3)
print(np.sum(pred3 == Y_test) / float(len(Y_test)))

df_cm = pd.DataFrame(cm3, range(2), range(2))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="YlGnBu", annot=True, annot_kws={"size": 16},fmt="d")# font size
plt.title("NB")

cr3 = classification_report(Y_test, pred3)
print(cr3)


#models = ["SVM","L.R","N.B"]
#results = [52.3,71.4,52.3]

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

