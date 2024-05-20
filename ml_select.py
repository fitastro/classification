#!/usr/bin/python


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#-------------------------------------------
# Importing the dataset
dataset = pd.read_csv('/Users/igezer/ALLWISE/W3/classification/w3_final_training.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#------------------------------------------
# Splitting the dataset into the Training set and Test set (Kfold)
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=42, shuffle=True)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test =  X.loc[train_index], X.loc[test_index]
    y_train, y_test =  y[train_index], y[test_index]



#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
#------------------------------------------
#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn import metrics 
#------------------------------------------
# Training the Logistic Regression model on the Training set
print("Logistic Regression.......")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("LR model accuracy:", metrics.accuracy_score(y_test, y_pred))
print("-----------------------------")
import joblib
joblib.dump(lr, 'model_lr.pkl')

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#accuracy_score(y_test, y_pred)

#------------------------------------------
# # Training the K-NN model on the Training set
# from sklearn.neighbors import KNeighborsClassifier
# print("KNeighbors Classifier.............")
# knn = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print("KNeighbors accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("-----------------------------")
# import joblib
# joblib.dump(knn, 'model_knn.pkl')

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#accuracy_score(y_test, y_pred)

#------------------------------------------
# Training the  Kernel SVM model on the Training set
from sklearn.svm import SVC
print("SVC.............")
svc = SVC(kernel = 'rbf', random_state = 0, gamma='auto',probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("SVC accuracy:", metrics.accuracy_score(y_test, y_pred))
print("-----------------------------")
import joblib
joblib.dump(svc, 'model_svc.pkl')

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#accuracy_score(y_test, y_pred)


#------------------------------------------
# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
print("NaiveBayes Gaussian .............")
nbg = GaussianNB()
nbg.fit(X_train, y_train)
y_pred = nbg.predict(X_test)
print("NaiveBayes Gaussian accuracy:", metrics.accuracy_score(y_test, y_pred))
print("-----------------------------")
import joblib
joblib.dump(nbg, 'model_nbg.pkl')

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#accuracy_score(y_test, y_pred)

##------------------------------------------
## Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
print("DecisionTree Classifier.............")
dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("DecisionTree Classifier accuracy:", metrics.accuracy_score(y_test, y_pred))
print("-----------------------------")
import joblib
joblib.dump(dtc, 'model_dtc.pkl')

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#accuracy_score(y_test, y_pred)


#------------------------------------------
# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
print("Random Forest Classifier........")
rfc = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state=0, class_weight="balanced")
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("RFC model accuracy:", metrics.accuracy_score(y_test, y_pred))
print("RFC model score:", rfc.score(X_test, y_test))
print("-----------------------------")
import joblib
joblib.dump(rfc, 'model_rfc.pkl')



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)
print(cm)
accuracy_score(y_test, y_pred)


import seaborn as sns
#cm_perc = cm/np.sum(cm)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float)
ax= plt.subplot()
sns.heatmap(cm_perc, annot=True, fmt='.2%', cbar=False, cmap= 'Pastel1_r',ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['real', 'fake'])
ax.yaxis.set_ticklabels(['real', 'fake'])

plt.show()


#------------------------------------------





#-----------------------------------------
#classifier and report
from sklearn.metrics import classification_report
print("------Classification report------")
print
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.show()
#-----------------------------------------

