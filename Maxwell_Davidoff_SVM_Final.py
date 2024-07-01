# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:08:39 2024

@author: mbdav
"""

### Path
import os
print(os.getcwd())

### Importing Data
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\InputData')
import Maxwell_Davidoff_HW1_Data as MDD
loans_mdf = MDD.loans_mdf
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\Code')


###Importing prior classification models and data used in classification models
import Maxwell_Davidoff_ClassificationHW2 as MDC
loans_mdf_train = MDC.loans_mdf_train
loans_mdf_valid = MDC.loans_mdf_valid


### Packages
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots , cm
import sklearn.model_selection as skm
from ISLP import confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import roc_curve, auc

from ISLP.models import (ModelSpec as MS,
                         summarize)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from prettytable import PrettyTable

###Standardize data 
scalen = StandardScaler()
loans_mdf_train = loans_mdf_train.reset_index(drop = True)
loans_mdf_valid = loans_mdf_valid.reset_index(drop = True)
print(loans_mdf_train)

loans_mdf_train['standing']=loans_mdf_train['loan_status'].replace({'Fully Paid':1,'Current':1,
                                                                       'Charged Off':-1})
loans_mdf_valid['standing']=loans_mdf_valid['loan_status'].replace({'Fully Paid':1,'Current':1,
                                                                       'Charged Off':-1})

loans_mdf_train['verification_status'] = pd.Categorical(loans_mdf_train['verification_status'])
loans_mdf_valid['verification_status'] = pd.Categorical(loans_mdf_valid['verification_status'])
ncs = loans_mdf_train.select_dtypes(include=['int64', 'float64']).columns
loans_mdf_train[ncs] = scalen.fit_transform(loans_mdf_train[ncs])
loans_mdf_valid[ncs] = scalen.fit_transform(loans_mdf_valid[ncs])

### create variables
terms = MS(loans_mdf_train.columns.drop(['standing','loan_status']))
X_train = terms.fit_transform(loans_mdf_train)
y_train = loans_mdf_train['standing']
X_valid = terms.transform(loans_mdf_valid)
y_valid = loans_mdf_valid['standing']
print(X_train.shape)

# Drop intercept as sklearn automatically has a intercept in SVM/SVC models
X_train = X_train.drop(X_train.columns[[0]], axis =1)
X_valid = X_valid.drop(X_valid.columns[[0]], axis = 1)
print(X_train)

### Create a smaller sample for computations
X_train = X_train.reset_index(drop = True)
print(X_train)
y_train = y_train.reset_index(drop = True)
X_subtrain = X_train.sample(n = 12000, random_state = 26)
print(X_subtrain)
idx = list(X_subtrain.index)
y_subtrain = y_train.iloc[idx]
print(y_subtrain)

# For radial
X_subtrainradial = X_train.sample(n=8000, random_state =26)
idxr = list(X_subtrainradial.index)
y_subtrainradial = y_train.iloc[idxr]

### cross validation grid search to find optimal cost term

svc = SVC(kernel='linear')

kfold = skm.KFold(5,
                  shuffle=True ,
                  random_state =10)

param_grid = {'C':[0.001 ,0.01 ,0.1 ,1 ,5 ,10 ,100]}
gssvc = skm.GridSearchCV(svc, param_grid = param_grid,cv = kfold,refit = True,scoring = 'accuracy', n_jobs =  -1)
gssvc.fit(X_subtrain, y_subtrain)
print('best C',gssvc.best_params_)
print('cross val score', gssvc.best_score_)
print('cv error scores',gssvc.cv_results_ [('mean_test_score')])

## Refitting the model
svc1 = SVC(C = 10, probability = True, kernel ='linear')
svc1.fit(X_subtrain, y_subtrain)
print('coefficient',svc1.coef_)


## Accuracy train and test
svc1trainpred = svc1.predict(X_subtrain)
svc1predvalid = svc1.predict(X_valid)
print(confusion_table(svc1trainpred,y_subtrain))
print('percentage correct in the training model', ((1272+10300)/12000)*100)
svc1trainerror = ((40+388)/12000)*100
print('training error rate', svc1trainerror)

print(confusion_table(svc1predvalid,y_valid))
print('percentage correct using valid data', ((776+6631)/7716)*100)
svc1validerror = ((42+267)/7716)*100
print('valid error rate', svc1validerror)

## ROC curve
ROCvalpred1 = svc1.predict_proba(X_valid)[:,1]
fpr, tpr, thresholds = roc_curve(y_valid, ROCvalpred1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVC Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

### Kernel Ploynomial
## Grid search for cost parameter/ degree of polynomial
svmpoly = SVC(kernel = 'poly')
param_grid = {
    'C': [0.001 ,0.01 ,0.1 ,1 ,5 ,10 ,100],
    'degree': [1, 2, 3, 4]
    }
gspolysvm = skm.GridSearchCV(svmpoly, param_grid = param_grid,cv = kfold,refit = True, scoring = 'accuracy', n_jobs = -1)
gspolysvm.fit(X_subtrain, y_subtrain)
print('best params', gspolysvm.best_params_)
print('cross val score', gspolysvm.best_score_)

## Refitting the model
svmpolyfinal = SVC(C = 10, probability = True, kernel ='poly',degree = 2)
svmpolyfinal.fit(X_subtrain, y_subtrain)

### Accuracy train and test
svmptrainpred = svmpolyfinal.predict(X_subtrain)
svmppredvalid = svmpolyfinal.predict(X_valid)
print(confusion_table(svmptrainpred,y_subtrain))
print('percentage correct in the training model', ((1333+10333)/12000)*100)
svpolytrainerror = ((7+327)/12000)*100
print('training error rate', svpolytrainerror)

print(confusion_table(svmppredvalid,y_valid))
print('percentage correct using valid data', ((780+6651)/7716)*100)
svpolyvaliderror = ((22+263)/7716)*100
print('valid error rate', svpolyvaliderror)

## ROC curve
ROCvalpred2 = svmpolyfinal.predict_proba(X_valid)[:,1]
fpr, tpr, thresholds = roc_curve(y_valid, ROCvalpred2)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM poly Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

### Kernel Radial
## Grid search for C and gamma
svmr = SVC(kernel = 'rbf')
param_grid = {
    'C': [.001, 1, 5, 10, 100],
    'gamma': [.001,0.5,1,2,]
    }
gsr = skm.GridSearchCV(svmr, param_grid = param_grid, cv = kfold, refit = True, scoring = 'accuracy', n_jobs = -1)
gsr.fit(X_subtrainradial, y_subtrainradial)
print('best params', gsr.best_params_)
print('cross val score', gsr.best_score_)

## Refitting the model
svmradial = SVC(C = 100, probability = True, kernel ='rbf',gamma = .001)
svmradial.fit(X_subtrainradial, y_subtrainradial)

### Accuracy train and test
svmrtrainpred = svmradial.predict(X_subtrainradial)
svmrpredvalid = svmradial.predict(X_valid)
print(confusion_table(svmrtrainpred,y_subtrainradial))
print('percentage correct in the training model', ((813+6882)/8000)*100)
svmradialtrainerror = ((7+298)/8000)*100
print('training error rate', svmradialtrainerror)

print(confusion_table(svmrpredvalid,y_valid))
print('percentage correct using valid data', ((725+6658)/7716)*100)
svmradialvaliderror = ((15+318)/7716)*100
print('valid error rate', svmradialvaliderror)

## ROC curve
ROCvalpred3 = svmradial.predict_proba(X_valid)[:,1]
fpr, tpr, thresholds = roc_curve(y_valid, ROCvalpred3)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM radial Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

### error tables for train and test 
logittrainerror = MDC.logittrainerro
logitvaliderror = MDC.logitvaliderror
ldatrainerror = MDC.ldatrainerror
ldavaliderror = MDC.ldavaliderror
qdavaliderror = MDC.qdavaliderror
qdatrainerror  = MDC.qdatrainerror
naivetrainerror = MDC.naivetrainerror
naivevaliderror = MDC.naivevaliderror
knn1trainerror = MDC.knn1trainerror
knn1validerror = MDC.knn1validerror
knncv6trainerror= MDC.knncv6trainerror
knncv6validerror = MDC.knncv6validerror


train_table = PrettyTable(['final_logistic', 'LDA', 'QDA', 'Naive', 'KNN1', 'KNN6(CV)','SVC linear', 'SVM poly', 'SVM Radial'])
train_table.add_row([logittrainerror,ldatrainerror,qdatrainerror,naivetrainerror,knn1trainerror,
                     knncv6trainerror, svc1trainerror,svpolytrainerror,svmradialtrainerror])
print(train_table)

valid_table = PrettyTable(['final_logistic', 'LDA', 'QDA', 'Naive', 'KNN1', 'KNN6(CV)', 'SVC linear', 'SVM poly', 'SVM Radial' ])
valid_table.add_row([logitvaliderror,ldavaliderror,qdavaliderror,naivevaliderror,knn1validerror,
                     knncv6validerror,svc1validerror,svpolyvaliderror,svmradialvaliderror ])
print(valid_table)