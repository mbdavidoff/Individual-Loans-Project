# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:50:22 2024

@author: mbdav
"""

### Path
import os
print(os.getcwd())

### Importing Data
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\InputData')
import Maxwell_Davidoff_HW1_Data as MDD
loans_mdf_train = MDD.loans_mdf_train
loans_mdf = MDD.loans_mdf
loans_mdf_valid = MDD.loans_mdf_valid
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\Code')
loans_mdf_train.head()


### Importing packages needed
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from matplotlib.pyplot import subplots
from ISLP.models import (ModelSpec as MS,
                         summarize)
from ISLP.models import contrast
from ISLP import confusion_table
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA ,
     QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF

from ISLP.models import \
    (Stepwise ,
     sklearn_selected ,
     sklearn_selection_path)
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from statsmodels.graphics.gofplots import ProbPlot
from prettytable import PrettyTable
from sklearn.model_selection import  cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc


### Creation of a binary outcome variable where 1 = good standing and 0 = deliquent
loans_mdf_train['standing']=loans_mdf_train['loan_status'].replace({'Fully Paid':1,'Current':1,
                                                                       'Charged Off':0})
loans_mdf_valid['standing']=loans_mdf_valid['loan_status'].replace({'Fully Paid':1,'Current':1,
                                                                       'Charged Off':0})

### standing value counts
print(loans_mdf_train['standing'].value_counts())
sns.countplot(loans_mdf_train['standing'])


### Creating X and Y variables for logit train and test

terms = MS(loans_mdf_train.columns.drop(['standing','loan_status']))
X_train = terms.fit_transform(loans_mdf_train)
y_train = loans_mdf_train['standing']
X_valid = terms.transform(loans_mdf_valid)
y_valid = loans_mdf_valid['standing']
print(X_train.shape)


### Negative AIC for forward selection

def negAIC(estimator, X, Y):
    "Negative AIC"
    Y = np.array(Y)
    n, p = X.shape
    Yhat = estimator.predict(X)
    LL =  np.sum(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat))
    print(-1*((-2)*LL + 2*(p/n)))
    return -1*((-2)*LL + 2*(p))



strategy = Stepwise.first_peak(terms,
                               direction='forward',
                               max_terms=len(terms.terms))
m_aic = sklearn_selected(sm.GLM,
                               strategy,
                               scoring = negAIC)
m_aic.fit(loans_mdf_train, y_train)

print(m_aic.selected_state_)

### Feature Selection
X_train_logit = X_train.drop(X_train.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                              16,17, 18, 19, 20,35,36,37,41,43,
                                              44,45,46,47,48,49,50,51,52]], axis =1)
X_valid_logit = X_valid.drop(X_valid.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                              16,17, 18, 19, 20,35,36,37,41,43,
                                              44,45,46,47,48,49,50,51,52]], axis =1)
print(X_train_logit)

### FIrst Model
logm2 = sm.GLM(y_train, X_train_logit, family = sm.families.Binomial())


results2 = logm2.fit()
print(results2.summary())

### VIF
vals = [VIF(X_train_logit, i)
        for i in range(1, X_train_logit.shape [1])]
vif = pd.DataFrame ({'vif':vals},
                    index=X_train_logit.columns [1:])
print(vif)

### Second Logit
X_train_logit = X_train_logit.drop(X_train_logit.columns[[17]], axis =1)
X_valid_logit = X_valid_logit.drop(X_valid_logit.columns[[17]], axis =1)

logm3 = sm.GLM(y_train, X_train_logit, family = sm.families.Binomial())

results3 = logm3.fit()
print(results3.summary())

###Second model plot
yh = results3.predict(X_train_logit)
sns.regplot(x = yh, y = y_train, color="blue",  logistic= True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()
### VIF for second logit model
vals = [VIF(X_train_logit, i)
        for i in range(1, X_train_logit.shape [1])]
vif = pd.DataFrame ({'vif':vals},
                    index=X_train_logit.columns [1:])
print(vif)


### Confusion Matrix logit for accuracy and test error rate. trained model and valid data
ptrain = results3.predict(X_train_logit)
print(ptrain)

labs = np.array([0]*30713)
labs[ptrain>0.5] = 1
print(confusion_table(labs, loans_mdf_train.standing))
print('percentage correct in the training model', ((3129+26176)/30713)*100)
print('training error rate', ((1139+269)/30713)*100)
logittrainerro = ((1139+269)/30713)*100

pvalid = results3.predict(X_valid_logit)
labst = np.array([0]*7716)
labst[pvalid>0.5] = 1
print(confusion_table(labst, loans_mdf_valid.standing))
print('percentage correct using the model for valid data', ((740+6608)/7716)*100)
print('valid error rate', ((303+65)/7716)*100)
logitvaliderror = ((303+65)/7716)*100

### ROC curve
fpr, tpr, thresholds = roc_curve(y_valid, pvalid)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

### LDA Model
lda = LDA(store_covariance=True)
loans_mdfa = loans_mdf_train.drop('standing', axis = 'columns')
loans_mdfa = loans_mdfa.drop('loan_status', axis = 'columns')
loans_mdfa['emp_length']=pd.Categorical(loans_mdfa['emp_length'].cat.codes)
loans_mdfa['home_ownership']=pd.Categorical(loans_mdfa['home_ownership'].cat.codes)
loans_mdfa['purpose']=pd.Categorical(loans_mdfa['purpose'].cat.codes)
loans_mdfa['region']=pd.Categorical(loans_mdfa['region'].cat.codes)
loans_mdfa['term']=pd.Categorical(loans_mdfa['term'].cat.codes)
loans_mdfa['grade']=pd.Categorical(loans_mdfa['grade'].cat.codes)

loans_mdfb = loans_mdf_valid.drop('standing', axis = 'columns')
loans_mdfb = loans_mdfb.drop('loan_status', axis = 'columns')
loans_mdfb['emp_length']=pd.Categorical(loans_mdfb['emp_length'].cat.codes)
loans_mdfb['home_ownership']=pd.Categorical(loans_mdfb['home_ownership'].cat.codes)
loans_mdfb['purpose']=pd.Categorical(loans_mdfb['purpose'].cat.codes)
loans_mdfb['region']=pd.Categorical(loans_mdfb['region'].cat.codes)
loans_mdfb['term']=pd.Categorical(loans_mdfb['term'].cat.codes)
loans_mdfb['grade']=pd.Categorical(loans_mdfb['grade'].cat.codes)

### Feature Selection for LDA

sfs = SFS(lda,
          k_features='best',  
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(loans_mdfa, y_train)
selected_features = list(sfs.k_feature_names_)
print(selected_features)

X_train_lda = loans_mdfa.drop('total_acc', axis = 'columns')
X_train_lda = X_train_lda.drop('purpose', axis = 'columns')
X_valid_lda = loans_mdfb.drop('total_acc', axis = 'columns')
X_valid_lda = X_valid_lda.drop('purpose', axis = 'columns')
### Fitting the MODEL
lda.fit(X_train_lda, y_train)
print(lda.classes_)
print(lda.means_)
print(lda.priors_)
print(lda.scalings_)

### Confusion matrices for LDA accuracy and test error rate. Trained and valid data
ldap_train = lda.predict_proba(X_train_lda)[:,1]
labs2 = np.array([0]*30713)
labs2[ldap_train>0.5] = 1
print(confusion_table(labs2, loans_mdf_train.standing))
print('percentage correct in the training model', ((2281+26371)/30713)*100)
print('training error rate', ((74+1987)/30713)*100)
ldatrainerror = ((74+1987)/30713)*100

ldap_valid = lda.predict_proba(X_valid_lda)[:,1]
labst2 = np.array([0]*7716)
labst2[ldap_valid>0.5] = 1
print(confusion_table(labst2, loans_mdf_valid.standing))
print('percentage correct using the model for valid data', ((541+6660)/7716)*100)
print('valid error rate', ((502+13)/7716)*100)
ldavaliderror = ((502+13)/7716)*100

### ROC curve
fpr, tpr, thresholds = roc_curve(y_valid, ldap_valid)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LDA Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

### QDA Model
### Feature Selection
qda = QDA(store_covariance=True)

### Fitting the model
sfs = SFS(qda,
          k_features='best',  
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(loans_mdfa, y_train)
selected_features = list(sfs.k_feature_names_)
print(selected_features)

X_train_qda = loans_mdfa.drop('grade', axis = 'columns')
X_train_qda = X_train_qda.drop('verification_status', axis = 'columns')
X_train_qda = X_train_qda.drop('annual_income', axis = 'columns')
X_train_qda = X_train_qda.drop('loan_amount', axis = 'columns')
X_valid_qda = loans_mdfb.drop('grade', axis = 'columns')
X_valid_qda = X_valid_qda.drop('verification_status', axis = 'columns')
X_valid_qda = X_valid_qda.drop('annual_income', axis = 'columns')
X_valid_qda = X_valid_qda.drop('loan_amount', axis = 'columns')


qda.fit(X_train_qda , y_train)
print(qda.classes_)
print(qda.means_)
print(qda.priors_)
print(qda.scalings_)

### Confusion matrices for QDA accuracy and test error rate. Trained and valid data
qdap_train = qda.predict_proba(X_train_qda)[:,1]
labs3 = np.array([0]*30713)
labs3[qdap_train>0.5] = 1
print(confusion_table(labs3, y_train))
print('percentage correct in the training model', ((2694+25927)/30713)*100)
print('training error rate', ((1574+518)/30713)*100)
qdatrainerror = ((1574+518)/30713)*100

qdap_valid = qda.predict_proba(X_valid_qda)[:,1]
labst3 = np.array([0]*7716)
labst3[qdap_valid>0.5] = 1
print(confusion_table(labst3, loans_mdf_valid.standing))
print('percentage correct using the model for valid data', ((647+6544)/7716)*100)
print('valid error rate', ((396+129)/7716)*100)
qdavaliderror = ((396+129)/7716)*100


### ROC curve
fpr, tpr, thresholds = roc_curve(y_valid, qdap_valid)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('QDA Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
### Naive Bayes Model

naive = GaussianNB()

## Feature Selection
sfs = SFS(naive,
          k_features='best',  
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(loans_mdfa, y_train)
selected_features = list(sfs.k_feature_names_)
print(selected_features)

X_train_nb = loans_mdfa.drop('dti', axis = 'columns')
X_train_nb = X_train_nb.drop('int_rate', axis = 'columns')
X_train_nb = X_train_nb.drop('total_acc', axis = 'columns')
X_valid_nb = loans_mdfb.drop('dti', axis = 'columns')
X_valid_nb = X_valid_nb.drop('int_rate', axis = 'columns')
X_valid_nb = X_valid_nb.drop('total_acc', axis = 'columns')

naive.fit(X_train_nb, y_train)
print(naive.classes_)
print(naive.theta_)
print(naive.class_prior_)
print(naive.var_)
print(X_train_nb[y_train == 1].var(ddof =0))

### Confusion matrices for Naive Bayes model accuracy and test error rate. Trained and valid data
naivep_train = naive.predict_proba(X_train_nb)[:,1]
labs4 = np.array([0]*30713)
labs4[naivep_train>0.5] = 1
print(confusion_table(labs4, loans_mdf_train.standing))
print('percentage correct in the training model', ((165+26414)/30713)*100)
print('training error rate', ((4103+31)/30713)*100)
naivetrainerror = ((4103+31)/30713)*100

naivep_valid = naive.predict_proba(X_valid_nb)[:,1]
labst4 = np.array([0]*7716)
labst4[naivep_valid>0.5] = 1
print(confusion_table(labst4, loans_mdf_valid.standing))
print('percentage correct using the model for valid data', ((39+6668)/7716)*100)
print('valid error rate', ((1004+5)/7716)*100)
naivevaliderror = ((1004+5)/7716)*100

### ROC curve
fpr, tpr, thresholds = roc_curve(loans_mdf_valid.standing, naivep_valid)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

### K Nearest Neighbors model, K =1 
knn1 = KNeighborsClassifier(n_neighbors =1)
## Feature Selection 
sfs = SFS(knn1,
          k_features='best',  
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(loans_mdfa, y_train)
selected_features = list(sfs.k_feature_names_)
print(selected_features)

X_train_knn1 = loans_mdfa.drop('annual_income', axis = 'columns')
X_valid_knn1 = loans_mdfb.drop('annual_income', axis = 'columns')
# fitting the model/confusion matrix/test error
knn1.fit(X_train_knn1 , y_train)
knn1p_train = knn1.predict_proba(X_train_knn1)[:,1]
labs5 = np.array([0]*30713)
labs5[knn1p_train>0.5] = 1
print(confusion_table(labs5, loans_mdf_train.standing))
print('percentage correct in the training model', ((4268+26445)/30713)*100)
print('training error rate', ((0)/30713)*100)
knn1trainerror = ((0)/30713)*100

knn1p_valid = knn1.predict_proba(X_valid_knn1)[:,1]
labst5 = np.array([0]*7716)
labst5[knn1p_valid>0.5] = 1
print(confusion_table(labst5, loans_mdf_valid.standing))
print('percentage correct using the model for valid data', ((818+6511)/7716)*100)
print('valid error rate', ((225+162)/7716)*100)
knn1validerror = ((225+162)/7716)*100

### ROC curve
fpr, tpr, thresholds = roc_curve(loans_mdf_valid.standing, knn1p_valid)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN1 Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
### K Nearest Neighbors model, grid search to find K using same predictors as above

## Combined feature slection and grid search for k

knncv = KNeighborsClassifier()

K = 5
kfold = skm.KFold(K,
                  random_state =0,
                  shuffle=True)



param_grid = { 'n_neighbors': range(1, 20)}

knn_gs = skm.GridSearchCV(knncv, param_grid, cv = kfold, scoring='accuracy')

knn_gs.fit(X_train_knn1, y_train)
print('bestk value',knn_gs.best_params_)

### Test errors and accuracy score

knn6 = KNeighborsClassifier(n_neighbors =5)
knn6.fit(X_train_knn1, y_train)
knn6p_train = knn6.predict_proba(X_train_knn1)[:,1]
labs6 = np.array([0]*30713)
labs6[knn6p_train>0.5] = 1
print(confusion_table(labs6, loans_mdf_train.standing))
print('percentage correct in the training model', ((3524+26222)/30713)*100)
print('training error rate', ((223+744)/30713)*100)
knncv6trainerror = ((223+744)/30713)*100

knn6p_valid = knn6.predict_proba(X_valid_knn1)[:,1]
labst6 = np.array([0]*7716)
labst6[knn6p_valid>0.5] = 1
print(confusion_table(labst6, loans_mdf_valid.standing))
print('percentage correct using the model for valid data', ((808+6590)/7716)*100)
print('valid error rate', ((235+83)/7716)*100)
knncv6validerror = ((235+83)/7716)*100

### ROC curve
fpr, tpr, thresholds = roc_curve(loans_mdf_valid.standing, knn6p_valid)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN6 Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
### Results tablle
valid_table = PrettyTable(['final_logistic', 'LDA', 'QDA', 'Naive', 'KNN1', 'KNN6(CV)'])
valid_table.add_row([logitvaliderror,ldavaliderror,qdavaliderror,naivevaliderror,knn1validerror,knncv6validerror])
print(valid_table)
