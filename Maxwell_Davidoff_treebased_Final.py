# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:55:32 2024

@author: mbdav
"""

### Path
import os
print(os.getcwd())

### Importing Data
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\InputData')
import Maxwell_Davidoff_HW1_Data as MDD
loans_mdf = MDD.loans_mdf
loans_mdf_train = MDD.loans_mdf_train
loans_mdf_valid = MDD.loans_mdf_valid
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\Code')
loans_mdf_train.head()


### Packages

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
import sklearn.model_selection as skm
from ISLP import load_data , confusion_table
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC ,
                          DecisionTreeRegressor as DTR ,
                          plot_tree ,
                          export_text)
from sklearn.metrics import (accuracy_score ,
log_loss)
from sklearn.ensemble import \
(RandomForestRegressor as RF ,
GradientBoostingRegressor as GBR)
from ISLP.bart import BART
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable

### Creating X and Y training and test

terms = MS(loans_mdf_train.columns.drop('int_rate')).fit(loans_mdf_train)
X_train = terms.transform(loans_mdf_train)
X_train = X_train.drop(X_train.columns[[0]], axis =1)
fn = list(X_train.columns)
print(X_train)
X_train = np.array(X_train)
y_train = loans_mdf_train['int_rate']
X_valid = terms.transform(loans_mdf_valid)
X_valid = X_valid.drop(X_valid.columns[[0]], axis =1)
y_valid = loans_mdf_valid['int_rate']


### Regression Tree:
## finding the depth through grid search
kfold = skm.KFold(5,
                  shuffle=True ,
                  random_state =10)

rtreed = DTR()
param_grid = {'max_depth': np.arange(1, 8)}

gsd = GridSearchCV(rtreed, param_grid, cv=kfold, scoring='neg_mean_squared_error')
gsd.fit(X_train, y_train)

optdepth = gsd.best_params_['max_depth']
print('Best Tree Depth',optdepth)

## refitted model
rt = DTR(max_depth=optdepth)
depth_results = rt.fit(X_train, y_train)

## Decision Plot
ax = subplots(figsize =(50 ,15))[1]
plot_tree(rt ,
          feature_names=fn,
          ax=ax)
plt.show()

## Test and train mse and decision path after depth selection
trainds = np.mean((y_train - rt.predict(X_train))**2)
print('rtree train MSE',trainds)
validds = np.mean((y_valid - rt.predict(X_valid))**2)
print('rtree valid MSE', validds)

## Prunning based on cost complexity alpha
rta = DTR(max_depth = optdepth)
ccp_path = rta.cost_complexity_pruning_path(X_train , y_train)
kfold = skm.KFold(5,
                  shuffle=True ,
                  random_state =10)
gsm = skm.GridSearchCV(rta ,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True ,
                        cv=kfold ,
                        scoring='neg_mean_squared_error')



prunedresults = gsm.fit(X_train, y_train)

alpha = gsm.best_estimator_
print(alpha)

## refitted model
rt2 = DTR(max_depth=optdepth, ccp_alpha = 0.00027885604792085485)
decision_results = rt2.fit(X_train, y_train)

## Decision Plot
ax = subplots(figsize =(50 ,15))[1]
plot_tree(rt2,
          feature_names=fn,
          ax=ax)
plt.show()

## Test train MSE after prunning and decision tree
final_decision_tree_train = np.mean((y_train - rt2.predict(X_train))**2)
print('train MSE for decision tree', final_decision_tree_train)
final_decision_tree_valid = np.mean((y_valid - rt2.predict(X_valid))**2)
print('valid MSE for decision tree' , final_decision_tree_valid)

### Small T for interpretation sake
t = DTR(max_depth=3)
depth_results = t.fit(X_train, y_train)

## Decision Plot
ax = subplots(figsize =(50 ,15))[1]
plot_tree(t ,
          feature_names=fn,
          ax=ax);
plt.show()
### Bagging on all features
bagm = RF(max_features=X_train.shape [1], random_state =0)
bagged_results = bagm.fit(X_train , y_train)

## Feature Importance bagging
feature_importance = pd.DataFrame(
    {'importance ':bagged_results.feature_importances_},
    index=fn)
print(feature_importance.sort_values(by='importance ', ascending=False))
## Train and test MSE for bagging
bagpredval = bagm.predict(X_valid)
bagpredtrain = bagm.predict(X_train)
bagged_train = np.mean((y_train - bagpredtrain)**2)
bagged_valid = np.mean((y_valid - bagpredval)**2)
print('train MSE for bagging model', bagged_train)
print('valid MSE for bagging model', bagged_valid)

## Plot of fit for bagging valid
ax = subplots(figsize =(8 ,8))[1]
ax.scatter(bagpredval, y_valid)
ax.set_xlabel('yhat')
ax.set_ylabel('y')
plt.show()

### Random Forest
## Performing grid search to find max features

param_grid = {'max_features': ['sqrt', 'log2',0.5, 0.8]}
rf = RF(random_state = 30)
grids_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
rfg_results = grids_rf.fit(X_train, y_train)

feat = grids_rf.best_params_['max_features']
print("Best Parameters:", grids_rf.best_params_)
print("Best Cross Validated MSE:", grids_rf.best_score_)

rf2 = RF(max_features = feat, random_state = 30)
rf_results = rf2.fit(X_train, y_train)

## Feature Importance rf
feature_importance = pd.DataFrame(
    {'importance ':rf_results.feature_importances_},
    index=fn)
print(feature_importance.sort_values(by='importance ', ascending=False))

## Train and test MSE for random forest
rfpredval = rf2.predict(X_valid)
rfpredtrain = rf2.predict(X_train)
random_forest_train = np.mean((y_train - rfpredtrain)**2)
random_forest_valid = np.mean((y_valid - rfpredval)**2)
print('train MSE for random forest', random_forest_train)
print('valid MSE for random forest', random_forest_valid)


## Plot of fit for random forest valid
ax = subplots(figsize =(8 ,8))[1]
ax.scatter(rfpredval, y_valid)
ax.set_xlabel('yhat')
ax.set_ylabel('y')
plt.show()


### Boosting Model

## Grid search for optimal tuning parameter, depth of each tree, and number of trees
gb = GBR(max_depth = 3, n_estimators = 100, random_state =5)
param_grid = {'learning_rate': [0.1, 0.01, 0.001, 0.0001]}
gs3 = GridSearchCV(estimator=gb, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')


gs3.fit(X_train, y_train)
print('best parameters', gs3.best_params_)
print('cross validation MSE', gs3.best_score_)

## Fitting a boosting model based on the grid search
boostm = GBR(n_estimators =100, learning_rate = .1, max_depth = 3, random_state = 5)
boostm.fit(X_train, y_train)

## Train and test MSE for Boosting
boostpredval = boostm.predict(X_valid)
boostpredtrain = boostm.predict(X_train)
boosting_train = np.mean((y_train - boostpredtrain)**2)
boosting_valid = np.mean((y_valid - boostpredval)**2)
print('train MSE for  boosting', boosting_train)
print('valid MSE for boosting', boosting_valid)

## Visualizing how the test error decreases
teste = np.zeros_like(boostm.train_score_)
for idx , y_ in enumerate(boostm.staged_predict(X_valid)):
    teste[idx] = np.mean(( y_valid - y_)**2)
    
plot_idx = np.arange(boostm.train_score_.shape [0])
ax = subplots(figsize =(8 ,8))[1]
ax.plot(plot_idx ,
        boostm.train_score_ ,
        'b',
        label='Training ')
ax.plot(plot_idx ,
        teste ,
        'r',
        label='Test ')
ax.legend ();
plt.show()
## Plot of fit for boosting
ax = subplots(figsize =(8 ,8))[1]
ax.scatter(boostpredval, y_valid)
ax.set_xlabel('yhat')
ax.set_ylabel('y')
plt.show()
    
## BART Model
bartm = BART(random_state =0, burnin=100, ndraw =200, n_jobs = -1, num_trees = 20)
bartm.fit(X_train , y_train)
X_valid2 = np.array(X_valid)

## Train and Test for BART
bartpredval = bartm.predict(X_valid2)
bartpredtrain = bartm.predict(X_train)
bart_train = np.mean((y_train - bartpredtrain)**2)
bart_valid = np.mean((y_valid - bartpredval)**2)
print('train MSE for BART', bart_train)
print('valid MSE for BART', bart_valid)

## Plot of fit for bart
ax = subplots(figsize =(8 ,8))[1]
ax.scatter(bartpredval, y_valid)
ax.set_xlabel('yhat')
ax.set_ylabel('y')
plt.show()
### MSE train/valid tables for all continuous int_rate models
OLSmsetrain = 0.9857499067992311
OLSmsevalid =1.0291283599477434
Ridgemsetrain = 1.0116281220871903
Ridgemsevalid = 1.02174837549837
PolyOLSmsetrain = 0.9838383752054192
PolyOLSmsevalid = 1.0279530950257345
GAMmsetrain = 0.9899473219007321
GAMmsevalid = 1.0051598626131026
train_table = PrettyTable(['OLS', 'Ridge', 'PolyOLS', 'GAM', 'Decision Tree(7)', 
                           'Bagging', 'Random Forest', 'Boosting', 'BART'])
train_table.add_row([OLSmsetrain,Ridgemsetrain,PolyOLSmsetrain,GAMmsetrain,final_decision_tree_train,
                     bagged_train,random_forest_train,boosting_train, bart_train ])
print(train_table)

valid_table = PrettyTable(['OLS', 'Ridge', 'PolyOLS', 'GAM', 'Decision Tree(7)', 'Bagging', 
                           'Random Forest', 'Boosting', 'BART'])
valid_table.add_row([OLSmsevalid,Ridgemsevalid,PolyOLSmsevalid,GAMmsevalid,final_decision_tree_valid,
                     bagged_valid,random_forest_valid,boosting_valid, bart_valid ])
print(valid_table)