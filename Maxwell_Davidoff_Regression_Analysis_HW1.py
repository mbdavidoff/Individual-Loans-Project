# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:37:42 2024

@author: mbdav
"""

### Path
import os
print(os.getcwd)

### Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)

from functools import partial
from sklearn.model_selection import \
     (cross_validate ,
      KFold ,
      ShuffleSplit)
from sklearn.base import clone
from ISLP.models import sklearn_sm
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
from ISLP.models import \
    (Stepwise ,
     sklearn_selected ,
     sklearn_selection_path)
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import \
    (Stepwise ,
     sklearn_selected ,
     sklearn_selection_path)
from sklearn.linear_model import Ridge

### import data
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\InputData')
import Maxwell_Davidoff_HW1_Data as MDD;
loans_mdf_train = MDD.loans_mdf_train
loans_mdf_valid = MDD.loans_mdf_valid
loans_mdf = MDD.loans_mdf
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\Code')
loans_mdf_train.head()
dir()

### Creating variables
terms = MS(loans_mdf.columns.drop('int_rate')).fit(loans_mdf_train)
X_train = terms.transform(loans_mdf_train)
y_train = loans_mdf_train['int_rate']
X_valid = terms.transform(loans_mdf_valid)
y_valid = loans_mdf_valid['int_rate']

print(X_train.head())
print(y_train.head())
print('observations_train', X_train.shape[0])
print('observations_test', y_valid.shape[0])

### Forward elimination to for variable selection using Cp
def nCp(s2 , e , X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = e.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    return -(RSS + 2 * p * s2) / n


sigma = sm.OLS(y_train,X_train).fit().scale


n_Cp = partial(nCp , sigma)

strategy = Stepwise.first_peak(terms,
                               direction='forward',
                               max_terms=len(terms.terms))

m_cp = sklearn_selected(sm.OLS ,
                               strategy,
                               scoring=n_Cp)
m_cp.fit(loans_mdf_train, y_train)
print(m_cp.selected_state_)


### Commentary
# USing forward selection with the negative CP function, we narrowed down our independent variables to 14
# Forward selection runs forward until no improvements are made in the models CP

### Creating linear model using the forward selected variables


X_train = X_train.drop(X_train.columns[[1,2,3,4,5,6,7,8,9,10,38,44,45,46,47,48,49,50,51]], axis =1)
X_valid = X_valid.drop(X_valid.columns[[1,2,3,4,5,6,7,8,9,10,38,44,45,46,47,48,49,50,51]], axis =1)
model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())
print('RSS model 1', results.ssr)



# Residual plot
plt.scatter(results.fittedvalues, results.resid)
plt.xlabel('fitted')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.show()
fig = plt.figure(figsize=(14, 8)) 

# QQ Plot
QQ = ProbPlot(results.get_influence().resid_studentized_internal)
plot_l2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_l2.axes[0].set_title('QQ Plot')
plot_l2.axes[0].set_xlabel('Theoretical Quantiles')
plot_l2.axes[0].set_ylabel('Standardized Residuals');
abs_norm_resid = np.flip(np.argsort(np.abs(results.get_influence().resid_studentized_internal)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]
for r, i in enumerate(abs_norm_resid_top_3):
    plot_l2.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   results.get_influence().resid_studentized_internal[i]));
# High Leverage Plot   
plot_l4 = plt.figure();
plt.scatter(results.get_influence().hat_matrix_diag, results.get_influence().resid_studentized_internal, alpha=0.5);
sns.regplot(x=results.get_influence().hat_matrix_diag, y=results.get_influence().resid_studentized_internal,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_l4.axes[0].set_xlim(0, max(results.get_influence().hat_matrix_diag)+0.01)
plot_l4.axes[0].set_ylim(-3, 5)
plot_l4.axes[0].set_title('Residuals vs Leverage')
plot_l4.axes[0].set_xlabel('Leverage')
plot_l4.axes[0].set_ylabel('Standardized Residuals');

  
leverage_top_3 = np.flip(np.argsort(results.get_influence().cooks_distance[0]), 0)[:3]
for i in leverage_top_3:
    plot_l4.axes[0].annotate(i,
                               xy=(results.get_influence().hat_matrix_diag[i],
                                  results.get_influence().resid_studentized_internal[i]));

fig = plt.figure(figsize=(7, 20))
pplot = sm.graphics.plot_partregress_grid(results, fig=fig )

## Cooks distance plot
influence = results.get_influence()
cd = influence.cooks_distance

plt.figure(figsize = (12, 8))
plt.scatter(X_train.index, cd[0])
plt.axhline(y = (2*14)/30713, color ="green", linestyle =":")
plt.xlabel('Row Number', fontsize = 12)
plt.ylabel('Cooks Distance', fontsize = 12)
plt.title('Influencial Points', fontsize = 22)
plt.show()

#VIF
vals = [VIF(X_train, i)
        for i in range(1, X_train.shape [1])]
vif = pd.DataFrame ({'vif':vals},
                    index=X_train.columns [1:])
print(vif)


### Model 2 Adjusting for variable inflation 
#by taking out loan_amount and total_payment

X_train = X_train.drop(X_train.columns[[30, 32]], axis =1)
X_valid = X_valid.drop(X_valid.columns[[30, 32]], axis =1)

model2 = sm.OLS(y_train, X_train)
results2 = model2.fit()
print(results2.summary())
print('RSS model 2', results2.ssr)
fp_test = anova_lm(results2, results)
print(fp_test)


vals = [VIF(X_train, i)
        for i in range(1, X_train.shape [1])]
vif = pd.DataFrame ({'vif':vals},
                    index=X_train.columns [1:])
print(vif)


### Find influential points based on cooks distance and remove
cutoff = (2*(11+1))/30713
inf_points = X_train.index[cd[0] > cutoff]
print(inf_points)

X_train = X_train.drop(index =[24714, 24751, 35492, 35486,  7656, 35567, 37653, 27657, 35489,
            38185,  1976, 24953, 35490,  6615,  9665,  2566,  9662, 31666,
            25074, 36593, 28285, 24711, 37660, 27658,  6238, 34300, 24979,
            35054,  9655, 30834,  2603, 32476, 35451, 35465, 33643, 14160,
             5632, 31926,  7744, 30974, 38515,  6246, 17699, 28694, 24713,
             7038, 27953, 26530, 36597, 35458, 35502,  6432,  9653, 31739,
             6764, 35436, 35559, 36443, 17698, 36899,  4566, 35498,  9659,
            35716, 24912, 35066, 31667, 31740, 22311, 35435, 35485, 24915])
y_train = y_train.drop(index = [24714, 24751, 35492, 35486,  7656, 35567, 37653, 27657, 35489,
            38185,  1976, 24953, 35490,  6615,  9665,  2566,  9662, 31666,
            25074, 36593, 28285, 24711, 37660, 27658,  6238, 34300, 24979,
            35054,  9655, 30834,  2603, 32476, 35451, 35465, 33643, 14160,
             5632, 31926,  7744, 30974, 38515,  6246, 17699, 28694, 24713,
             7038, 27953, 26530, 36597, 35458, 35502,  6432,  9653, 31739,
             6764, 35436, 35559, 36443, 17698, 36899,  4566, 35498,  9659,
            35716, 24912, 35066, 31667, 31740, 22311, 35435, 35485, 24915])

### Model 3
model3 = sm.OLS(y_train, X_train)
results3 = model3.fit()
print(results3.summary())
print('RSS model 3', results3.ssr)


# Residual Plot
plt.scatter(results3.fittedvalues, results3.resid)
plt.xlabel('fitted')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.show()
fig = plt.figure(figsize=(14, 8)) 

#QQ Plot
QQ = ProbPlot(results3.get_influence().resid_studentized_internal)
plot_l2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_l2.axes[0].set_title('QQ Plot')
plot_l2.axes[0].set_xlabel('Theoretical Quantiles')
plot_l2.axes[0].set_ylabel('Standardized Residuals');
abs_norm_resid = np.flip(np.argsort(np.abs(results3.get_influence().resid_studentized_internal)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]
for r, i in enumerate(abs_norm_resid_top_3):
    plot_l2.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   results3.get_influence().resid_studentized_internal[i]));
# High leverage plot    
plot_l4 = plt.figure();
plt.scatter(results3.get_influence().hat_matrix_diag, results3.get_influence().resid_studentized_internal, alpha=0.5);
sns.regplot(x=results3.get_influence().hat_matrix_diag, y=results3.get_influence().resid_studentized_internal,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_l4.axes[0].set_xlim(0, max(results3.get_influence().hat_matrix_diag)+0.01)
plot_l4.axes[0].set_ylim(-3, 5)
plot_l4.axes[0].set_title('Residuals vs Leverage')
plot_l4.axes[0].set_xlabel('Leverage')
plot_l4.axes[0].set_ylabel('Standardized Residuals');

  
leverage_top_3 = np.flip(np.argsort(results.get_influence().cooks_distance[0]), 0)[:3]
for i in leverage_top_3:
    plot_l4.axes[0].annotate(i,
                             xy=(results.get_influence().hat_matrix_diag[i],
                                 results.get_influence().resid_studentized_internal[i]));

fig = plt.figure(figsize=(7, 20))
pplot = sm.graphics.plot_partregress_grid(results3, fig=fig )

## Cooks distance plot
influence = results3.get_influence()
cd = influence.cooks_distance

plt.figure(figsize = (12, 8))
plt.scatter(X_train.index, cd[0])
plt.axhline(y = (2*13)/30668, color ="green", linestyle =":")
plt.xlabel('Row Number', fontsize = 12)
plt.ylabel('Cooks Distance', fontsize = 12)
plt.title('Influencial Points', fontsize = 22)
plt.show()

#VIF
vals = [VIF(X_train, i)
        for i in range(1, X_train.shape [1])]
vif = pd.DataFrame ({'vif':vals},
                    index=X_train.columns [1:])
print(vif)

### Comparing MSE with highly influential points taken out

validp = results3.predict(X_valid)
trainp = results3.predict(X_train)
print('MSE validate', np.mean((y_valid - validp)**2))
print('MSE train', np.mean((y_train - trainp)**2))
print('R-Sqaured model 3', .929)
Ybar = y_valid.mean()
sst = ((y_valid - Ybar)**2).sum()
ssr = ((y_valid - validp)**2).sum()
R2_valid = 1-(ssr/sst)
print('R-Squared validate', R2_valid)


### Ridge Regression creating lambdas and grid
scalen = StandardScaler()
loans_mdf_train = loans_mdf_train.reset_index(drop = True)
loans_mdf_valid = loans_mdf_valid.reset_index(drop = True)
print(loans_mdf_train)


loans_mdf_train['verification_status'] = pd.Categorical(loans_mdf_train['verification_status'])
loans_mdf_valid['verification_status'] = pd.Categorical(loans_mdf_valid['verification_status'])
loans_mdf_train['int_rate']= pd.Categorical(loans_mdf_train['int_rate'])
loans_mdf_valid['int_rate'] =pd.Categorical(loans_mdf_valid['int_rate'])
ncs = loans_mdf_train.select_dtypes(include=['int64', 'float64']).columns
loans_mdf_train[ncs] = scalen.fit_transform(loans_mdf_train[ncs])
loans_mdf_valid[ncs] = scalen.fit_transform(loans_mdf_valid[ncs])
int_ratet = []
for e in loans_mdf_train['int_rate']:
 
    int_ratet.append(float(str(e).split(" ")[0]))
int_ratev = []
for e in loans_mdf_valid['int_rate']:
 
    int_ratev.append(float(str(e).split(" ")[0]))

loans_mdf_train['int_rate'] = int_ratet
loans_mdf_valid['int_rate'] = int_ratev
    



terms = MS(loans_mdf_train.columns.drop(['int_rate']))
X_train_ridge = terms.fit_transform(loans_mdf_train)
y_train_ridge = loans_mdf_train['int_rate']
X_valid_ridge = terms.transform(loans_mdf_valid)
y_valid_ridge = loans_mdf_valid['int_rate']
print(X_train_ridge.shape)

X_valid_ridge = X_valid_ridge.drop('intercept', axis =1)
X_train_ridge = X_train_ridge.drop('intercept', axis =1)

#Y = np.array(loans_mdf['int_rate'])
#C = terms.transform(loans_mdf)
#C = C.drop('intercept', axis = 1)
#X = np.asarray(C)

#Xs = X - X.mean (0)[None ,:]
#X_scale = X.std (0)
#Xs = Xs / X_scale[None ,:]
lambdas = 10**np.linspace (8, -2, 100) / y_train_ridge.std()
s_array = skl.ElasticNet.path(X_train_ridge,
                              y_train_ridge,
                              l1_ratio =0.,
                              alphas=lambdas)[1]
print(s_array.shape)

s_path = pd.DataFrame(s_array.T,
                          columns=X_train_ridge.columns ,
                          index=-(np.log(lambdas)))
s_path.index.name = '$-\log(\ lambda)$'
print(s_path)

#Plot to show the variance of the effect on the coefficient depending on the lambda
path_fig , ax = plt.subplots(figsize =(8 ,8))
s_path.plot(ax=ax , legend=False)
ax.set_xlabel('$-\log(\ lambda)$', fontsize =20)
ax.set_ylabel('Standardized coefficients', fontsize =20);

               
#Finding the best lambda using k folds
ridge = skl.ElasticNet(alpha=lambdas, l1_ratio =0)
pipe = Pipeline(steps =[('ridge', ridge)])
K = 5
kfold = skm.KFold(K,
                  random_state =0,
                  shuffle=True)
p_grid = {'ridge__alpha': lambdas}
grid = skm.GridSearchCV(pipe ,
                        p_grid ,
                        cv=kfold ,
                        scoring='neg_mean_squared_error')
results_ridge = grid.fit(X_train_ridge, y_train_ridge)

## Summary
print(results_ridge)
bh = s_path.loc[s_path.index [99]]
print(lambdas [99], bh)
print('cross validation MSE', results_ridge.best_score_)
print('best estimator', results_ridge.best_estimator_)
print(results_ridge.best_params_)


gridr2 = skm.GridSearchCV(pipe ,
                          p_grid ,
                          cv=kfold,
                          )
results_ridge2 = gridr2.fit(X_train_ridge, y_train_ridge)
print('cross validation R2', results_ridge2.best_score_)
print('best estimator', results_ridge2.best_estimator_)


#MSE plot 
rf , ax = plt.subplots(figsize =(8 ,8))
ax.errorbar(-np.log(lambdas),
            -grid.cv_results_['mean_test_score'],
            yerr=grid.cv_results_['std_test_score'] / np.sqrt(K))
#ax.set_ylim ([50000 ,250000])
ax.set_xlabel('$-\log(\ lambda)$', fontsize =20)
ax.set_ylabel('Cross -validated MSE', fontsize =20);

#R2 plot
r2 , ax = plt.subplots(figsize =(8 ,8))
ax.errorbar(-np.log(lambdas),
            gridr2.cv_results_['mean_test_score'],
            yerr=gridr2.cv_results_['std_test_score'] / np.sqrt(K)
                )
ax.set_xlabel('$-\log(\ lambda)$', fontsize =20)
ax.set_ylabel('Cross -validated $R^2$', fontsize =20);

### Refit Train and test MSE
ridge_final = Ridge(alpha = .00269, fit_intercept= True)
ridge_final.fit(X_train_ridge, y_train_ridge)


validpr = ridge_final.predict(X_valid_ridge)
trainpr = ridge_final.predict(X_train_ridge)
print('MSE validate', np.mean((y_valid_ridge - validpr)**2))
print('MSE train', np.mean((y_train_ridge - trainpr)**2))

