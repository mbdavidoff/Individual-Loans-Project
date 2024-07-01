# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:13:47 2024

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
import statsmodels.api as sm
from ISLP.models import (summarize ,
                         poly ,
                         ModelSpec as MS)
from statsmodels.stats.anova import anova_lm
from pygam import (s as s_gam ,
                   l as l_gam ,
                   f as f_gam ,
                   LinearGAM ,
                   LogisticGAM)
from ISLP.transforms import (BSpline ,
                             NaturalSpline)
from ISLP.models import bs , ns
from ISLP.pygam import (approx_lam ,
                        degrees_of_freedom ,
                        plot as plot_gam ,
                        anova as anova_gam)
from matplotlib.pyplot import subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import \
     (cross_validate ,
      KFold ,
      ShuffleSplit)
from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF
from statsmodels.graphics.gofplots import ProbPlot
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.model_selection as skm

### Refitting the best OLS Regression from the first HW, interest rate is multiplied by 100

terms = MS(loans_mdf_train.columns.drop('int_rate')).fit(loans_mdf_train)
X_train = terms.transform(loans_mdf_train)
y_train = loans_mdf_train['int_rate']
X_valid = terms.transform(loans_mdf_valid)
y_valid = loans_mdf_valid['int_rate']

print(X_train.head())
print(y_train.head())
print('observations_train', X_train.shape[0])
print('observations_test', y_valid.shape[0])

## Based on Forward Selection from HW1
X_train = X_train.drop(X_train.columns[[1,2,3,4,5,6,7,8,9,10,38,44,45,46,47,48,49,50,51]], axis =1)
X_valid = X_valid.drop(X_valid.columns[[1,2,3,4,5,6,7,8,9,10,38,44,45,46,47,48,49,50,51]], axis =1)

## Based on multicollinearity from HW1
X_train = X_train.drop(X_train.columns[[30, 32]], axis =1)
X_valid = X_valid.drop(X_valid.columns[[30, 32]], axis =1)

## Based on influential points from HW1

X_train = X_train.drop(index = [24714, 24751, 35492, 35486,  7656, 35567, 37653, 27657, 35489,
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
## Fitting OLS model from HW1
lm_OLS = sm.OLS(y_train, X_train)
results = lm_OLS.fit()
print(results.summary())

### Looking at linearity using Lowess lines in plots
sns.regplot(x = X_train['dti'], y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['dti'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

sns.regplot(x = X_train['installment'], y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['installment'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

sns.regplot(x = X_train['total_acc'], y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['total_acc'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

sns.regplot(x = X_train['date_difference_issue'], y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['date_difference_issue'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()
sns.regplot(x = X_train['total_acc'], y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['total_acc'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

sns.regplot(x = X_train['date_difference_last_payment'], y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['date_difference_last_payment'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

# Full model
yhat = results.predict(X_train)
sns.regplot(x = yhat, y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = yhat, y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

### Testing polynomial transformation for total_acc

X_train['total_acc^2'] = X_train['total_acc']**2
X_valid['total_acc^2'] = X_valid['total_acc']**2
polyr = sm.OLS(y_train, X_train)
results_poly = polyr.fit()
print(results_poly.summary())

X_train['total_acc^3'] = X_train['total_acc']**3
X_valid['total_acc^3'] = X_valid['total_acc']**3
polyr2 = sm.OLS(y_train, X_train)
results_poly2 = polyr2.fit()
print(results_poly2.summary())

# Ftests
print(anova_lm(results, results_poly))
print(anova_lm(results,results_poly2))
print(anova_lm(results_poly,results_poly2))

# Dropping total_acc^3 because it did not have statistically significant effect in capturing the dependent variables variance
X_train = X_train.drop(X_train.columns[[34]], axis =1)
X_valid = X_valid.drop(X_valid.columns[[34]], axis =1)
### testing polynomials of dti
X_train['dti^2'] = X_train['dti']**2
X_valid['dti^2'] = X_valid['dti']**2
polyr3 = sm.OLS(y_train, X_train)
results_poly3 = polyr3.fit()
print(results_poly3.summary())

X_train['dti^3'] = X_train['dti']**3
X_valid['dti^3'] = X_valid['dti']**3
polyr4 = sm.OLS(y_train, X_train)
results_poly4 = polyr4.fit()
print(results_poly4.summary())

X_train['dti^4'] = X_train['dti']**4
X_valid['dti^4'] = X_valid['dti']**4
polyr5 = sm.OLS(y_train, X_train)
results_poly5 = polyr5.fit()
print(results_poly5.summary())

X_train['dti^5'] = X_train['dti']**5
X_valid['dti^5'] = X_valid['dti']**5
polyr6 = sm.OLS(y_train, X_train)
results_poly6 = polyr6.fit()
print(results_poly6.summary())

# Ftests
print(anova_lm(results_poly,results_poly3))
print(anova_lm(results_poly3,results_poly4))
print(anova_lm(results_poly4,results_poly5))
print(anova_lm(results_poly5,results_poly6))
print(anova_lm(results_poly,results_poly4))
print(anova_lm(results_poly,results_poly5))
print(anova_lm(results_poly,results_poly6))
print(anova_lm(results_poly3,results_poly5))

# Dropping dti^5
X_train = X_train.drop(X_train.columns[[37]], axis =1)
X_valid = X_valid.drop(X_valid.columns[[37]], axis =1)
### Lowess plots after transformations

sns.regplot(x = X_train['total_acc'], y = y_train, color="blue", order = 2,
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['total_acc'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

sns.regplot(x = X_train['dti'], y = y_train, color="blue", order = 4,
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = X_train['dti'], y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

### Final Polynomial Regression
final_polyr = sm.OLS(y_train, X_train)
finalpoly_results = final_polyr.fit()
print(finalpoly_results.summary())
print('RSS final polynomial', finalpoly_results.ssr)
#VIF
vals = [VIF(X_train, i)
        for i in range(1, X_train.shape [1])]
vif = pd.DataFrame ({'vif':vals},
                    index=X_train.columns [1:])
print(vif)

### Diagnostic Plots
# Residual Plot
plt.scatter(finalpoly_results.fittedvalues, finalpoly_results.resid)
plt.xlabel('fitted')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.show()
fig = plt.figure(figsize=(14, 8)) 

#QQ Plot
QQ = ProbPlot(finalpoly_results.get_influence().resid_studentized_internal)
plot_l2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_l2.axes[0].set_title('QQ Plot')
plot_l2.axes[0].set_xlabel('Theoretical Quantiles')
plot_l2.axes[0].set_ylabel('Standardized Residuals');
abs_norm_resid = np.flip(np.argsort(np.abs(finalpoly_results.get_influence().resid_studentized_internal)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]
for r, i in enumerate(abs_norm_resid_top_3):
    plot_l2.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   finalpoly_results.get_influence().resid_studentized_internal[i]));
# High leverage plot    
plot_l4 = plt.figure();
plt.scatter(finalpoly_results.get_influence().hat_matrix_diag, finalpoly_results.get_influence().resid_studentized_internal, alpha=0.5);
sns.regplot(x=finalpoly_results.get_influence().hat_matrix_diag, y=finalpoly_results.get_influence().resid_studentized_internal,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_l4.axes[0].set_xlim(0, max(finalpoly_results.get_influence().hat_matrix_diag)+0.01)
plot_l4.axes[0].set_ylim(-3, 5)
plot_l4.axes[0].set_title('Residuals vs Leverage')
plot_l4.axes[0].set_xlabel('Leverage')
plot_l4.axes[0].set_ylabel('Standardized Residuals');

  
leverage_top_3 = np.flip(np.argsort(finalpoly_results.get_influence().cooks_distance[0]), 0)[:3]
for i in leverage_top_3:
    plot_l4.axes[0].annotate(i,
                             xy=(finalpoly_results.get_influence().hat_matrix_diag[i],
                                 finalpoly_results.get_influence().resid_studentized_internal[i]));

fig = plt.figure(figsize=(7, 20))
pplot = sm.graphics.plot_partregress_grid(finalpoly_results, fig=fig )

## Cooks distance plot
influence = finalpoly_results.get_influence()
cd = influence.cooks_distance

plt.figure(figsize = (12, 8))
plt.scatter(X_train.index, cd[0])
plt.axhline(y = (2*13)/30678, color ="green", linestyle =":")
plt.xlabel('Row Number', fontsize = 12)
plt.ylabel('Cooks Distance', fontsize = 12)
plt.title('Influencial Points', fontsize = 22)
plt.show()

yhat2 = finalpoly_results.predict(X_train)
sns.regplot(x = yhat2, y = y_train, color="blue", 
            line_kws={"color":"red", "linewidth":3})
sns.regplot(x = yhat2, y = y_train, color="blue",  lowess = True,
            line_kws={"color":"magenta", "linewidth":3})
plt.show()

### Train vs Test error of polynomial
validpolyp = finalpoly_results.predict(X_valid)
trainpolyp = finalpoly_results.predict(X_train)
print('MSE validate', np.mean((y_valid - validpolyp)**2))
print('MSE train', np.mean((y_train - trainpolyp)**2))
print('R-Sqaured model 3', .929)
Ybar = y_valid.mean()
sst = ((y_valid - Ybar)**2).sum()
ssr = ((y_valid - validpolyp)**2).sum()
R2_valid = 1-(ssr/sst)
print('R-Squared validate', R2_valid)

### GAMS Model
# Setting up data for GAM using same features as polynomial regression
Ytrain_gam = loans_mdf_train['int_rate']
Ytest_gam = loans_mdf_valid['int_rate']
C_train = loans_mdf_train.drop(loans_mdf_train.columns[[0,7,10,11,13,14]], axis =1)

C_train['grade']=C_train['grade'].cat.codes
C_train['home_ownership']=C_train['home_ownership'].cat.codes
C_train['loan_status']=C_train['loan_status'].cat.codes
C_train['purpose']=C_train['purpose'].cat.codes
C_train['term']=C_train['term'].cat.codes
Xg_train = np.array(C_train)
C_test = loans_mdf_valid.drop(loans_mdf_valid.columns[[0,7,10,11,13,14]], axis =1)

C_test['grade']=C_test['grade'].cat.codes
C_test['home_ownership']=C_test['home_ownership'].cat.codes
C_test['loan_status']=C_test['loan_status'].cat.codes
C_test['purpose']=C_test['purpose'].cat.codes
C_test['term']=C_test['term'].cat.codes
Xg_test = np.array(C_test)

### Original GAM with the default .6 lambda 

gam = LinearGAM(terms = f_gam(0,lam=0) + f_gam(1,lam=0)+f_gam(2,lam=0)+f_gam(3,lam=0)+
                f_gam(4, lam=0)+ f_gam(5,lam=0)+s_gam(6)+s_gam(7)+s_gam(8)+
                s_gam(9)+s_gam(10)).fit(Xg_train, Ytrain_gam)
print(gam.summary())

lams = np.exp(np.random.randn(100,11) * 6 - 3)
gam.gridsearch(X = Xg_train, y = Ytrain_gam, lam=lams)

print(gam.summary())

### FInal GAM with appropriate Lambda shrinkage 

gamf = LinearGAM(terms = f_gam(0,lam=0) + f_gam(1,lam=0)+f_gam(2,lam=0)+f_gam(3,lam=0)+f_gam(4, lam=0)+ 
                 f_gam(5,lam=0)+s_gam(6,lam = 197.6547)+s_gam(7,lam = .0294)+s_gam(8,lam=7012.6212)+
                 s_gam(9, lam = .0051)+s_gam(10, lam=5.2547)).fit(Xg_train, Ytrain_gam)
print(gamf.summary())

### Partial plots 

fig , ax = subplots(figsize =(8 ,8))
plot_gam(gamf, 6, ax=ax)
ax.set_xlabel('dti')
ax.set_ylabel('Effect on interest rate')
ax.set_title('Partial dependence of dti on interest rate',
fontsize =20)

fig , ax = subplots(figsize =(8 ,8))
plot_gam(gamf, 7, ax=ax)
ax.set_xlabel('installment')
ax.set_ylabel('Effect on interest rate')
ax.set_title('Partial dependence of installment on interest rate',
fontsize =20)

fig , ax = subplots(figsize =(8 ,8))
plot_gam(gamf, 8, ax=ax)
ax.set_xlabel('total_acc')
ax.set_ylabel('Effect on interest rate')
ax.set_title('Partial dependence of total_acc on interest rate',
fontsize =20)

fig , ax = subplots(figsize =(8 ,8))
plot_gam(gamf, 9, ax=ax)
ax.set_xlabel('date_difference_issue')
ax.set_ylabel('Effect on interest rate')
ax.set_title('Partial dependence of date_difference_issue on interest rate',
fontsize =20)

fig , ax = subplots(figsize =(8 ,8))
plot_gam(gamf, 10, ax=ax)
ax.set_xlabel('date_difference_last_payment')
ax.set_ylabel('Effect on interest rate')
ax.set_title('Partial dependence of date_difference_last_payment on interest rate',
fontsize =20)


### MSE for train and test
gampredtrain = gamf.predict(Xg_train)
gampredtest = gamf.predict(Xg_test)
print('MSE validate', np.mean((Ytest_gam-gampredtest)**2))
print('MSE train', np.mean((Ytrain_gam - gampredtrain)**2))
print('R-Sqaured model 3', .9284)
Ybar = Ytest_gam.mean()
sst = ((Ytest_gam - Ybar)**2).sum()
ssr = ((Ytest_gam - gampredtest)**2).sum()
R2_valid = 1-(ssr/sst)
print('R-Squared validate', R2_valid)

