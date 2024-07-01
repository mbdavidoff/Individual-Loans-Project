# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:19:02 2024

@author: mbdav
"""


### Path
import os
print(os.getcwd())

### Packages
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plot
import statsmodels.api as sm
import matplotlib
import seaborn as sns


### Import datafram from input data 
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\InputData')
import Maxwell_Davidoff_HW1_Data as MDD
loans_mdf_train = MDD.loans_mdf_train

os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\Code')
loans_mdf_train.head()


### Summary of nummerical variables
numloans_df = pd.DataFrame(loans_mdf_train[['annual_income','dti','installment','int_rate','loan_amount',
                                  'total_acc','total_payment','date_difference_issue',
                                  'date_difference_last_payment']])
print(numloans_df)
print(numloans_df.describe())

### Histogram plots of nummerical variables
numloans_df.hist(bins=50, figsize=(20, 10))
plot.show()

print(numloans_df.loc[numloans_df['annual_income'] >= 1000000])

### Box plots of numerical variables

numloans_df.plot(
    kind='box', 
    subplots=True, 
    sharey=False, 
    figsize=(20, 10)
)
plot.show()

###Commentary
# Annual Income has a large range of values from $4,000 to $6,000,000. Could be some outliers as the mean is only $69,644. 14 values above $1,000,000
# Multiple skewed right distributions including intallment, interest rate, loan amount, total acounts, total payment.
# highest frequency for date difference of credit pull and issue date is between -150 days to -200 days and has a mean of -106 days
# highest frequency for date difference of the credit pull and last payment is close to 0 days with a mean of -48 days
# Max DTI is less than .3 showing that all of the individuals have more assets than liabilities

### Categorical variables summary
catloans_df = pd.DataFrame(loans_mdf_train[['emp_length', 'grade', 'home_ownership', 'loan_status', 'purpose', 
                                  'term', 'region', 'verification_status']])
print(catloans_df.describe(include = 'all'))
verification_status_cat = pd.Categorical(catloans_df['verification_status'])
print(verification_status_cat.describe())

### Categorical variables bar plots



catloans_df['emp_length'].value_counts().plot(kind='bar')
plot.xlabel('employment length')
plot.ylabel('frequency')
plot.show()
catloans_df['grade'].value_counts().plot(kind='bar')
plot.xlabel('grade')
plot.ylabel('frequency')
plot.show()
catloans_df['home_ownership'].value_counts().plot(kind='bar')
plot.xlabel('home ownership status')
plot.ylabel('frequency')
plot.show()
catloans_df['loan_status'].value_counts().plot(kind='bar')
plot.xlabel('loan status')
plot.ylabel('frequency')
plot.show()
catloans_df['purpose'].value_counts().plot(kind='bar')
plot.xlabel('purpose')
plot.ylabel('frequency')
plot.show()
catloans_df['term'].value_counts().plot(kind='bar')
plot.xlabel('term')
plot.ylabel('frequency')
plot.show()
catloans_df['region'].value_counts().plot(kind='bar')
plot.xlabel('region')
plot.ylabel('frequency')
plot.show()
verification_status_cat.value_counts().plot(kind='bar')
plot.xlabel('verificication')
plot.ylabel('frequency')
plot.show()


### Commentary on Categorical Variables

# Highest frequency for emplyment length was 10+ years with 8870 observations
# Top Grade was B with 11674 observations
# Top home ownership was renting with 18439 observations
# A high amount of the loans have been payed with 32145 observations
# The purpose of the loans were primarily debt consolidation with 18214 observations
# A majority of the loans with 36 month terms with 28237 observations
# The most common region was South Atlantic with 8836 observations
# A majority of the loans have been verified with 22112 observations

### Scatter Plots

plot.figure(figsize=(8, 6))
sns.pairplot(numloans_df)

### Commentary

# No real signs of linear relationships between variables and interest rates. Most data points look scattered.
# No linear realtionships between most of the variables showing possible low correlation
# Installment and total payment have a possible linear realtionship. Makes sense because if the loan amount is higher the installment payment should be higher.

### Correlation Coefficient Heat Map
plot.figure(figsize=(16, 6))
sns.heatmap(numloans_df.corr(),annot=True,square=True)

### Commentary
# Only 3 variables with  high correlation coefficients
# Loan amount and installment have a correlation coefficient of .93 (the highest)
# Loan amount and total payment has a hgih correlation coefficient of .89 (to be expected)
# Total payment and installment has a high correlation coefficient of .86
# Some correlation with date differences of .45



