# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:33:58 2024

@author: mbdav
"""
### Path
import pathlib
import os
print(os.getcwd())

### Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib.pyplot import subplots
import datetime
from sklearn.model_selection import train_test_split

### Import Data
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\RawData')
print(os.getcwd())
loans = pd.read_csv('Loan_Data.csv')
os.chdir(r'C:\\Users\\mbdav\\OneDrive\\Documents\\Big_Data_Econometric\\Git_Su24_ADEC7430\\InputData')

### Checking the Loans Dataframe
print(loans.head())
print('Rows', loans.shape[0])
print('columns',loans.shape[1])

## Employment title with 1438 NAs, however, not using this column for model
print(loans.isna().sum())

### Changing data types for many of the variables
print(loans.dtypes)

## Turning address_state into a regional categorical variable based on CDC regions
print(loans['address_state'])

region = []
for n in loans['address_state']:
    if n == 'CA' or n == 'AK' or n== 'OR' or n== 'WA' or n=='HI':
        region.append('PC')
    elif n == 'ID' or n == 'NV' or n=='MT' or n == 'WY' or n == 'UT' or n == 'AZ' or n=='CO' or n=='NM':
        region.append('MT')
    elif n == 'ND' or n == 'SD' or n == 'NE' or n == 'KS' or n == 'MS' or n == 'IA' or n == 'MO':
        region.append('WNC')
    elif n == 'TX' or n == 'OK' or n == 'AR' or n == 'LA':
        region.append('WSC')
    elif n == 'MI' or n=='WI' or n == 'IL' or n == 'IN' or n == 'OH':
        region.append('ENC')
    elif n == 'ME' or n == 'VT' or n == 'NH' or n == 'MA' or n == 'CT' or n == 'RI':
        region.append('NE')
    elif n == 'KY' or n == 'TN' or n == 'MS' or n == 'AL' or n == 'MS':
        region.append('ESC')
    elif n == 'NY' or n == 'NJ' or n == 'PA':
        region.append('MA')
    else:
        region.append('SA')

print(region[:10])
loans['region'] = pd.Categorical(region)
#loans = loans.astype({"region" : 'category'})
print(loans['region'].dtype)
print(loans.head())

loans.info()

## Turning Employment Length into a categorical variable
loans['emp_length'] = pd.Categorical(loans['emp_length'])
print(loans['emp_length'].dtype)


## Turning Loan Grade into a categorical variable
loans['grade'] = pd.Categorical(loans['grade'])
print(loans['grade'].dtype)


## Turning home ownership into a categorical variable

loans['home_ownership'] = pd.Categorical(loans['home_ownership'])
print(loans['home_ownership'].dtype)


## Turning all dates into dates data type
loans['issue_date'] = pd.to_datetime(loans['issue_date'], infer_datetime_format=True)
print(loans['issue_date'])

loans['last_payment_date'] = pd.to_datetime(loans['last_payment_date'], infer_datetime_format=True)
print(loans['last_payment_date'])

loans['next_payment_date'] = pd.to_datetime(loans['next_payment_date'], infer_datetime_format=True)
print(loans['next_payment_date'])

loans['last_credit_pull_date'] = pd.to_datetime(loans['last_credit_pull_date'], infer_datetime_format=True)
print(loans['last_credit_pull_date'])

## Date difference between between credit pull and issue date and last payment date and issue date
date_difference_credit_pull_vs_issue = loans['last_credit_pull_date'] - loans['issue_date']
print(date_difference_credit_pull_vs_issue)

date_difference_credit_pull_vs_last = loans['last_credit_pull_date'] - loans['last_payment_date']

date_difference_issue=[]
for a in date_difference_credit_pull_vs_issue:
    date_difference_issue.append(float(str(a).split(" ")[0]))

date_difference_last_payment=[]
for q in date_difference_credit_pull_vs_last:
    date_difference_last_payment.append(float(str(q).split(" ")[0]))
    
loans['date_difference_issue'] = date_difference_issue
loans['date_difference_last_payment'] = date_difference_last_payment

print(loans['date_difference_issue'].dtype)


## Turning the reason for the loan into categories

loans['purpose'] = pd.Categorical(loans['purpose'])
print(loans['purpose'].dtype)


## Turning Loan Status into a categories

loans['loan_status'] = pd.Categorical(loans['loan_status'])
print(loans['loan_status'].dtype)

## Turning Loan term into a categories

loans['term'] = pd.Categorical(loans['term'])
print(loans['term'].dtype)


## Turning Verification Status into a summy 1 for either verfied or source verified 0 for not verified
loans['verification_status'] = pd.Categorical(loans['verification_status'])
print(loans['verification_status'].dtype)



### Removiing irrelevant features and Splitting the data set before proceeding with the same features

loans_mdf = loans.select_dtypes(exclude=['object'])
loans_mdf = loans_mdf.select_dtypes(exclude=['datetime64[ns]'])
loans_mdf = loans_mdf.drop('member_id', axis = 'columns')
loans_mdf = loans_mdf.drop('id', axis = 'columns')
print(loans_mdf.dtypes)

loans_mdf_train, loans_mdf_valid = train_test_split(loans_mdf,test_size = .2, random_state=1070)
pd.set_option('display.max_columns', None)
print(loans_mdf_train.head())
print('Rows',loans_mdf_train.shape[0])
print('Columns',loans_mdf_train.shape[1])
print(loans_mdf_train.dtypes)
print(loans_mdf_train.describe())

## No NAs

print(loans_mdf_train.isna().sum())

### Turning verification status into a dummy due to frequencies and verification and source verified are equivalent for our purposes

loans_mdf_train['verification_status']=loans_mdf_train['verification_status'].replace({'Source Verified':1,'Verified':1,
                                                                   'Not Verified':0})
loans_mdf_valid['verification_status']=loans_mdf_valid['verification_status'].replace({'Source Verified':1,'Verified':1,
                                                                   'Not Verified':0})


v_train = []
for e in loans_mdf_train['verification_status']:
    
    v_train.append(int(str(e).split(" ")[0]))

v_valid = []
for e in loans_mdf_valid['verification_status']:
 
    v_valid.append(int(str(e).split(" ")[0]))
    
loans_mdf_train['verification_status'] = v_train
loans_mdf_valid['verification_status'] = v_valid


print(loans_mdf_train['verification_status'])
print(loans_mdf_train['verification_status'].dtype)
print(loans_mdf_train['verification_status'].describe())

### Notice that there were 0 values for DTI dropping observations where DTI = 0
indexdti0 = loans_mdf_train[(loans_mdf_train['dti'] == 0)].index
print(indexdti0)
print(len(indexdti0))
loans_mdf_train.drop(indexdti0, inplace=True)


### Taking log of annual salaries due to the nature of the variable
loans_mdf_train['annual_income'] = np.log(loans_mdf_train['annual_income'])
loans_mdf_valid['annual_income'] = np.log(loans_mdf_valid['annual_income'])
print(loans_mdf_train['annual_income'])


### Multiplying interest rate by 100 to get it in terms of percentage
loans_mdf_train['int_rate'] = 100*loans_mdf_train['int_rate']
loans_mdf_valid['int_rate'] = 100*loans_mdf_valid['int_rate']
print(loans_mdf_train['int_rate'])
####### Final Results after data cleaning & feauture creation
pd.set_option('display.max_columns', None)
print(loans_mdf_train.head())
print('Rows',loans_mdf_train.shape[0])
print('Columns',loans_mdf_train.shape[1])
print(loans_mdf_train.dtypes)
print(loans_mdf_train.describe())










