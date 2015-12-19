# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:19:46 2015

@author: Michael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

df = pd.read_csv('gapminder.csv')

# convert to numeric format
df['incomeperperson'] = pd.to_numeric(df['incomeperperson'], errors='coerce')
df['polityscore'] = pd.to_numeric(df['polityscore'], errors='coerce')
df['urbanrate'] = pd.to_numeric(df['urbanrate'], errors='coerce')

# listwise deletion of missing values
subset = df[['incomeperperson', 'polityscore', 'urbanrate']].dropna()

# This function converts the polity score to a category
def convert_polityscore_to_category(polityscore):
    if polityscore == 10:
        return 1
    else:
        return 0

# Now we can use the function to create the new variable
subset['full_democracy'] = subset['polityscore'].apply(convert_polityscore_to_category)

counts = subset.groupby('full_democracy').size()
print(counts)

# Create a threshold
income_threshold = np.mean(subset['incomeperperson'])
print(income_threshold)

# Set binary flag that income per person is greater than the threshold
def income_higher_than_threshold(income):
    if income > income_threshold:
        return 1
    else:
        return 0

subset['high_income'] = subset['incomeperperson'].apply(income_higher_than_threshold)

counts = subset.groupby('high_income').size()
print(counts)

# Create a threshold
urbanization_threshold = np.mean(subset['urbanrate'])
print(urbanization_threshold)

# Set binary flag that urbanization rate is greater than the threshold
def urbanrate_higher_than_threshold(urbanrate):
    if urbanrate > urbanization_threshold:
        return 1
    else:
        return 0

subset['high_urbanrate'] = subset['urbanrate'].apply(urbanrate_higher_than_threshold)

counts = subset.groupby('high_urbanrate').size()
print(counts)

# logistic regression with society type
lreg1 = smf.logit(formula = 'high_income ~ full_democracy', data = subset).fit()
print (lreg1.summary())

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

# logistic regression with society type and urbanization rate
lreg2 = smf.logit(formula = 'high_income ~ full_democracy + high_urbanrate', data = subset).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))