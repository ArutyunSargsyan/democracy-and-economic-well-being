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
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

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
def convert_polityscore_to_category(score):
    if score == 10:
        return('1-Full Democracy')
    elif score > 5:
        return('2-Democracy')
    elif score > 0:
        return ('3-Open Anocracy')
    elif score > -6:
        return ('4-Closed Anocracy')
    else:
        return('5-Autocracy')

# Now we can use the function to create the new variable
subset['SocietyType'] = subset['polityscore'].apply(convert_polityscore_to_category)
subset['SocietyType'] = subset['SocietyType'].astype('category')

# Create bar chart
sns.countplot(x='SocietyType', data=subset)
plt.ylabel('Count')
plt.xlabel('')

# Set binary flag that income per person is greater than the mean
avg_income = np.mean(subset['incomeperperson'])
def higher_than_average_income(income):
    if income > avg_income:
        return 1
    else:
        return 0

subset['higher_than_average_income'] = subset['incomeperperson'].apply(higher_than_average_income)

counts = subset.groupby('higher_than_average_income').size()
print(counts)

##############################################################################
# LOGISTIC REGRESSION
##############################################################################

# logistic regression with society type
lreg1 = smf.logit(formula = 'higher_than_average_income ~ C(SocietyType)', data = subset).fit()
print (lreg1.summary())

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

# logistic regression with society type and urbanization rate
lreg2 = smf.logit(formula = 'higher_than_average_income ~ C(SocietyType) + urbanrate', data = subset).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))