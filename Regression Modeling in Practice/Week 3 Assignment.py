# -*- coding: utf-8 -*-
"""
Week 3 Assignment
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

# Summarize the data
print(subset[['incomeperperson', 'urbanrate']].describe())

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
"""
# Create bar chart
sns.countplot(x='SocietyType', data=subset)
plt.ylabel('Count')
plt.xlabel('')

# Create Table
counts = subset.groupby('SocietyType').size()
print(counts)

sns.distplot(subset['incomeperperson']);
plt.xlabel('')
"""
median_incomeperperson = np.median(subset['incomeperperson'])

# This function converts the polity score to a category
def full_democracy_degree(score):
    full_democracy = score / 10
    return(full_democracy)
       
# Now we can use the function to create the new variable
subset['full_democracy_degree'] = subset['polityscore'].apply(full_democracy_degree)

# This function converts the polity score to a binary category flag
def is_full_democracy(score):
    if score == 10:
        return(1)
    else:
        return(0)
        
# Now we can use the function to create the new variable
subset['is_full_democracy'] = subset['polityscore'].apply(is_full_democracy)
"""
# Visualize the relationship
sns.lmplot(x='full_democracy_degree', y='incomeperperson', data=subset, fit_reg=False, hue='SocietyType')
plt.ylabel('GDP Per Person')
plt.xlabel('Full Democracy Degree')

# Visualize the relationship
sns.lmplot(x='urbanrate', y='incomeperperson', data=subset, fit_reg=False, hue='SocietyType')
plt.ylabel('GDP Per Person')
plt.xlabel('Urbanization Rate')

ols_model = smf.ols('incomeperperson ~ full_democracy_degree + urbanrate', data=subset).fit()
print (ols_model.summary())

#Q-Q plot for normality
fig = sm.qqplot(ols_model.resid, line='r')

# simple plot of residuals
stdres=pd.DataFrame(ols_model.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')

# additional regression diagnostic plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(ols_model, 'full_democracy_degree', fig=fig)

# leverage plot
fig = sm.graphics.influence_plot(ols_model, size=8)
"""
# Outlier detection

medians = subset.groupby('SocietyType')['incomeperperson'].agg([np.median])

def get_absolute_deviation(index, value):
    median = medians[medians.index == index]['median']
    absolute_deviation = np.absolute(value-median)
    return(absolute_deviation)

def get_MAD(index):
    return(MADs[MADs.index == index]['median'])
    
def is_outlier(absolute_deviations, MAD):
    threshold = 3
    if (absolute_deviations/MAD) > threshold:
        return 1
    else:
        return 0
        
subset['absolute_deviations'] = np.vectorize(get_absolute_deviation)(subset['SocietyType'], subset['incomeperperson'])
MADs = subset.groupby('SocietyType')['absolute_deviations'].agg([np.median])
subset['MAD'] = np.vectorize(get_MAD)(subset['SocietyType'])
subset['outlier'] = np.vectorize(is_outlier)(subset['absolute_deviations'], subset['MAD'])

# Get the count of outliers
subset.groupby('SocietyType')['outlier'].agg([np.sum])
"""
# Visualize the result
sns.lmplot(x='full_democracy_degree', y='incomeperperson', data=subset, fit_reg=False, hue='outlier')
plt.ylabel('GDP Per Person')
plt.xlabel('Full Democracy Degree')

no_outliers = subset[subset.outlier == 0]
ols_model2 = smf.ols('incomeperperson ~ full_democracy_degree + urbanrate', data=no_outliers).fit()
print (ols_model2.summary())

#Q-Q plot for normality
fig = sm.qqplot(ols_model2.resid, line='r')

# simple plot of residuals
stdres=pd.DataFrame(ols_model2.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
"""
subset['log_incomeperperson'] = np.log(subset.incomeperperson)
# Summarize the data
print(subset[['log_incomeperperson']].describe())


# Outlier detection

medians = subset.groupby('is_full_democracy')['log_incomeperperson'].agg([np.median])
  
subset['absolute_deviations'] = np.vectorize(get_absolute_deviation)(subset['is_full_democracy'], subset['log_incomeperperson'])
MADs = subset.groupby('is_full_democracy')['log_incomeperperson'].agg([np.median])
subset['MAD'] = np.vectorize(get_MAD)(subset['is_full_democracy'])
subset['outlier'] = np.vectorize(is_outlier)(subset['absolute_deviations'], subset['MAD'])


model = smf.ols('log_incomeperperson ~ is_full_democracy + urbanrate', data=subset).fit()

model = smf.ols('incomeperperson ~ full_democracy_degree + I(full_democracy_degree**2) + I(urbanrate**2)', data=subset).fit()

print(model.summary())

#Q-Q plot for normality
fig = sm.qqplot(model.resid, line='r')

# simple plot of residuals
stdres=pd.DataFrame(model.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')

# Visualize the result
sns.lmplot(x='urbanrate', y='log_incomeperperson', data=subset, fit_reg=False, hue='outlier')
plt.ylabel('GDP Per Person')
plt.xlabel('Full Democracy Degree')



