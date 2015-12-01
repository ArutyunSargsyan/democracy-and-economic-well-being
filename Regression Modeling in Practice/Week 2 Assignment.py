# -*- coding: utf-8 -*-
"""
Week 2 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the Data
df = pd.read_csv('gapminder.csv', low_memory=False)

# Print some basic statistics
n = str(len(df))
cols = str(len(df.columns))
print('Number of observations: '+ n +' (rows)')
print('Number of variables: '+ cols +' (columns)')
print('\n')

# Change the data type for variables of interest
# Response Variable
df['incomeperperson'] = pd.to_numeric(df['incomeperperson'], errors='coerce')
# Explanatory Variable
df['polityscore'] = pd.to_numeric(df['polityscore'], errors='coerce')

# Print the number of records with data
print ('Countries with a GDP Per Capita: ' + str(df['incomeperperson'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['incomeperperson'].count()) + ' missing)')
print ('Countries with a Democracy Score: ' + str(df['polityscore'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['polityscore'].count()) + ' missing)')
print('\n')

# Get the rows not missing a value
subset = df[np.isfinite(df['polityscore'])]
subset = subset[np.isfinite(subset['incomeperperson'])]
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('\n')

# These functions converts the polity score to a binary category flag
def is_full_democracy(score):
    if score == 10:
        return(1)
    else:
        return(0)

def is_full_democracy_text(score):
    if score == 10:
        return('Yes')
    else:
        return('No') 
		
# Now we can use the function to create the new variable
subset['is_full_democracy'] = subset['polityscore'].apply(is_full_democracy)
subset['is_full_democracy_text'] = subset['polityscore'].apply(is_full_democracy_text).astype('category')

# Visualize data using a boxplot
sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.boxplot(x='is_full_democracy_text', y='incomeperperson', data=subset)
plt.ylabel('Economic Well-Being (GDP Per Person)')
plt.xlabel('Is a Full Democracy')

# Create simple regression model
model = smf.ols('incomeperperson ~ is_full_democracy', data=subset).fit()
print(model.summary())

