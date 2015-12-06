# -*- coding: utf-8 -*-
"""
Week 2 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

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

# This function converts the polity score to a binary category flag
def is_full_democracy(score):
    if score == 10:
        return(1)
    else:
        return(0)
		
# Now we can use the function to create the new variable
subset['is_full_democracy'] = subset['polityscore'].apply(is_full_democracy)

# Create frequency table
full_democracy_counts = subset.groupby('is_full_democracy').size()
print(full_democracy_counts)
print('\n')

# Create simple regression model
model = smf.ols('incomeperperson ~ is_full_democracy', data=subset).fit()
print(model.summary())

