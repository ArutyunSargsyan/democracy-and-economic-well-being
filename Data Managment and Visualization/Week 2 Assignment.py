# -*- coding: utf-8 -*-
"""
Week 2 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np

# Read in the Data
df = pd.read_csv('gapminder.csv', low_memory=False)

# Print some basic statistics
n = str(len(df))
cols = str(len(df.columns))
print('Number of observations: '+ n +' (rows)')
print('Number of variables: '+ cols +' (columns)')
print('\n')

# Change the data type for variables of interest
df['polityscore'] = df['polityscore'].convert_objects(convert_numeric=True)
df['incomeperperson'] = df['incomeperperson'].convert_objects(convert_numeric=True)

print ('Countries with a Democracy Score: ' + str(df['polityscore'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['polityscore'].count()) + ' missing)')
print ('Countries with a GDP Per Capita: ' + str(df['incomeperperson'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['incomeperperson'].count()) + ' missing)')
print('\n')

# Get the rows not missing a value
subset = df[np.isfinite(df['polityscore'])]
subset = subset[np.isfinite(subset['incomeperperson'])]
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('\n')

# This function converts the polity score to a category
def convert_polityscore_to_category(score):
    if score == 10:
        return('1 - Full Democracy')
    elif score > 5:
        return('2 - Democracy')
    elif score > 0:
        return ('3 - Open Anocracy')
    elif score > -6:
        return ('4 - Closed Anocracy')
    else:
        return('5 - Autocracy')

# Now we can use the function to create the new variable
subset['democracy'] = subset['polityscore'].apply(convert_polityscore_to_category)

# Create income quintiles
subset['incomequintiles'] = pd.cut(subset['incomeperperson'], 5, labels=['Lowest','Second','Middle','Fourth','Highest'])

print('Countries by Democracy Score (-10=autocracy & 10=full democracy)')
polity_counts = subset.groupby('polityscore').size()
print(polity_counts)
print('\n')

print('Percent of Countries by Democracy Score')
polity_percents = polity_counts * 100 / len(subset)
print(polity_percents)
print('\n')

greater_than_zero =subset[subset['polityscore'] > 0]
greater_than_zero_percent = len(greater_than_zero) * 100 / len(subset)
print('Number of countries with a Polity score greater than zero: ' + str(len(greater_than_zero)))
print('Percent of countries with a Polity score greater than zero: ' + str(greater_than_zero_percent) + '%')
print('\n')

print('Countries by Per Capita GDP Quintiles')
incomequintiles_counts = subset.groupby('incomequintiles').size()
print(incomequintiles_counts)
print('\n')

print('Percent of Countries by Per Capita Quintiles')
incomequintiles_percents = incomequintiles_counts * 100 / len(subset)
print(incomequintiles_percents)
print('\n')

print('Countries by Democracy Category')
democracy_counts = subset.groupby('democracy').size()
print(democracy_counts)
print('\n')

print('Percent of Countries by Democracy Category')
democracy_percents = democracy_counts * 100 / len(subset)
print(democracy_percents)
