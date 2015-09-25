# -*- coding: utf-8 -*-
"""
Week 3 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np

# Read in the GapMinder Data
print('Reading in GapMinder data')
df = pd.read_csv('gapminder.csv', low_memory=False)
# Print some basic statistics about the GapMinder data
print('Number of observations: '+ str(len(df)) +' (rows)')
print('Number of variables: '+ str(len(df.columns)) +' (columns)')
print('\n')

# Read in my Continent Data Set
print('Reading in continent data')
continents = pd.read_csv('gapminder-continents.csv', low_memory=False)
# Print some basic statistics about the continents
print('Number of observations: '+ str(len(continents)) +' (rows)')
print('Number of variables: '+ str(len(continents.columns)) +' (columns)')
print('\n')

# Merge the two dataframes
print('Merging GapMinder and continent data')
df = pd.merge(df, continents, left_on='country', right_on='country')
# Print some updated basic statistics
print('Number of observations: '+ str(len(df)) +' (rows)')
print('Number of variables: '+ str(len(df.columns)) +' (columns)')
print('\n')

# Change the data type for variables of interest
df['polityscore'] = df['polityscore'].convert_objects(convert_numeric=True)
df['incomeperperson'] = df['incomeperperson'].convert_objects(convert_numeric=True)
# Print out the counts of valid and missing rows
print ('Countries with a Democracy Score: ' + str(df['polityscore'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['polityscore'].count()) + ' missing)')
print ('Countries with a GDP Per Capita: ' + str(df['incomeperperson'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['incomeperperson'].count()) + ' missing)')
print('\n')

# Get the rows not missing a value
print('Dropping rows with missing democracy score and per capita GDP')
subset = df[np.isfinite(df['polityscore'])]
subset = subset[np.isfinite(subset['incomeperperson'])]
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('Number of variables: '+ str(len(subset.columns)) +' (columns)')
print('\n')

# Drop unneeded columns
print('Dropping unneeded variables')
drops = ['country', 'alcconsumption', 'armedforcesrate', 'breastcancerper100th', 'co2emissions', 'femaleemployrate', 'hivrate', 'internetuserate', 'lifeexpectancy', 'oilperperson', 'relectricperperson', 'suicideper100th', 'employrate', 'urbanrate']
df.drop(drops,inplace=True,axis=1)
# Print more statistics
print('Number of observations: '+ str(len(df)) +' (rows)')
print('Number of variables: '+ str(len(df.columns)) +' (columns)')
print('\n')

# Recode polity score
print('Creating government type variable')
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

# Create per capita GDP quintiles
print('Creating GDP per capita quintiles')
subset['incomequintiles'] = pd.cut(subset['incomeperperson'], 5, labels=['Lowest','Second','Middle','Fourth','Highest'])
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('Number of variables: '+ str(len(subset.columns)) +' (columns)')
print('\n')


# Create crosstabs
ct1 = pd.crosstab(subset['incomequintiles'], subset['democracy'])
print('The Number of Countries by Income Quintile and Level of Democracy')
print(ct1)
print('\n')

pt1 = (pd.crosstab(subset['incomequintiles'], subset['democracy'])/len(subset))*100
print('The Percent of Countries by Income Quintile and Level of Democracy')
print(pt1)
print('\n')

ct2 = pd.crosstab(subset['continent'], subset['democracy'])
print('The Number of Countries by Continent and Level of Democracy')
print(ct2)
print('\n')