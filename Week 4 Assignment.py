# -*- coding: utf-8 -*-
"""
Week 4 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *

# Read in the GapMinder Data
print('Reading in GapMinder data')
df = pd.read_csv('gapminder.csv', low_memory=False)
# Print some basic statistics about the GapMinder data
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
subset.drop(drops,inplace=True,axis=1)
# Print more statistics
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('Number of variables: '+ str(len(subset.columns)) +' (columns)')
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
subset['democracy'] = subset['democracy'].astype('category')

# Create per capita GDP quartiles
print('Creating GDP per capita quartiles')
subset['incomequartiles'] = pd.cut(subset['incomeperperson'], 4, labels=['1 -  0% to 25%','2 - 25% to 50%','3 - 50% to 75%','4 - 75% to 100%'])
subset['incomequartiles'] = subset['incomequartiles'].astype('category')
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('Number of variables: '+ str(len(subset.columns)) +' (columns)')
print('\n')

# Exploratory Analysis
# Univariate Analysis
# Economic Well-Being
sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.distplot(subset['incomeperperson'])
plt.xlabel('Economic Well Being (GDP Per Person)')

sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.countplot(x='incomequartiles', data=subset)
plt.xlabel('Economic Well Being (GDP Per Person) Quartile')
plt.ylabel('Count')

income_quartile_counts = subset.groupby('incomequartiles').size()
print('The Number of Countries by Income Quartile')
print(income_quartile_counts)
print('\n')

# Level of Openness
sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.countplot(x='democracy', data=subset)
plt.ylabel('Count')
plt.xlabel('Level of Democratization')

openness_counts = subset.groupby('democracy').size()
print('The Number of Countries by Openness')
print(openness_counts)
print('\n')

# Economic Well-Being by Level of Democracy

# GDP Per Capita Quartiles by Level of Democracy
ct3 = pd.crosstab(subset['incomequartiles'], subset['democracy'])
print('The Number of Countries by Income Quartile and Level of Democracy')
print(ct3)
print('\n')

pt3 = (ct3/len(subset))*100
print('The Percent of Countries by Income Quartile and Level of Democracy')
print(pt3)
print('\n')

# Median Well-Being by Level of Democracy
print('GDP Statistics by Level of Democracy')
gdp_mean2 = subset.groupby('democracy')['incomeperperson'].agg([np.mean, np.std, np.median, len])
print(gdp_mean2)

# Visualize data using a boxplot
sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.boxplot(x="democracy", y="incomeperperson", data=subset)
plt.ylabel('Economic Well Being (GDP Per Person)')
plt.xlabel('Level of Democracy')

#plot = ggplot(subset, aes(x = 'incomeperperson', fill='democracy')) + geom_dotplot(binwidth=100)
#print(plot)