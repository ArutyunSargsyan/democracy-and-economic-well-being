# -*- coding: utf-8 -*-
"""
Week 3 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# Create per capita GDP quintiles
print('Creating GDP per capita quintiles')
subset['incomequintiles'] = pd.cut(subset['incomeperperson'], 5, labels=['1 - Lowest  (0% to 20%)','2 - Second  (20% to 40%)','3 - Middle  (40% to 60%)','4 - Fourth  (60% to 80%)','5 - Highest (80% to 100%)'])
subset['incomequintiles'] = subset['incomequintiles'].astype('category')

# Create per capita GDP quartiles
print('Creating GDP per capita quartiles')
subset['incomequartiles'] = pd.cut(subset['incomeperperson'], 4, labels=['1 -  0% to 25%','2 - 25% to 50%','3 - 50% to 75%','4 - 75% to 100%'])
subset['incomequartiles'] = subset['incomequartiles'].astype('category')
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('Number of variables: '+ str(len(subset.columns)) +' (columns)')
print('\n')

# Exploratory Analysis
# Relationship between Continent and Level of Democracy/Economic Well-Being
ct1 = pd.crosstab(subset['continent'], subset['democracy'])
print('The Number of Countries by Continent and Level of Democracy')
print(ct1)
print('\n')

pt1 = (ct1/len(subset))*100
print('The Percent of Countries by Continent and Level of Democracy')
print(pt1)
print('\n')

print('Countries by Continent')
country_counts = subset.groupby('continent').size()
print(country_counts)
print('\n')

print('Democracies by Continent')
subset2 = subset[(subset['democracy'] == '1 - Full Democracy') | (subset['democracy'] == '2 - Democracy')]
democracy_counts = subset2.groupby('continent').size()
print(democracy_counts)
print('\n')

print('Percent of Democracies by Continent')
democracy_percent = (democracy_counts / country_counts)*100
print(democracy_percent)
print('\n')

print('GDP Statistics by Continent')
gdp_mean = subset.groupby('continent')['incomeperperson'].agg([np.mean, np.std, np.median, len])
print(gdp_mean)
print('\n')

# Economic Well-Being by Level of Democracy

# Approach 1: GDP Per Capita Quintiles by Level of Democracy
ct2 = pd.crosstab(subset['incomequintiles'], subset['democracy'])
print('The Number of Countries by Income Quintile and Level of Democracy')
print(ct2)
print('\n')

pt2 = (ct2/len(subset))*100
print('The Percent of Countries by Income Quintile and Level of Democracy')
print(pt2)
print('\n')

# Approach 2: GDP Per Capita Quartiles by Level of Democracy
ct3 = pd.crosstab(subset['incomequartiles'], subset['democracy'])
print('The Number of Countries by Income Quartile and Level of Democracy')
print(ct3)
print('\n')

pt3 = (ct3/len(subset))*100
print('The Percent of Countries by Income Quartile and Level of Democracy')
print(pt3)
print('\n')

# Approach Number 3: Average Well-Being by Level of Democracy
print('GDP Statistics by Level of Democracy')
gdp_mean2 = subset.groupby('democracy')['incomeperperson'].agg([np.mean, np.std, np.median, len])
print(gdp_mean2)

# Visualize data using a boxplot
sns.set_style("whitegrid")
sns.set_context("poster")
plt.figure(figsize=(14, 7))
sns.boxplot(x="democracy", y="incomeperperson", data=subset)
plt.ylabel('Economic Well Being (GDP Per Person)')
plt.xlabel('Level of Democracy')