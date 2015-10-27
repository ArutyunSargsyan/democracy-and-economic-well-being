# -*- coding: utf-8 -*-
"""
Week 2 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

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
subset['openness'] = subset['polityscore'].apply(convert_polityscore_to_category)
subset['openness'] = subset['openness'].astype('category')

# Visualize the data
sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.countplot(x='openness', data=subset)
plt.ylabel('Count')
plt.xlabel('Level of Openness')

openness_counts = subset.groupby('openness').size()
print('The Number of Countries by Openness')
print(openness_counts)
print('\n')

# Create per capita GDP quartiles
print('Creating GDP per capita quartiles')
subset['incomequartiles'] = pd.cut(subset['incomeperperson'], 4, labels=['1 -  0% to 25%','2 - 25% to 50%','3 - 50% to 75%','4 - 75% to 100%'])
subset['incomequartiles'] = subset['incomequartiles'].astype('category')
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('Number of variables: '+ str(len(subset.columns)) +' (columns)')
print('\n')

# Quartile bar chart
sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.countplot(x='incomequartiles', data=subset)
plt.xlabel('Economic Well-Being (GDP Per Person) Quartile')
plt.ylabel('Count')

income_quartile_counts = subset.groupby('incomequartiles').size()
print('The Number of Countries by Income Quartile')
print(income_quartile_counts)
print('\n')

subset['top half'] = pd.cut(subset['incomeperperson'], 4, labels=['1','2','3','4'])
recode_map = {'1':'No','2':'No','3':'Yes','4':'Yes'}
subset['top half'] = subset['top half'].map(recode_map).astype('category')
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('Number of variables: '+ str(len(subset.columns)) +' (columns)')
print('\n')

# Chi Squared Test of Independance

crosstab = pd.crosstab(subset['top half'],subset['openness'])
print(crosstab)
print('\n')

column_sums = crosstab.sum(axis=0)
column_percents = crosstab / column_sums
print(column_percents)
print('\n')

chi2 = scipy.stats.chi2_contingency(crosstab)
print(chi2)
print('\n')

# Post-Hoc Chi Squared
p = 0.5
c = 10 # Number of comparisons
bonferroni = p/c

def bonferroni_test(pvalue, bonferroni):
    if pvalue > bonferroni:
        print('accept h0')
    else:
        print('reject h0')

def print_percent_crosstab(crosstab):
	column_sums = crosstab.sum(axis=0)
	column_percents = crosstab / column_sums
	print(column_percents)
	
# 1: 1 - Full Democracy vs 2 - Democracy
subset['1v2'] = subset['openness'].map({'1 - Full Democracy':'1 - Full Democracy', '2 - Democracy':'2 - Democracy'})
crosstab_1v2 = pd.crosstab(subset['top half'],subset['1v2'])
chi2_1v2 = scipy.stats.chi2_contingency(crosstab_1v2)
print('1 - Full Democracy vs 2 - Democracy')
print(crosstab_1v2)
print('\n')
print_percent_crosstab(crosstab_1v2)
print('\n')
print(chi2_1v2)
bonferroni_test(chi2_1v2[1], bonferroni)
print('\n')

# 2: 1 - Full Democracy vs 3 - Open Anocracy
subset['1v3'] = subset['openness'].map({'1 - Full Democracy':'1 - Full Democracy', '3 - Open Anocracy':'3 - Open Anocracy'})
crosstab_1v3 = pd.crosstab(subset['top half'],subset['1v3'])
chi2_1v3 = scipy.stats.chi2_contingency(crosstab_1v3)
print('1 - Full Democracy vs 3 - Open Anocracy')
print(crosstab_1v3)
print('\n')
print_percent_crosstab(crosstab_1v3)
print('\n')
print(chi2_1v3)
bonferroni_test(chi2_1v3[1], bonferroni)
print('\n')

# 3: 1 - Full Democracy vs 4 - Closed Anocracy
subset['1v4'] = subset['openness'].map({'1 - Full Democracy':'1 - Full Democracy', '4 - Closed Anocracy':'4 - Closed Anocracy'})
crosstab_1v4 = pd.crosstab(subset['top half'],subset['1v4'])
chi2_1v4 = scipy.stats.chi2_contingency(crosstab_1v4)
print('1 - Full Democracy vs 4 - Closed Anocracy')
print(crosstab_1v4)
print('\n')
print_percent_crosstab(crosstab_1v4)
print('\n')
print(chi2_1v4)
bonferroni_test(chi2_1v4[1], bonferroni)
print('\n')

# 4: 1 - Full Democracy vs 5 - Autocracy
subset['1v5'] = subset['openness'].map({'1 - Full Democracy':'1 - Full Democracy', '5 - Autocracy':'5 - Autocracy'})
crosstab_1v5 = pd.crosstab(subset['top half'],subset['1v5'])
chi2_1v5 = scipy.stats.chi2_contingency(crosstab_1v5)
print('1 - Full Democracy vs 5 - Autocracy')
print(crosstab_1v5)
print('\n')
print_percent_crosstab(crosstab_1v5)
print('\n')
print(chi2_1v5)
bonferroni_test(chi2_1v5[1], bonferroni)
print('\n')

# 5: 2 - Democracy vs 3 - Open Anocracy
subset['2v3'] = subset['openness'].map({'2 - Democracy':'2 - Democracy', '3 - Open Anocracy':'3 - Open Anocracy'})
crosstab_2v3 = pd.crosstab(subset['top half'],subset['2v3'])
chi2_2v3 = scipy.stats.chi2_contingency(crosstab_2v3)
print('2 - Democracy vs 3 - Open Anocracy')
print(crosstab_2v3)
print('\n')
print_percent_crosstab(crosstab_2v3)
print('\n')
print(chi2_2v3)
bonferroni_test(chi2_2v3[1], bonferroni)
print('\n')

# 6: 2 - Democracy vs 4 - Closed Anocracy
subset['2v4'] = subset['openness'].map({'2 - Democracy':'2 - Democracy', '4 - Closed Anocracy':'4 - Closed Anocracy'})
crosstab_2v4 = pd.crosstab(subset['top half'],subset['2v4'])
chi2_2v4 = scipy.stats.chi2_contingency(crosstab_2v4)
print('2 - Democracy vs 4 - Closed Anocracy')
print(crosstab_2v4)
print('\n')
print_percent_crosstab(crosstab_2v4)
print('\n')
print(chi2_2v4)
bonferroni_test(chi2_2v4[1], bonferroni)
print('\n')

# 7: 2 - Democracy vs 5 - Autocracy
subset['2v5'] = subset['openness'].map({'2 - Democracy':'2 - Democracy', '5 - Autocracy':'5 - Autocracy'})
crosstab_2v5 = pd.crosstab(subset['top half'],subset['2v5'])
chi2_2v5 = scipy.stats.chi2_contingency(crosstab_2v5)
print('2 - Democracy vs 5 - Autocracy')
print(crosstab_2v5)
print('\n')
print_percent_crosstab(crosstab_2v5)
print('\n')
print(chi2_2v5)
bonferroni_test(chi2_2v5[1], bonferroni)
print('\n')

# 8: 3 - Open Anocracy vs 4 - Closed Anocracy
subset['3v4'] = subset['openness'].map({'3 - Open Anocracy':'3 - Open Anocracy', '4 - Closed Anocracy':'4 - Closed Anocracy'})
crosstab_3v4 = pd.crosstab(subset['top half'],subset['3v4'])
chi2_3v4 = scipy.stats.chi2_contingency(crosstab_3v4)
print('3 - Open Anocracy vs 4 - Closed Anocracy')
print(crosstab_3v4)
print('\n')
print_percent_crosstab(crosstab_3v4)
print('\n')
print(chi2_3v4)
bonferroni_test(chi2_3v4[1], bonferroni)
print('\n')

# 9: 3 - Open Anocracy vs 5 - Autocracy
subset['3v5'] = subset['openness'].map({'3 - Open Anocracy':'3 - Open Anocracy', '5 - Autocracy':'5 - Autocracy'})
crosstab_3v5 = pd.crosstab(subset['top half'],subset['3v5'])
chi2_3v5 = scipy.stats.chi2_contingency(crosstab_3v5)
print('3 - Open Anocracy vs 5 - Autocracy')
print(crosstab_3v5)
print('\n')
print_percent_crosstab(crosstab_3v5)
print('\n')
print(chi2_3v5)
bonferroni_test(chi2_3v5[1], bonferroni)
print('\n')

# 10: 4 - Closed Anocracy vs 5 - Autocracy
subset['4v5'] = subset['openness'].map({'4 - Closed Anocracy':'4 - Closed Anocracy', '5 - Autocracy':'5 - Autocracy'})
crosstab_4v5 = pd.crosstab(subset['top half'],subset['4v5'])
chi2_4v5 = scipy.stats.chi2_contingency(crosstab_4v5)
print('4 - Closed Anocracy vs 5 - Autocracy')
print(crosstab_4v5)
print('\n')
print_percent_crosstab(crosstab_4v5)
print('\n')
print(chi2_4v5)
bonferroni_test(chi2_4v5[1], bonferroni)
print('\n')