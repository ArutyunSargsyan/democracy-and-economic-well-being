# -*- coding: utf-8 -*-
"""
Week 3 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import scipy
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
df['urbanrate'] = pd.to_numeric(df['urbanrate'], errors='coerce')
df['incomeperperson'] = pd.to_numeric(df['incomeperperson'], errors='coerce')
# Print out the counts of valid and missing rows
print ('Countries with an Urbanization Rate: ' + str(df['urbanrate'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['urbanrate'].count()) + ' missing)')
print ('Countries with a GDP Per Capita: ' + str(df['incomeperperson'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['incomeperperson'].count()) + ' missing)')
print('\n')

# Get the subset of complete data cases
print('Dropping rows with missing urbanization rate and per capita GDP')
subset = df[['urbanrate','incomeperperson']].dropna()
# Print more statistics
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('\n')

# Pearson's Correlation Coefficient
print ('Association Between Urbanization Rate and Economic Well-Being')
r = scipy.stats.pearsonr(subset['urbanrate'], subset['incomeperperson'])
print (r)
r_squared = r[0] * r[0]
print('R Squared = '+str(r_squared))

# Visualize the data
sns.set_context('poster')
plt.figure(figsize=(14, 7))
sns.regplot(x="urbanrate", y="incomeperperson", data=subset)
plt.ylabel('Economic Well-Being (GDP Per Person)')
plt.xlabel('Urbanization Rate')