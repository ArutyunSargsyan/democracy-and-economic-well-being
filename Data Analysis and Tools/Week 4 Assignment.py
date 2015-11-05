# -*- coding: utf-8 -*-
"""
Week 4 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as ssm
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
df['polityscore'] = pd.to_numeric(df['polityscore'], errors='coerce')
df['urbanrate'] = pd.to_numeric(df['urbanrate'], errors='coerce')
df['incomeperperson'] = pd.to_numeric(df['incomeperperson'], errors='coerce')
 
# Print out the counts of valid and missing rows
print ('Countries with a Democracy Score: ' + str(df['polityscore'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['polityscore'].count()) + ' missing)')
print ('Countries with an Urbanization Rate: ' + str(df['urbanrate'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['urbanrate'].count()) + ' missing)')
print ('Countries with a GDP Per Capita: ' + str(df['incomeperperson'].count()) + ' out of ' + str(len(df)) + ' (' + str(len(df) - df['incomeperperson'].count()) + ' missing)')
print('\n')

# Get the subset of complete data cases
print('Dropping rows with missing urbanization rate, democracy score or per capita GDP')
subset = df[['polityscore','urbanrate','incomeperperson']].dropna()
# Print more statistics
print('Number of observations: '+ str(len(subset)) +' (rows)')
print('\n')
"""
# Change income per person from float to int
df['incomeperperson'] = df['incomeperperson'].astype(int)
"""
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
subset['openness'] = subset['polityscore'].apply(convert_polityscore_to_category).astype('category')

# Recode urbanization rate
print('Creating urbanization type variable')
# This function converts the polity score to a category
def convert_urbanrate_to_category(urbanrate):
    if urbanrate < 33:
        return('Not Urban')
    elif urbanrate < 66:
        return('In Transition')
    else:
        return('Urban')

# Now we can use the function to create the new variable
subset['urban'] = subset['urbanrate'].apply(convert_urbanrate_to_category).astype('category')

datasets = {'Not Urban': subset[subset['urban']=='Not Urban'], 'In Transition': subset[subset['urban']=='In Transition'], 'Urban' : subset[subset['urban']=='Urban']}

for title, data in datasets.iteritems():
    # ANOVA
    ols_data = data[['incomeperperson','openness']].dropna()
    # Convert float to int
    #ols_data['incomeperperson'] = ols_data['incomeperperson'].astype(int)
    model = smf.ols(formula='incomeperperson ~ C(openness)', data=ols_data).fit()
    print('ANOVA for Countries Classified as '+title)    
    print(model.summary())
    print('\n')
    
    # Tukey Honestly Significantly Different
    tukey = ssm.MultiComparison(data['incomeperperson'], data['openness']).tukeyhsd()
    print('Tukey Honestly Significantly Different Test for Countries Classified as '+title)
    print(tukey.summary())
    print('\n')
    
    # Economic Well-Being by Level of Openness
    # Visualize Mean Economic Well Being by Level of Openness
    sns.set_context('poster')
    plt.figure(figsize=(14, 7))
    sns.factorplot(x='openness', y='incomeperperson', data=ols_data, kind='bar', ci=None, size=4, aspect=4)
    plt.ylabel('Average Economic Well-Being (GDP Per Person)')
    plt.xlabel('Level of Openness')
    plt.title(title)