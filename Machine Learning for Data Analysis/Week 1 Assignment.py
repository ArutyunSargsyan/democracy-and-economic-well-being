# -*- coding: utf-8 -*-
"""
Week 1 Assignment

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
#import matplotlib.pylab as plt
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

from io import StringIO
import pydotplus as pdp

# Set random number seed to make results reproducible
np.random.seed(0)

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
print("\n")

# Identify contries with a high level of income using the MAD (mean absolute deviation) method
subset['absolute_deviations'] = np.absolute(subset['incomeperperson'] - np.median(subset['incomeperperson']))
MAD = np.mean(subset['absolute_deviations'])

# This function converts the income per person absolute deviations to a high income flag
def high_income_flag(absolute_deviations):
    threshold = 3
    if (absolute_deviations/MAD) > threshold:
        return "Yes"
    else:
        return "No"
        
subset['high_income'] = subset['absolute_deviations'].apply(high_income_flag)
subset['high_income'] = subset['high_income'].astype('category')

# This function converts the polity score to a category
def convert_polityscore_to_category(polityscore):
    if polityscore == 10:
        return 1
    else:
        return 0

# Now we can use the function to create the new variable
subset['full_democracy'] = subset['polityscore'].apply(convert_polityscore_to_category)
subset['full_democracy'] = subset['full_democracy'].astype('category')

"""
Modeling and Prediction
"""
#Split into training and testing sets
predictors = subset[['full_democracy','urbanrate']]
targets = subset.high_income
training_data, test_data, training_target, test_target  = train_test_split(predictors, targets, test_size=.4)

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(training_data, training_target)

# Check how well the classifier worked
predictions=classifier.predict(test_data)

print('Confusion Matrix:')
print(sk.metrics.confusion_matrix(test_target,predictions))
print("\n")

print('Accuracy Score:')
print(sk.metrics.accuracy_score(test_target, predictions))
print("\n")

#Displaying the decision tree
out = StringIO()
sk.tree.export_graphviz(classifier, out_file=out)
graph=pdp.graph_from_dot_data(out.getvalue())