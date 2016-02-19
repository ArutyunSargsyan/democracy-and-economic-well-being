# -*- coding: utf-8 -*-
"""
Machine Learning for Data Analysis
Week 4 Assignment
K Means Clustering

Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Make results reproducible
np.random.seed(1234567890)

df = pd.read_csv('gapminder.csv')

variables = ['incomeperperson', 'polityscore', 'internetuserate', 'lifeexpectancy','urbanrate']

# convert to numeric format
for variable in variables:
    df[variable] = pd.to_numeric(df[variable], errors='coerce')

# listwise deletion of missing values
subset = df[variables].dropna()

# Print the rows and columns of the data frame
print('Size of study data')
print(subset.shape)
print("\n")
"""
" =============================  Data Management  =============================
"""
# Remove the first variable from the list since the target is derived from it
variables.pop(0)
variables.pop(0)

# Center and scale data
for variable in variables:
    subset[variable]=preprocessing.scale(subset[variable].astype('float64'))

"""
" ==================  Split Data into Traingin and Test Sets  ==================
"""
features = subset[variables]
targets = subset[['incomeperperson', 'polityscore']]

# Split into training and testing sets
training_data, test_data, training_target, test_target  = train_test_split(features, targets, test_size=.3)

# Identify number of clusters using the elbow method
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(training_data)
    clusassign=model.predict(training_data)
    meandist.append(sum(np.min(cdist(training_data, model.cluster_centers_, 'euclidean'), axis=1)) / training_data.shape[0])

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
