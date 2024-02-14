import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import calmap
from pandas_profiling import ProfileReport

# Task 1: Initial data exploration
# Read in the dataset and it's convention to name the dataframe "df"
df = pd.read_csv('supermarket_sales.csv')

# View the first five rows of dataset
df.head()

# View columns individually
df.columns

# View data types of each column
df.dtypes

# Change 'Date' column from type object to type datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convention to set date as index for dataframe
df.set_index('Date',inplace=True)

# View that the index change is a permanent change
df.head()

# View quick summary statistics
df.describe()

# Task 2: Univariate Analysis
# Question 1: What does the distribution of customer ratings looks like? Is it skewed?
sns.displot(df['Rating'])

# Input mean line
plt.axvline(x=np.mean(df['Rating']),c='red',ls='--',label='mean')

# Plot 25th and 75th percentile
plt.axvline(x=np.percentile(df['Rating'],25),c='green',ls='--',label='25-75th percentile')
plt.axvline(x=np.percentile(df['Rating'],75),c='green',ls='--')
plt.legend()

# View all numerical variables and their following distributions
df.hist(figsize=(10,10))

# Question 2: Do aggregate sales numbers differ by much between branches
# View sales by different branches
sns.countplot(x='Branch',data=df)

# To get the exact numbers
df['Branch'].value_counts()

# View users who use certain payment method versus other methods
## If you want to visualize the distribution of a categorical variable, use sns.countplot(). If you want to visualize the distribution of a continuous variable, use a histogram, which can be created using various functions including plt.hist() or sns.histplot().
sns.countplot(x='Payment',data=df)

# Task 3: Bivariate Analysis
# Question 3: Is there a relationship between gross income and customer ratings?
# use sns.regplot() when you want to explore the relationship between two continuous variables and visualize a linear regression fit
sns.regplot(df['Rating'],df['gross income'])

# See if there is a relationship between Branch and gross income.
# use sns.boxplot() when you want to visualize the distribution of a continuous variable across different categories or compare the distributions of multiple variables
sns.boxplot(x=df['Branch'],y=df['gross income'])

# See if there is a relationship between gender and gross income.
sns.boxplot(x=df['Gender'],y=df['gross income'])


# Question 4: Is there a noticeable time trend in gross income?
# As index is the date, there are multiple values. Hence use groupby to get individual dates.
df.groupby(df.index).mean()
sns.lineplot(x=df.groupby(df.index).mean().index, y=df.groupby(df.index).mean()['gross income'])

# Plot all the bivariate relationships possible
# Pairplot creates a grid of scatterplots for the numerical columns in a DataFrame df. This function is especially useful for visualizing pairwise relationships between multiple variables in a dataset.
sns.pairplot(df)

# Task 4: Dealing with duplicate rows and missing values
# Use duplicated function to get an index and a boolean which is either false or true if that particular row is duplicated. Add .sum() to get the total count.
# df.duplicated() for identifying duplicate rows.
df.duplicated().sum()

# See rows that are duplicated
df[df.duplicated()==True]

# Remove those duplicated rows
df.drop_duplicates(inplace=True)

# See the total number of missing values per column
# df.isna() for handling missing values in columns
df.isna().sum()
# Ratio of missing values divide that by the length of the dataframe.
df.isna().sum()/len(df)

# Seaborn's heat map function - see the index on left side and all the values and these white lines are the values that are missing
sns.heatmap(df.isnull(),cbar=False)

# Fill each missing value with 0 or mean
df.fillna(df.mean(),inplace=True)

# However, non-numerical values do not have a mean, hence have to fill it with mode
df.fillna(df.mode().iloc[0],inplace=True)

# Pandas profiler because it organizes EDA succinctly, shows the number of variables, observations, missing cells duplicate rows
# caveat for using Pandas profiler is if you have a large data set, this is not going to be feasible.
dataset = pd.read_csv('supermarket_sales.csv')
prof = ProfileReport(dataset)
prof

# Task 5: Correlation Analysis
# [1][0] is to get the specific correlation number, otherwise it is a 2x2 array
round(np.corrcoef(df['gross income'],df['Rating'])[1][0],2)

# as it is tedious to go through every combination, can use correlation matrix to see every numerical column pair
np.round(df.corr(),2)

# Visual way to look at it
sns.heatmap(np.round(df.corr(),2),annot=True)
