import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import calmap
from pandas_profiling import ProfileReport

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


