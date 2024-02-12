#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


import calmap
from pandas_profiling import ProfileReport


# In[20]:


df = pd.read_csv('supermarket_sales.csv')


# In[21]:


df.head()


# In[22]:


df.columns


# In[23]:


df.dtypes


# In[24]:


df['Date'] = pd.to_datetime(df['Date'])


# In[25]:


# Convention to set date as index for dataframe
df.set_index('Date',inplace=True)


# In[26]:


df.head()


# In[27]:


df.describe()


# In[ ]:




