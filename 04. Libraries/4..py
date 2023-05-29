#!/usr/bin/env python
# coding: utf-8

# **Loading Libraries**

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


# In[2]:


ipl=pd.read_csv('/Users/apple/Desktop/data.csv')


# In[3]:


ipl.head(10)


# In[4]:


ipl.shape


# In[5]:


ipl['player_of_match'].value_counts()


# In[6]:


ipl['player_of_match'].value_counts()[0:10]


# In[7]:


print(ipl.info())


# In[8]:


ipl['toss_winner'].value_counts()


# In[9]:


batting_first=ipl[ipl['win_by_runs']!=0]


# In[10]:


batting_first.head()


# In[11]:


plt.figure(figsize=(3,3))
plt.hist(batting_first['win_by_runs'])
plt.title("Distribution of Runs")
plt.xlabel("Runs")
plt.show()


# In[ ]:


batting_first['winner'].value_counts()


# In[ ]:


batting_second=ipl[ipl['win_by_wickets']!=0]


# In[ ]:


batting_second.head()


# In[ ]:


plt.figure(figsize=(5,7))
plt.hist(batting_second['win_by_wickets'],bins=30)
plt.show()


# In[ ]:


batting_second['winner'].value_counts()


# In[ ]:


# Number of matches played at each season
ipl['season'].value_counts()


# In[ ]:


# Number of matches played in each city
ipl['city'].value_counts()


# In[ ]:


# How many times a team won the match after winning toss

np.sum(ipl['toss_winner']==ipl['winner'])


# In[ ]:


291/577


# In[13]:


ipl.isna().sum()


# In[ ]:


# ipl.dropna(inplace=True)


# In[ ]:


# ipl.shape


# In[14]:


ipl = ipl.drop('umpire3', axis=1)


# In[15]:


ipl.shape


# In[16]:


ipl.isna().sum()


# In[17]:


ipl.dropna(inplace=True)


# In[18]:


ipl.shape


# In[19]:


ipl.describe()


# **Are there Outliers present in the Data?**

# In[ ]:


ipl.boxplot(column=['win_by_wickets'],figsize=(12, 7));

