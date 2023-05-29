#!/usr/bin/env python
# coding: utf-8

# Content Credit: https://www.kaggle.com/learn/pandas

# # All About Pandas Library

# 
# *   Pandas is a library generally used for data manipulation 
# *   Pandas is used to handle tabular data.
# 
# There are two sorts of data structures in Pandas –
# 
# 
# 1.   Series
# 2.   Dataframes
# 
# 
# 
# 
# 
# 

# In[1]:


# import pandas

import pandas as pd


# ## 1. Creating DataFrame

# In[2]:


# Data Frame ==> Table

pd.DataFrame({'Y':[5,1], 'N':[10,2]})


# In[3]:


pd.DataFrame({'Person 1': ['Nice', 'Bad'], 'Person 2': ['Damn Good.', 'worst']})


# In[4]:


# Changing default Index to user defined index

pd.DataFrame({'Person 1': ['Nice', 'Bad'], 'Person 2': ['Damn Good.', 'worst']}, index=['A','B'])


# ## Series

# In[5]:


# A series is a List
# It is a single column of a Dataframe
# Series doesn't have a column name
# A Dataframe is a bunch of series

pd.Series([1,2,3,4,5,6,7,8,9])


# In[6]:


# Changing default Index to user defined index

pd.Series([10,20,30,40,50], index=['A','B','C','D','E'])


# In[7]:


# Reading CSV file

# way 1
Data1=pd.read_csv("/Users/apple/Desktop/SampleFile.csv")

# way 2: Mount google-drive

#from google.colab import drive
#drive.mount('/content/gdrive')
#DATA_PATH = 'gdrive/My Drive/'


# In[8]:


# Shape Attribute to find the size of Dataframe

Data1.shape


# In[9]:


# head command to see the content of DataFrame

Data1.head()


# In above result we can see that even if CSV file has inbuilt index, still pandas does not pick up that automatically. It adds its own index.

# In[10]:


# To make pandas use inbuilt CSV index column, we can set index column to 0

Data1=pd.read_csv("/Users/apple/Desktop/SampleFile.csv", index_col=0)


# In[11]:


Data1.head()


# ### DataFrame Series Access

# These are the two ways of selecting a specific Series 
# 
# *   .SeriesName
# *   ['SeriesName']
# 
# Here, second way has the advantage over First. If our series name would have been "First Name" ( Means with space), we can not handle that using first approach. 
# 
# 
# 

# In[12]:


# Access First_name of Data1 Dataframe

Data1.First_name


# In[13]:


# In case of python Dictionary above details can be accessed in following way as well

Data1['First_name']


# In[14]:


# Accessing specific value of a series 

Data1['First_name'][1]


# ## 2. Indexing in Pandas

# pandas has its own accessor operators
#  loc and iloc
# 
# *   iloc : Index based selection: selecting data based on its numerical position in the data
# *   loc : label based selection: selecting data based on index value rather than numerical position in the data
# 
# 
# 
# 
# 

# ### iloc or Index Based Selection

# In[15]:


Data1.iloc[0]


# In[ ]:


# In Both loc and iloc are row-first, column-second
# First index represents: row
# Second index represents: column


# In[16]:


# To get a column with iloc

Data1.iloc[:,0]


# In[17]:


# To select First name column from just the first, second, and third row, we will do:

Data1.iloc[:3, 0]


# In[18]:


# To select just the second and third entries, we will do:

Data1.iloc[1:3, 0]


# In[ ]:


# We can also pass a list as below:

Data1.iloc[[1,2,3], 0]


# In[19]:


# We can also use negative numbers for selection.
# This will start counting forwards from the end of the values.
# Last four elements of the dataset can be selected as below:

Data1.iloc[-4:]


# ### loc or Label Based Selection

# In[20]:


Data1.loc[1,'First_name'] 


# In[21]:


Data1.loc[:,['First_name','Last_name']]


# ### iloc vs loc

# iloc and loc methods use slightly different indexing schemes.
# 
# 
# *   iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9.
# 
# *   loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.
# 
# 
#  

# ### Index Manipulation

# The index we use is not mutable. If we want to manipulate the index we use set_index() method

# In[22]:


# set_index to First_name

Data1.set_index("First_name")


# In[23]:


Data1.head()


# ## Conditional Selection

# Condition Selection is used when we are asked questions based on conditions

# In[24]:


Data1.First_name=="Sarah"


# In[25]:


# Above result can then be used inside of loc to select the relevant data:

Data1.loc[Data1.First_name=="Sarah"]


# In[ ]:


# Conditional selection Example
# Data1.loc[(data1.First_name == 'Sarah') & (age >= 30)]


# Pandas comes with a few built-in conditional selectors, two of them are given below:
# 
# *   isin
# *   isnull or notnull
# 

# In[26]:


# isin

Data1.loc[Data1.First_name.isin(['Andrea','Sarah'])]


# In[27]:


# notnull

Data1.loc[Data1.First_name.notnull()]


# In[28]:


# isnull

Data1.loc[Data1.First_name.isnull()]


# ## Assigning Data

# In[29]:


Data1['First_name']='Deep'
Data1


# In[30]:


Data1


# ## 3. Summary Functions and Maps

# In[ ]:


Data1


# In[31]:


Data1.describe()


# In[32]:


# for specific field

Data1.First_name.describe()


# If we want to get some particular simple summary statistic about a column in a DataFrame, we can use inbuilt helpfull pandas function.
# 
# For example, to see the mean of the any numerical values containing column, we can use the mean() function:

# In[ ]:


# Data1.column_name.mean()


# In[33]:


#To see  unique values unique() function can be used as shown below:

Data1.First_name.unique()


# In[34]:


# To see unique values and how frequently they are occuring in the dataset, value_counts() method can be used as shown below:

Data1.First_name.value_counts()


# ## Maps

# In data science we often have a need for creating new representations from existing data, or for transforming data from the format it is in now to the format that we want it to be in later. Maps are what handle this work, making them extremely important for getting your work done!
# 
# There are two mapping methods that you will use often.
# *   map()
# *   apply()
# 
# 

# ## map
# 
# *   map term is borrowed from mathematics
# *   It takes one set of values and "maps" them to another set of values
# 
# 
# 
# 
# 

# In[35]:


Data1_age_mean = Data1.age.mean()
Data1.age.map(lambda p: p - Data1_age_mean)


# ## apply
# 
# *   apply() is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.
# 
# 
# 

# In[36]:


def remean_age(row):
    row.age = row.age - Data1_age_mean
    return row

Data1.apply(remean_age, axis='columns')


# ## 4. Grouping and Sorting

# Maps allow us to transform data in a DataFrame or Series one value at a time for an entire column. 
# 
# However, often we want to group our data, and then do something specific to the group the data is in.
# 
# We do this with the groupby() operation.

# In[37]:


Data1.groupby('age').age.count()


# The above function is equivalent to value_counts().

# In[38]:


# to get minimum age group by First_name

Data1.groupby('First_name').age.min()


# ## 4. Data Types and Missing Values

# Dtypes
# 
# 
# 
# *   The data type for a column in a DataFrame or a Series is known as the dtype.
# *   We can use the dtype property to grab the type of a specific column.
# *   For instance, we can get the dtype of the age column in the Data1 DataFrame:
# 
# 
#   

# In[39]:


Data1.age.dtype


# In[40]:


# Alternatively, the dtypes property returns the dtype of every column in the DataFrame:

Data1.dtypes


# 
# *   Data types tell us something about how pandas is storing the data internally.
# *   float64 means that it's using a 64-bit floating point number;
# *   int64 means a similarly sized integer instead, and so on.
# *   columns consisting entirely of strings do not get their own type; they are instead given the object type.
# *   It's possible to convert a column of one type into another wherever such a conversion makes sense by using the astype() function. 
# 
# 
#  

# In[ ]:


Data2=Data1


# In[ ]:


Data2.age=Data1.age.astype('float64')


# In[ ]:


Data2.age.dtype


# In[ ]:


# A DataFrame or Series index has its own dtype, too:

Data1.index.dtype


# ## Missing data

# *   Entries missing values are given the value NaN, short for "Not a Number".
# *   For technical reasons these NaN values are always of the float64 dtype.
# 
#  

# In[41]:


# Pandas provides some methods specific to missing data. 
# To select NaN entries you can use pd.isnull()

Data1[pd.isnull(Data1.First_name)]


# Replacing missing values is a common operation. Pandas provides a really handy method for this problem: fillna(). fillna() provides a few different strategies for mitigating such data. For example, we can simply replace each NaN with an "Unknown":

# In[42]:


Data1.age.fillna("Unknown")


# In[ ]:


Data1.Last_name.replace("Ross","Kamboj")


# ## 6. Renaming and Combining

# In[ ]:


Data1.rename(columns={'age': 'score'})


# In[ ]:


# Lets rename index or column values by specifying a index or column keyword parameter, respectively. 

Data1.rename(index={1: 'firstEntry', 2: 'secondEntry'})


# ## Combining

# *   When performing operations on a dataset, we will sometimes need to combine different DataFrames and/or Series in non-trivial ways.
# *   Pandas has three core methods for doing this. In order of increasing complexity, these are concat(), join(), and merge().
# *   Most of what merge() can do can also be done more simply with join()
# 
# 
# 

# In[ ]:


# Following Data Download Link: https://www.kaggle.com/datasets/datasnaek/youtube-new?resource=download

youtube_Data1 = pd.read_csv("/content/CAvideos.csv")


# In[ ]:


youtube_Data2 = pd.read_csv("/content/GBvideos.csv")


# In[ ]:


pd.concat([youtube_Data1, youtube_Data2])


# In[ ]:


# join() lets us combine different DataFrame objects which have an index in common.

# For example, to pull down videos that happened to be trending on the same day in both Canada and the UK, 
# we could do the following:

left = youtube_Data1.set_index(['title', 'trending_date'])
right = youtube_Data2.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')


# In[ ]:





# In[ ]:




