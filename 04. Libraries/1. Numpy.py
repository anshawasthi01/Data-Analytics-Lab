#!/usr/bin/env python
# coding: utf-8

# Content Credit: https://www.kaggle.com/code/orhansertkaya/numpy-tutorial-for-beginners

# 
# ###1. Introduction to NumPy

# 
# *   NumPy is a Python library used for working with arrays.
# *   It also has functions for working in domain of linear algebra, fourier transform, and matrices.
# *   NumPy stands for Numerical Python.
# 
# 

# In[2]:


import numpy as np


# In[3]:


np.array([3.2,4,6,5])


# In[4]:


np.array([1,4,2,5,3]) ## integer array:


# ### Understanding Data Types in Python

# In[5]:


np.array([1,2,3,4], dtype="str")


# In[6]:


np.array([3,6,2,3], dtype="float32")


# In[8]:


# nested lists result in multidimensional arrays

np.array([range(i,i+3) for i in [2,4,6]])


# ### Creating Arrays from Scratch

# In[9]:


# Create a length-10 integer array filled with zeros
# np.zeros() is a NumPy function that returns a new array of a specified shape and data type, filled with zeros. 

np.zeros(10, dtype="int") 


# In[10]:


np.zeros((5,6), dtype="float")


# In[11]:


# Create a 3x5 floating-point array filled with 1s

# np.ones() is a NumPy function that returns a new array of a specified shape and data type, filled with ones.

np.ones((3,5), dtype="float")


# In[12]:


# Create a 3x5 array filled with 3.14
# np.full() is a NumPy function that returns a new array of a specified shape and data type, filled with a specified fill value.

np.full((3,5), 3.14)


# In[13]:


# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
# np.arange() is a NumPy function that returns an array of evenly spaced values within a specified interval.

np.arange(0,20,2)


# In[14]:


# Create an array of five values evenly spaced between 0 and 1
# np.linspace() is a NumPy function that returns an array of evenly spaced numbers over a specified interval.

np.linspace(0,1,5)


# In[15]:


# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
# np.random.random() is a NumPy function that returns an array of random numbers that are uniformly distributed between 0 and 1. 

np.random.random((3,3))


# In[16]:


# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
# np.random.normal() is a NumPy function that generates an array of random numbers that are normally distributed, with a specified mean and standard deviation. 

np.random.normal(0,1,(3,3))


# In[17]:


# Create a 3x3 array of random integers in the interval [0, 10)

# np.random.randint() is a NumPy function that generates an array of random integers from a low (inclusive) to a high (exclusive) range.

np.random.randint(0,10,(3,3))


# In[18]:


# Create a 3x3 identity matrix
# np.eye() is a NumPy function that returns a 2-D array with ones on the diagonal and zeros elsewhere.

np.eye(3)


# ### NumPy Standard Data Types

# In[19]:


# Return a new array of given shape and type, with random values

# np.empty() is a NumPy function that creates an uninitialized array of a specified shape and data type.

np.empty((3,3),dtype="int")


# In[20]:


#or using the associated NumPy object:

np.zeros(10,dtype=np.int16)


# ### NumPy Array Attributes

# In[22]:


#NumPy Array Attributes
#We’ll use NumPy’s random number generator, which we will seed with a set value in order to ensure that the same random arrays are generated each time this code is run:

# np.random.seed(0) # seed for reproducibility
x1 = np.random.randint(10, size=6) # One-dimensional array


# In[23]:


# Each array has attributes ndim (the number of dimensions), 
# shape (the size of each dimension), 
# and size (the total size of the array)

x1 = np.random.randint(10, size=6) #it's same ((np.random.randint((0,10), size=6))) # One-dimensional array
x2 = np.random.randint(10, size=(3,4)) # Two-dimensional array
x3 = np.random.randint(10, size=(3,4,5)) # Three-dimensional array


# In[24]:


x1


# In[25]:


x2


# In[26]:


x3


# In[27]:


print("x1 ndim: ",x1.ndim)
print("x1 shape: ",x1.shape)
print("x1 size: ",x1.size) #totaly,6 elements


# In[28]:


print("x2 ndim: ",x2.ndim)
print("x2 shape: ",x2.shape)
print("x2 size: ",x2.size) #totaly,12 elements


# In[29]:


print("x3 ndim: ",x3.ndim)
print("x3 shape: ",x3.shape)
print("x3 size: ",x3.size)#totaly,60 elements


# In[30]:


print("dtype: ",x1.dtype) #the data type of the array


# In[31]:



print("itemsize:",x1.itemsize,"bytes") #lists the size (in bytes) of each array element
print("nbytes:",x1.nbytes,"bytes") #lists the total size (in bytes) of the array


# In[32]:


print("dtype: ",x2.dtype)


# In[ ]:



print("itemsize:",x2.itemsize,"bytes")
print("nbytes:",x2.nbytes,"bytes")


# In[ ]:


# print("dtype: ",x3.dtype) #the data type of the array
# print("itemsize:",x3.itemsize,"bytes")
# print("nbytes:",x3.nbytes,"bytes") 


# ### Array Indexing: Accessing Single Elements

# *   If you are familiar with Python’s standard list indexing, indexing in NumPy will feel quite familiar. 
# *   In a one-dimensional array, you can access the ith value (counting from zero) by specifying the desired index in square brackets, just as with Python lists:
# 
# 

# In[33]:


x1


# In[34]:


x1[0]


# In[35]:


x1[3]


# In[36]:


#To index from the end of the array, you can use negative indices:

x1[-1]


# In[37]:


x1[-2]


# In[38]:


x2


# In[39]:


# In a multidimensional array, you access items using a comma-separated tuple of indices:

x2[2,1]


# In[40]:


x2[2,0]


# In[41]:


x2[2,-4]


# In[42]:


x2[-2,-3]


# In[43]:


#We can also modify values using any of the above index notation:

x2[0,0]=12
x2


# In[44]:


x1


# In[45]:


x1[0] = 4.14159 # this will be truncated!
x1


# ### Array Slicing: Accessing Subarrays

# *   Just as we can use square brackets to access individual array elements, we can also use them to access subarrays with the slice notation, marked by the colon (:) character. 
# 
# *   The NumPy slicing syntax follows that of the standard Python list; to access a slice of an array x, use this:
# x[start:stop:step]
# 
# *   If any of these are unspecified, they default to the values start=0, stop=size of dimension, step=1.
# 
# 
# 
# 
# 

# ##### One-dimensional subarrays

# In[46]:


x = np.arange(10)
x


# In[47]:


x[:5] # first five elements


# In[48]:


x[5:] # elements after index 5


# In[49]:


x[4:7]# middle subarray


# In[50]:


x[::2] # every other element


# In[51]:


x[1::2] 


# In[ ]:


x[-7:-2:2]


# In[ ]:


# x[-4:-2:1]


# In[ ]:


# A potentially confusing case is when the step value is negative. In this case, the
# defaults for start and stop are swapped. This becomes a convenient way to reverse
# an array:

# x[::-1] # all elements, reversed


# In[ ]:


# x[5::-2]# reversed every other from index 5


# In[ ]:


# x[5:1:-2]


# In[ ]:


# x[5:-8:-1]


# In[ ]:


# x[7:-6:-1]


# In[ ]:


# x[-7:-8:-1]


# ### Multidimensional subarrays

# In[52]:


# Multidimensional slices work in the same way, with multiple slices separated by commas.
# For example:

x2


# In[53]:


# two rows, three columns

x2[:2, :3]


# In[54]:




x2[:3,::2]


# In[ ]:




x2[::-1,::-1]


# ### Accessing array rows and columns

# In[ ]:


# One commonly needed routine is accessing single
# Rows or columns of an array. You can do this by combining indexing and slicing,
# Using an empty slice marked by a single colon (:)

# print(x2[:, 0]) # first column of x2


# In[ ]:


# print(x2[0,:]) # first row of x2


# In[ ]:


#In the case of row access, the empty slice can be omitted for a more compact syntax:

# print(x2[0]) # equivalent to x2[0, :]


# ### Subarrays as no-copy views

# *   One important—and extremely useful—thing to know about array slices is that they return views rather than copies of the array data.
# 
# *   This is one area in which NumPy array slicing differs from Python list slicing: in lists, slices will be copies. 
# 
# 

# In[ ]:


# print(x2)


# In[ ]:


#Let’s extract a 2×2 subarray from this:

# x2_sub = x2[:2,:2]
# print(x2_sub)


# In[ ]:


#Now if we modify this subarray, we’ll see that the original array is changed! Observe:

# x2_sub[0,0] = 99
# print(x2_sub)


# In[ ]:


# print(x2)


# This default behavior is actually quite useful: it means that when we work with large datasets, we can access and process pieces of these datasets without the need to copy the underlying data buffer.

# ####Creating copies of arrays

# Despite the nice features of array views, it is sometimes useful to instead explicitly copy the data within an array or a subarray. This can be most easily done with the copy() method:

# In[ ]:


# x2_sub_copy = x2[:2,:2].copy()
# print(x2_sub_copy)


# In[ ]:


#If we now modify this subarray, the original array is not touched:

# x2_sub_copy[0,0] = 42
# print(x2_sub_copy)


# In[ ]:


# print(x2)


# ###Reshaping of Arrays

# In[ ]:


# Another useful type of operation is reshaping of arrays. The most flexible way of
# doing this is with the reshape() method. For example, if you want to put the numbers
# 1 through 9 in a 3×3 grid, you can do the following:

# grid = np.arange(1,10,1).reshape(3,3)
# print(grid)


# Another common reshaping pattern is the conversion of a one-dimensional array into a two-dimensional row or column matrix. You can do this with the reshape method, or more easily by making use of the newaxis keyword within a slice operation:

# In[ ]:


# x = np.array([1, 2, 3])
# x.shape # x is a vector (3,)


# In[ ]:


# x


# In[ ]:


# x.reshape(1,3).shape


# In[ ]:


# x


# In[ ]:


# x.reshape(1,-1).shape


# In[ ]:


# row vector via newaxis

# The expression x[np.newaxis, :] is equivalent to calling x.reshape(1, -1), 
# which also adds an extra dimension to the array x at position 0 with a size of 1.

# x[np.newaxis, :].shape
# x


# In[ ]:


# x.reshape(1,-1).shape


# In[ ]:


# column vector via reshape

# x.reshape((3, 1))


# In[ ]:


# column vector via newaxis

# x[:, np.newaxis]


# In[ ]:


# x.reshape(-1,1).shape


# ### Array Concatenation and Splitting

# ####Concatenation of arrays

# In[ ]:


# x = np.array([1,2,3])
# y = np.array([3,2,1])
# np.concatenate((x, y))


# In[ ]:


# You can also concatenate more than two arrays at once:

# z = np.array([99,99,99]) #z =[99,99,99]

# print(np.concatenate((x,y,z)))


# In[ ]:


# np.concatenate can also be used for two-dimensional arrays:

# grid = np.array([[1,2,3],
               # [4,5,6]])


# In[ ]:


# concatenate along the first axis

# np.concatenate((grid,grid))


# In[ ]:


# concatenate along the second axis (zero-indexed)

# np.concatenate((grid, grid), axis=1)


# *   For working with arrays of mixed dimensions, it can be clearer to use the 
# *   np.vstack(vertical stack) and np.hstack (horizontal stack) functions:
# *   Similarly, np.dstack will stack arrays along the third axis.
# 
# 
# 
# 
# 
# 

# In[ ]:


# x = np.array([1,2,3])
# grid = np.array([[9,8,7],
                # [6,5,4]])
# vertically stack the arrays
# np.vstack([x,grid])


# In[ ]:


# y = np.array([[99],
           # [99]])
# np.hstack([grid,y])


# ### Splitting of arrays

# The opposite of concatenation is splitting, which is implemented by the functions np.split, np.hsplit, and np.vsplit. For each of these, we can pass a list of indices giving the split points:

# In[ ]:


# x = np.array([1,2,3,99,99,3,2,1])
# x1, x2, x3, x4 = np.split(x, [3,4,5])
# print(x1, x2, x3,x4)


# Notice that N split points lead to N + 1 subarrays. The related functions np.hsplit and np.vsplit are similar

# In[ ]:


# grid = np.arange(36,dtype=np.float).reshape((6,6))
# grid


# In[ ]:


# upper, lower = np.vsplit(grid, [3])
# print(upper)
# print(lower)


# In[ ]:


# upper,middle, lower = np.vsplit(grid, [2,3])
# print("upper: ",upper)
# print("middle: ",middle)
# print("lower: ",lower)


# In[ ]:


# left, right = np.hsplit(grid, [2])
# print(left)
# print(right)


# In[ ]:


# left, right = np.hsplit(grid, 2)
# print(left)
# print(right)


# ## Computation on NumPy Arrays: Universal Functions

# Exploring NumPy’s UFuncs
# 
# *   Ufuncs exist in two flavors: unary ufuncs, which operate on a single input, and binary ufuncs, which operate on two inputs. We’ll see examples of both these types of functions here.
# 
# 

# ### Array arithmetic

# In[ ]:


x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) # floor division


# In[ ]:


#There is also a unary ufunc for negation, a ** operator for exponentiation, and a %
#operator for modulus:

print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)


# In[ ]:


# In addition, these can be strung together however you wish, and the standard order
# of operations is respected:

-(0.5*x+1) ** 2


# In[ ]:


# All of these arithmetic operations are simply convenient wrappers around specific
# functions built into NumPy; for example, the + operator is a wrapper for the add
# function:

print(np.add(3,2))


# In[ ]:


print(np.add(x,2)) #Addition +


# In[ ]:


print(np.subtract(x,5)) #Subtraction -


# In[ ]:


print(np.negative(x)) #Unary negation -


# In[ ]:


print(np.multiply(x,3)) #Multiplication *


# In[ ]:


print(np.divide(x,2)) #Division /


# In[ ]:


print(np.floor_divide(x,2)) #Floor division //


# In[ ]:


print(np.power(x,2)) #Exponentiation **


# In[ ]:


print(np.mod(x,2)) #Modulus/remainder **


# In[ ]:


print(np.multiply(x, x))


# In[ ]:


x = np.array([-2,-1,0,1,2])
abs(x)


# In[ ]:


# The corresponding NumPy ufunc is np.absolute, which is also available under the
# alias np.abs:

print(np.absolute(x))
print(np.abs(x))


# ### Trigonometric functions

# In[ ]:


# NumPy provides a large number of useful ufuncs, and some of the most useful for the
# data scientist are the trigonometric functions. We’ll start by defining an array of angles

theta = np.linspace(0,np.pi,3)


# In[ ]:


# Now we can compute some trigonometric fuctions on these values:

print("theta      =",theta)


# In[ ]:


print("sin(theta) =",np.sin(theta))


# In[ ]:


print("cos(theta) =",np.cos(theta))


# In[ ]:


print("tan(theta) =",np.tan(theta))


# In[ ]:


x = [-1, 0, 1]


# In[ ]:


print("x = ", x)


# In[ ]:


print("arcsin(x) = ", np.arcsin(x))


# In[ ]:


print("arccos(x) = ", np.arccos(x))


# In[ ]:


print("arctan(x) = ", np.arctan(x))


# #### Exponents and logarithms
# 
# Another common type of operation available in a NumPy ufunc are the exponentials:

# In[ ]:


x = [1,2,3]


# In[ ]:


print("x      =",x)


# In[ ]:


print("e^x    =",np.exp(x))


# In[ ]:


print("2^x    =",np.exp2(x))


# In[ ]:


print("3^x    =",np.power(3,x))


# The inverse of the exponentials, the logarithms, are also available. The basic np.log gives the natural logarithm; if you prefer to compute the base-2 logarithm or the base-10 logarithm, these are available as well:
# 

# In[ ]:


x = [1, 2, 4, 10]


# In[ ]:


print("x        =", x)


# In[ ]:


print("ln(x)    =", np.log(x))


# In[ ]:


print("log2(x)  =", np.log2(x))


# In[ ]:


print("log10(x) =", np.log10(x))


# ### Advanced Ufunc Features

# ###Specifying output
# 
# For large calculations, it is sometimes useful to be able to specify the array where the result of the calculation will be stored. Rather than creating a temporary array, you can use this to write computation results directly to the memory location where you’d like them to be. For all ufuncs, you can do this using the out argument of the function:

# In[ ]:


x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)


# In[ ]:


#This can even be used with array views. For example, we can write the results of a
#computation to every other element of a specified array:

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

# If we had instead written y[::2] = 2 ** x, this would have resulted in the creation
# of a temporary array to hold the results of 2 ** x, followed by a second operation
# copying those values into the y array. This doesn’t make much of a difference for such
# a small computation, but for very large arrays the memory savings from careful use of
# the out argument can be significant.


# In[ ]:


x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)


# ### Aggregates
# 
# For binary ufuncs, there are some interesting aggregates that can be computed directly from the object. For example, if we’d like to reduce an array with a particular operation, we can use the reduce method of any ufunc. A reduce repeatedly applies a given operation to the elements of an array until only a single result remains. For example, calling reduce on the add ufunc returns the sum of all elements in the array:

# In[ ]:


x = np.arange(1,6)
print(np.add.reduce(x))
print(np.subtract.reduce(x))
print(np.multiply.reduce(x))


# In[ ]:


#If we’d like to store all the intermediate results of the computation, we can instead use accumulate:

print(np.add.accumulate(x))
print(np.subtract.accumulate(x))
print(np.multiply.accumulate(x))
print(np.divide.accumulate(x))
print(np.floor_divide.accumulate(x))


# In[ ]:


x = np.arange(1,6)
np.multiply.outer(x, x)


# ### Aggregations: Min, Max, and Everything in Between
# 
# ####Summing the Values in an Array

# In[ ]:


# As a quick example, consider computing the sum of all values in an array. Python
# itself can do this using the built-in sum function:

L = np.random.random(100)


# In[ ]:


L


# In[ ]:


sum(L)


# #### Minimum and Maximum

# In[ ]:


min(L)


# In[ ]:


max(L)


# #### Multidimensional aggregates
# 

# In[ ]:


# One common type of aggregation operation is an aggregate along a row or column.
# Say you have some data stored in a two-dimensional array:

M = np.random.random((3,4))
print(M)

M.sum()


# ### Computation on Arrays: Broadcasting
# 
# Broadcasting is simply a set of rules for applying binary ufuncs (addition, subtraction, multiplication, etc.) on arrays of different sizes.

# In[ ]:


import numpy as np

a = np.array([0,1,2])
b = np.array([5,5,5])
a+b


# In[ ]:


a+5


# In[ ]:


M = np.ones((3,3))
M


# In[ ]:


M+a


# In[ ]:


# here we’ve stretched both a and b to match a common shape, and the result is a two-
# dimensional array!

a = np.arange(3) #(3,) 1 dimensional
b = np.arange(3)[:,np.newaxis] #(3,1) 2 dimensional
print(a)
print(b)


# In[ ]:


a+b


# Broadcasting in NumPy follows a strict set of rules to determine the interaction between the two arrays:
# 
# Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.
# 
# Rule 2: If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
# 
# Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

# ### Comparisons, Masks, and Boolean Logic

# In[ ]:


x = np.array([1,2,3,4,5])

print(x<3)  # less than
print(x>3)  # greater than
print(x<=3) #less than or equal
print(x>=3) #greater than or equal
print(x!=3) #not equal
print(x==3) #equal


# In[ ]:


# It is also possible to do an element-by-element comparison of two arrays, and to
# include compound expressions:

(2*x) == (2**x)


# In[ ]:


rng = np.random.RandomState(seed=0)
x = rng.randint(10, size=(3,4))
print(x)

x<6


# In[ ]:


# In Python, all nonzero integers will evaluate as True .
bool(42), bool(0), bool(-1)


# In[ ]:


bool(42 and 0)


# In[ ]:


bool(42 or 0)


# In[ ]:


# When you have an array of Boolean values in NumPy, this can be thought of as a
# string of bits where 1 = True and 0 = False , and the result of & and | operates in a
# similar manner as before:

A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B


# In[ ]:


x = np.arange(10)
(x > 4) & (x < 8)


# ### Fancy Indexing

# In[ ]:


import numpy as np

rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)


# In[ ]:


[x[3],x[7],x[2]]


# In[ ]:


ind = [3,7,4]
x[ind]


# In[ ]:


ind = np.array([[3, 7],
                [4, 5]])
x[ind]


# In[ ]:


X = np.arange(12).reshape((3,4))
X


# In[ ]:


row = np.array([0,1,2])
col = np.array([2,1,3])
X[row,col]


# #### Combined Indexing

# In[ ]:


print(X)


# In[ ]:


X[2,[2,0,1]]


# In[ ]:


X[1:, [2, 0, 1]]


# In[ ]:


# Modifying Values with Fancy Indexing

x = np.arange(10)
i = np.array([2,1,8,4])
x[i] = 99
print(x)


# In[ ]:


x[i] -= 10
print(x)


# In[ ]:


x = np.zeros(10)
x[[0, 2]] = [4, 6]
print(x)


# In[ ]:


i = [2, 3, 3, 4, 4, 4]
x[i] += 1
x


# In[ ]:


x = np.zeros(10)
np.add.at(x, i, 1)
print(x)


# ### Sorting Arrays

# In[ ]:


# Fast Sorting in NumPy: np.sort and np.argsort

x = np.array([2,1,4,3,5])
np.sort(x)


# In[ ]:


x.sort()
print(x)


# In[ ]:


#return indices
x = np.array([2,1,4,3,5])
i = np.argsort(x)     # Returns the indices that would sort an array
print(i)

x[i]


# ### Sorting along rows or columns

# In[ ]:


# A useful feature of NumPy’s sorting algorithms is the ability to sort along specific
# rows or columns of a multidimensional array using the axis argument. For example:

rand = np.random.RandomState(42)
X = rand.randint(0,10,(4,6))
print(X)


# In[ ]:


# sort each column of X

np.sort(X, axis=0)


# In[ ]:


# sort each row of X

np.sort(X, axis=1)


# ### Partial Sorts: Partitioning

# In[ ]:


# Note that the first three values in the resulting array are the three smallest in the
# array, and the remaining array positions contain the remaining values. Within the
# two partitions, the elements have arbitrary order.

x = np.array([7, 2, 1, 3, 6, 5, 4])
np.partition(x, 3)


# In[ ]:


# The result is an array where the first two slots in each row contain the smallest values
# from that row, with the remaining values filling the remaining slots.

np.partition(X, 2, axis=1)


# In[ ]:


np.partition(X, 2, axis=0)


# In[ ]:




