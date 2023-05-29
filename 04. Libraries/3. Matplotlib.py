#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Content Credit: https://www.kaggle.com/code/berkayalan/matplotlib-a-complete-data-visualization-guide


# In[1]:


from matplotlib import pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


from matplotlib import font_manager as fm


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[ ]:


from datetime import datetime, timedelta #It's for time series


# ### Pyplot
# 
# Pyplot is a collection of functions that make matplotlib work like MATLAB. Each pyplot function makes some change to a figure

# In[4]:


x = [0,2,4,5,6,7,8,9,10]
y = [60,13,45,29,48,77,102,95,58]


# In[5]:


plt.plot(x, y)
plt.show()


# ## 1. Line Plot
# 
# A Line plot can be defined as a graph that displays data as points or check marks above a number line, showing the frequency of each value.

# In[6]:


experience = [1,3,4,5,7,8,10,12]

salary = [6500, 9280, 12050, 13200, 16672, 21000, 23965, 29793]


# In[7]:


plt.plot(experience,salary)
plt.show()


# In[8]:


# Adding a Title

plt.plot(experience,salary)
plt.title("Salary of Data Scientists by their experiences")
plt.show()


# In[9]:


# Adding Labels to x and y

plt.plot(experience,salary)
plt.title("Salary of Data Scientists by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


# In[11]:


# Plotting Multiple Graphs in One Graph

experience = [1,3,4,5,7,8,10,12]

data_scientists_salary = [6500, 9280, 12050, 13200, 16672, 21000, 23965, 29793]

software_engineers_salary = [9020, 12873, 15725, 18000, 19790, 20196, 25769,32000 ]


# In[12]:


plt.plot(experience,data_scientists_salary)
plt.plot(experience,software_engineers_salary)

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


# In[13]:


# We can't understand which line represents what, we need to add legend. 
# We can add them as a list or we can add them in the beginning.

plt.plot(experience,data_scientists_salary)
plt.plot(experience,software_engineers_salary)

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend(["Data Scientists","Software Engineers"])

plt.show()


# In[14]:


plt.plot(experience,data_scientists_salary, label= "Data Scientists")
plt.plot(experience,software_engineers_salary, label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()

plt.show()


# In[15]:


# We can also change the location of legend with loc argument

plt.plot(experience,data_scientists_salary, label= "Data Scientists")
plt.plot(experience,software_engineers_salary, label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend(loc="lower right")

plt.show()


# In[16]:


# A format string consists of a part for color, marker and line
# Each of them is optional. 
# If not provided, the value from the style cycle is used.

plt.plot(experience,data_scientists_salary,color="r", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g", label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()

plt.show()


# In[17]:


plt.plot(experience,data_scientists_salary,color="r", linestyle="--", label= "Data Scientists") #We can also make lines different
plt.plot(experience,software_engineers_salary, color="g",linestyle=':', label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()

plt.show()


# In[18]:


#We can also add markers

plt.plot(experience,data_scientists_salary,color="r", linestyle="--",marker="o", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g",linestyle=':',marker=".", label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()

plt.show()


# In[19]:


#We can also adjust line width by using linewidth argument.

plt.plot(experience,data_scientists_salary,color="r", linestyle="--",linewidth=6,marker="o", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g",linestyle=':',marker=".",linewidth=6, label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()

plt.show()


# tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area. This is an experimental feature and may not work for some cases. 

# In[20]:




plt.plot(experience,data_scientists_salary,color="r", linestyle="--",linewidth=6,marker="o", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g",linestyle=':',marker=".",linewidth=6, label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.tight_layout()
plt.show()


# In[21]:


#We can also add grids by using grids argument

plt.plot(experience,data_scientists_salary,color="r", linestyle="--",linewidth=6,marker="o", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g",linestyle=':',marker=".",linewidth=6, label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


# In[22]:


# We can fill below of the lines by using stackplot.

plt.stackplot(experience,data_scientists_salary, colors="g")
plt.title("Salary of Data Scientists by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


# In[23]:


# We can change the style of the plots. In order to see all available styles:

plt.style.available


# In[24]:


plt.style.use('dark_background')

plt.plot(experience,data_scientists_salary,color="r", linestyle="--",linewidth=6,marker="o", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g",linestyle=':',marker=".",linewidth=6, label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


# In[ ]:


# We can save figures by using savefig argument.


# In[ ]:


plt.style.use('seaborn-dark')

plt.plot(experience,data_scientists_salary,color="r", linestyle="--",linewidth=6,marker="o", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g",linestyle=':',marker=".",linewidth=6, label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.tight_layout()
plt.grid(True)

plt.savefig("plot1.png")

plt.show()


# ## 2. Bar Plot
# 
# A barplot (or barchart) is one of the most common types of graphic. It shows the relationship between a numeric and a categoric variable. Each entity of the categoric variable is represented as a bar. The size of the bar represents its numeric value.

# In[25]:


x = ["A", "B", "C", "D"]
y = [3, 8, 1, 10]


# In[26]:


plt.bar(x,y)
plt.show()


# In[27]:


experience = [1,2,3,4,5,6,7,8]

data_scientists_salary = [6500, 9280, 12050, 13200, 16672, 21000, 23965, 29793]


# In[28]:


plt.style.use('seaborn-paper')

plt.bar(experience,data_scientists_salary,color="b")

plt.title("Salary of Data Scientists")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.tight_layout()
plt.grid(False)

plt.show()


# We can combine bar and line plot.

# In[29]:


experience = [1,2,3,4,5,6,7,8]

data_scientists_salary = [6500, 9280, 12050, 13200, 16672, 21000, 23965, 29793]

software_engineers_salary = [9020, 12873, 15725, 18000, 19790, 20196, 25769,32000 ]


# In[30]:


plt.style.use('tableau-colorblind10')

plt.bar(experience,data_scientists_salary,color="r", label= "Data Scientists")
plt.plot(experience,software_engineers_salary, color="g",label= "Software Engineers" )

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.grid(False)

plt.show()


# We can specify the width with width argument.

# In[31]:


width = 0.2

plt.style.use('tableau-colorblind10')

plt.bar(experience,data_scientists_salary,color="m",width=width, label= "Data Scientists")

plt.title("Salary of Data Scientists by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.grid(False)

plt.show()


# We can also plot multiple bar plots.

# In[32]:


plt.style.use("fivethirtyeight")

plt.bar(experience,software_engineers_salary, color="g",linewidth=3,label= "Software Engineers" )
plt.bar(experience,data_scientists_salary,color="r",linewidth=3, label= "Data Scientists")

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.grid(False)

plt.show()


# In[33]:


experience_indexes = np.arange(len(experience))


# In[34]:


experience_indexes


# In[35]:


plt.style.use("fivethirtyeight")

width = 0.4

plt.bar(experience_indexes - width,software_engineers_salary, color="g",width=width,linewidth=3,label= "Software Engineers" )
plt.bar(experience_indexes+width,data_scientists_salary,color="r",linewidth=3,width=width, label= "Data Scientists")

plt.title("Salary of Data Scientists and Software Engineers by their experiences")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.legend()
plt.grid(False)

plt.show()


# ## 3. Pie Chart

# A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion. In a pie chart, the arc length of each slice (and consequently its central angle and area), is proportional to the quantity it represents.

# In[36]:


experience = [1,2,3,4,5,6,7,8]

data_scientists_salary = [6500, 9280, 12050, 13200, 16672, 21000, 23965, 29793]

software_engineers_salary = [9020, 12873, 15725, 18000, 19790, 20196, 25769,32000 ]


# In[37]:


plt.title("Pie Chart Example")

slices = [60,40]

plt.pie(slices)

plt.tight_layout()

plt.show()


# In[38]:


list_1 = [40,56,72,38,4]

plt.pie(list_1)

plt.show()


# In[39]:


incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

plt.pie(incomes,labels=persons)

plt.show()


# By default the plotting of the first wedge starts from the x-axis and move counterclockwise.
# 
# But you can change the start angle by specifying a startangle parameter. The startangle parameter is defined with an angle in degrees, default angle is 0.

# In[40]:


incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

plt.pie(incomes,labels=persons,startangle=180)

plt.show()


# If we want pull out one or more slices

# In[41]:


incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

myexplode = [0,0.2, 0, 0, 0]

plt.pie(incomes,labels=persons,startangle=180,explode = myexplode)

plt.show()


# In[42]:


# We can also change size of the chart with figsize argument

plt.figure(figsize=(10,10))

plt.rcParams['font.size'] = 20

incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

myexplode = [0,0.2, 0, 0, 0]

plt.pie(incomes,labels=persons,startangle=180,explode = myexplode)

plt.show()


# In[43]:


# We can also add shadows by making shadow argument True.

plt.figure(figsize=(10,10))

plt.rcParams['font.size'] = 20

incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

myexplode = [0,0.2, 0, 0, 0]

plt.pie(incomes,labels=persons,startangle=180,explode = myexplode,shadow=True)

plt.show()


# We can also set color of each wedge with colors parameter.
# 
# Some of possible color options are here:
# 
# Shortage	Colour
# "r"	Red
# "g"	Green
# "b"	Blue
# "c"	Cyan
# "m"	Magenta
# "y"	Yellow
# "k"	Black
# "w"	White
# 

# In[44]:


plt.figure(figsize=(10,10))

plt.rcParams['font.size'] = 20

incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

myexplode = [0,0.2, 0, 0, 0]

colors = ["black","g","y","hotpink","#4CAF70"]

plt.pie(incomes,labels=persons,startangle=180,explode = myexplode,shadow=True,colors=colors)

plt.show()


# In[45]:


# In order to add a list of explanation for each wedge, we can use the legend() function.

plt.figure(figsize=(7,7))

incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

colors = ["black","g","y","hotpink","#4CAF70"]

plt.pie(incomes,labels=persons,colors=colors)
plt.legend()
plt.show()


# In[46]:


# We can also add a title to legends by using title parameter.

plt.style.use("fivethirtyeight")

plt.figure(figsize=(7,7))

incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

colors = ["black","g","y","hotpink","#4CAF70"]

plt.pie(incomes,labels=persons,colors=colors)
plt.legend(title="Persons")
plt.show()


# In[47]:


# We can add percentages of slices by using autopct argument.

plt.style.use("fivethirtyeight")

plt.figure(figsize=(7,7))

incomes = [40,56,72,38,4]

persons = ["Josh","Berkay","Maria","Michael","Anastacia"]

colors = ["black","g","y","hotpink","#4CAF70"]

plt.pie(incomes,labels=persons,colors=colors, autopct="%1.1f%%")
plt.legend(title="Persons")
plt.show()


# ## 4. Histograms
# 
# A histogram is a graph showing frequency distributions.
# 
# It is a graph showing the number of observations within each given interval.
# 
# We use hist() function in order to create histograms.

# In[48]:


notes = [30,74,94,14,55,47,63,28,88,44,53,18,66,74,81]


# In[49]:


plt.style.use("fivethirtyeight")

plt.hist(notes)

plt.show()


# In[ ]:


plt.style.use("fivethirtyeight")

plt.hist(notes,color="r")

plt.title("Notes")
plt.xlabel("Notes")
plt.ylabel("Person")

plt.tight_layout()
plt.grid(False)
plt.show()


# In[ ]:


# We can add edge colors in order to interpret the table better.

plt.style.use("fivethirtyeight")

plt.hist(notes,color="r",edgecolor="black")

plt.title("Notes")
plt.xlabel("Notes")
plt.ylabel("Person")

plt.tight_layout()
plt.grid(False)
plt.show()


# In[ ]:


# We can specify the size of bins

plt.style.use("fivethirtyeight")

plt.hist(notes,bins=5,color="g",edgecolor="black")

plt.title("Notes")
plt.xlabel("Notes")
plt.ylabel("Person")

plt.tight_layout()
plt.grid(False)
plt.show()


# In[ ]:


# We can give bin values spesifically

plt.style.use("fivethirtyeight")

bins = [10,45,65,80,100]

plt.hist(notes,bins=bins,color="g",edgecolor="black")

plt.title("Notes")
plt.xlabel("Notes")
plt.ylabel("Person")

plt.tight_layout()
plt.grid(False)
plt.show()


# In[ ]:


# Let's plot a normal distribution(bell shape)

x = np.random.normal(170, 10, 250)

plt.hist(x,color="gray",edgecolor="black")

plt.title("Normal Distribution")
plt.xlabel("Numbers")
plt.ylabel("Count")

plt.tight_layout()
plt.grid(False)
plt.show()


# ## 5. Scatter Plots

# Scatter plots are used to plot data points on horizontal and vertical axis in the attempt to show how much one variable is affected by another.

# In[50]:


first_exam_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
second_exam_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]


# In[51]:


plt.title("Exam Grades Scatter plot")

plt.scatter(first_exam_grades,second_exam_grades)

plt.tight_layout()
plt.xlabel("First Exam Grades")
plt.ylabel("Second Exam Grades")
plt.grid(True)
plt.show()


# We can change the dot size and color.

# In[ ]:


plt.title("Exam Grades Scatter plot")

plt.scatter(first_exam_grades,second_exam_grades,s=100,color="r")

plt.tight_layout()
plt.xlabel("First Exam Grades")
plt.ylabel("Second Exam Grades")
plt.grid(True)
plt.show()


# You can change dot size by values.

# In[ ]:


plt.title("Exam Grades Scatter plot")

sizes = np.array([20,50,100,200,500,1000,60,90,10,300])

plt.scatter(first_exam_grades,second_exam_grades,s=sizes,color="r")

plt.tight_layout()
plt.xlabel("First Exam Grades")
plt.ylabel("Second Exam Grades")
plt.grid(True)
plt.show()


# We can also change the marker.

# In[ ]:


plt.title("Exam Grades Scatter plot")

plt.scatter(first_exam_grades,second_exam_grades,s=100,color="green",marker="x")

plt.tight_layout()
plt.xlabel("First Exam Grades")
plt.ylabel("Second Exam Grades")
plt.grid(True)
plt.show()


# We can also plot two different plots.

# In[ ]:


first_exam_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
first_study_hours = [6,8,3,9,9,1,4,2,2,5]


# In[ ]:


second_exam_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
second_study_hours = [2,7,1,5,3,3,2,6,3,2]


# In[ ]:


plt.title("Exam Grades Scatter plot")

plt.scatter(first_exam_grades,first_study_hours,s=100,color="green",marker="x")
plt.scatter(second_exam_grades,second_study_hours,s=100,color="red")

plt.tight_layout()
plt.xlabel("Exam Grades")
plt.ylabel("Study Hours")

plt.grid(True)
plt.show()


# You can also add features and colormaps.

# In[ ]:


first_exam_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
second_exam_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
colors = [7, 5, 9, 7, 5, 7, 2, 5, 3, 7]
sizes = [209, 486, 381, 255, 191, 315, 185, 228, 174,538]


# In[ ]:


plt.title("Exam Grades Scatter plot")

plt.scatter(first_exam_grades,second_exam_grades,s=sizes,c=colors,cmap="Blues",edgecolor="black")

cbar = plt.colorbar()
cbar.set_label("Exam Grades")
plt.tight_layout()
plt.xlabel("First Exam Grades")
plt.ylabel("Second Exam Grades")

plt.grid(True)
plt.show()


# ## 6.Contour(Level) Plots
# 
# Contour plots (sometimes called Level Plots) are a way to show a three-dimensional surface on a two-dimensional plane. It graphs two predictor variables X Y on the y-axis and a response variable Z as contours. These contours are sometimes called the z-slices or the iso-response values.

# In[ ]:


x = [0,3,6,9,13,15,19,23,26,29,33,35,39,41,47,56] 
y = [5,8,13,16,17,20,25,26,30,33,37,39,41,44,48,59]


# In[ ]:


# Creating 2-D grid of features 
[X, Y] = np.meshgrid(x, y) 
  
fig, ax = plt.subplots(1, 1) 
  
Z = np.sqrt(X**2+Y**2)
  
# plots contour lines 
ax.contour(X, Y, Z) 
  
ax.set_title('Contour Plot') 
ax.set_xlabel('X values') 
ax.set_ylabel('Y values') 

plt.show()


# We can also fill inside of plot by using contourf() function.

# In[ ]:


# Creating 2-D grid of features 
[X, Y] = np.meshgrid(x, y) 
 
fig, ax = plt.subplots(1, 1) 
  
Z = np.sqrt(X**2+Y**2)
  
# plots contour lines 
ax.contourf(X, Y, Z) 
  
ax.set_title('Contour Plot') 
ax.set_xlabel('X values') 
ax.set_ylabel('Y values')

plt.show()


# ##7. Violin Plots
# 
# Violin plots are similar to box plots, except that they also show the probability density of the data at different values. These plots include a marker for the median of the data and a box indicating the interquartile range, as in the standard box plots.

# In[ ]:


x = [0,3,6,9,13,15,19,23,26,29,33,35,39,41,47,56] 
y = [5,8,13,16,17,20,25,26,30,33,37,39,41,44,48,59]


# In[ ]:


data=[x,y]#First we will combine the collections

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

bp = ax.violinplot(data)

plt.grid(False)
plt.title("Violin Plot")
plt.show()


# ## 8. Plotting Time Series
# 
# A time series is a sequence of numerical data points in successive order. In investing, a time series tracks the movement of the chosen data points, such as a security's price, over a specified period of time with data points recorded at regular intervals.

# In[ ]:


dates = [
    datetime(2021, 3, 10),
    datetime(2021, 3, 13),
    datetime(2021, 3, 14),
    datetime(2021, 3, 15),
    datetime(2021, 3, 16),
    datetime(2021, 3, 17),
    datetime(2021, 3, 18),
    datetime(2021, 3, 19)
]

values = [0,3,4,7,5,3,5,6]


# In[ ]:


plt.title("Time Series")

plt.plot_date(dates, values)

plt.xticks(rotation='vertical')

plt.show() 


# We can add a line to plot

# In[ ]:


plt.title("Time Series")

plt.plot_date(dates, values,linestyle="solid",marker = 'o',ms = 20, mfc = 'r',c="b" )

plt.xticks(rotation='vertical')
plt.xlabel("Dates")
plt.ylabel("Values")
plt.grid(False)

plt.show() 


# ## 9. Box Plot
# 
# In descriptive statistics, a box plot or boxplot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending from the boxes (whiskers) indicating variability outside the upper and lower quartiles. We will use plt.boxplot() function for that.

# In[ ]:


Salaries = [6900,7500,4700,11997,22000,16550,9655,8670,15090,29000,7600,14980,1250]


# In[ ]:


plt.boxplot(Salaries)

plt.title("Box Plot of Salaries")

plt.ylabel("Salaries")

plt.show()


# In[ ]:


# By making notch argument True, we can create notched boxes.

plt.boxplot(Salaries, notch=True)

plt.title("Box Plot of Salaries")

plt.ylabel("Salaries")

plt.show()


# In[ ]:


# We can change the colors

green_diamond = dict(markerfacecolor='g', marker='D')

plt.boxplot(Salaries, notch=True, flierprops=green_diamond)

plt.title("Box Plot of Salaries")

plt.ylabel("Salaries")

plt.show()


# In[ ]:


# We can make showfliers argument False in order to hide Outlier Points

plt.boxplot(Salaries, notch=True, showfliers=False)

plt.title("Box Plot of Salaries")

plt.ylabel("Salaries")

plt.show()


# In[ ]:


# We can plot it horizontal by making vert argument False.

plt.boxplot(Salaries, notch=True, showfliers=False, vert=False)

plt.title("Box Plot of Salaries")

plt.xlabel("Salaries")

plt.show()


# ## 10. Heat Map
# 
# It is often desirable to show data which depends on two independent variables as a color coded image plot. This is often referred to as a heatmap. If the data is categorical, this would be called a categorical heatmap.
# 
# A heat map is a data visualization technique that shows magnitude of a phenomenon as color in two dimensions. The variation in color may be by hue or intensity, giving obvious visual cues to the reader about how the phenomenon is clustered or varies over space. It's generally used to understand correlations between variables.
# 
# Matplotlib's imshow() or heatmap() function makes production of such plots particularly easy. matplotlib.pyplot.pcolormesh() is an alternative function.

# In[ ]:


data = np.random.random(( 6 , 6 )) 
data


# In[ ]:


plt.imshow( data , cmap = 'autumn' ) 
  
plt.title( "2-D Heat Map" ) 
plt.show() 


# In[ ]:


sns.heatmap( data , linewidth = 0.5 , cmap = 'coolwarm' ) 
  
plt.title( "2-D Heat Map" ) 
plt.show() 


# In[ ]:


plt.pcolormesh( data , cmap = 'summer' ) 
  
plt.title( '2-D Heat Map' ) 

