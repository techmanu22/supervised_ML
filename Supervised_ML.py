#!/usr/bin/env python
# coding: utf-8

# #  Manasi Santosh Mirgal
# 

# ### Data science and Business analytics intern
# 

# ### <b>GRIP @The Sparks Foundation 

# # Task 1 : Prediction using Supervised ML
# ### Task Description :  1 Predict the percentage of a student based on the no. of study hours. 2 What will be predicted score if a student studies for 9.25 hrs/ day?

# In[11]:


import numpy as np


# In[12]:


import matplotlib.pyplot as plt


# ## Loading the Dataset

# In[14]:


link="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
student_score=pd.read_csv(link)
student_score.head()


# In[21]:


student_score.shape


# In[16]:


student_score.describe()


# ## Checking for any missing value in the Dataset.

# In[22]:


student_score.isna().sum()


# ## Plotting the dataset

# In[24]:


student_score.plot(x='Hours',y='Scores',style=".")
plt.title('Study hours v/s Percentage')
plt.xlabel('No. of Study hours')
plt.ylabel('Percentage')
plt.show()


# In[31]:


X=student_score.iloc[:,0].values
Y=student_score.iloc[:,1].values
X=X.reshape(-1,1)


# ## Data Splitting 

# In[52]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[55]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
print("Data trained Successfully")


# ## Creating Model

# In[56]:


#finding slope and intercept for Regression line
slope=regressor.coef_
inter=regressor.intercept_
print("Slope: ",slope,"/n Y-intercept: ",inter)


# In[58]:


#plotting the trained regression's output using Line equation (mx+y)
line_eq=slope*x+inter
plt.scatter(X,Y)
plt.plot(X,line_eq)
plt.show()


# ## Making prediction using Trained data

# In[59]:


y_predict=regressor.predict(X_test)


# In[65]:


df1=pd.DataFrame({'Actual Score': Y_test,'Predicted Score':y_predict})
df1


# In[66]:


print("Hours studied\n",X_test)


# ## Predict the score a of student studying for 9.25hrs/day

# In[80]:


hours = 9.25
sc_predict = regressor.predict(pd.DataFrame({hours}))
print("No of hours = ",hours)
print("Predicted score = ",sc_predict)
print("The predicted score of the student is",sc_predict[0].round(2),"if the student studies for",hours,"hours")
                   


# In[81]:


#line graph to observe the variation between actual percentage and predicted percentage
df1.plot(kind="line")
plt.title("Actual Percentage v/s Predicted Percentage")
plt.xlabel("No. of Student --->" )
plt.ylabel("Percentage scored --->")
plt.show()


# In[83]:


from sklearn import metrics
print('Mean Absolute Error',metrics.mean_absolute_error(Y_test, y_predict))


# In[ ]:




