#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np


# In[4]:


pd = pd.read_csv('Crop_Recommendation.csv')


# In[6]:


pd.head()


# In[7]:


pd.tail()


# In[9]:


pd. describe()


# In[11]:


pd.count()


# In[13]:


df = pd


# In[16]:


df.isnull().sum()


# In[27]:


df['Crop'].unique() 


# In[31]:


eachcrops =df.Crop.value_counts()


# In[32]:


eachcrops


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[59]:


fig, axes = plt.subplots(nrows = 2,ncols = 4,figsize = (18,8))

sns.violinplot(data = df, y = 'Nitrogen',ax = axes[0,0])
sns.violinplot(data = df, y='Phosphorus', ax = axes[0,1])
sns.violinplot(data= df, y = 'Potassium' , ax = axes[0,2])
sns.violinplot(data = df , y = 'Temperature', ax = axes[0,3])
sns.violinplot(data = df, y = 'Humidity',ax = axes[1,0])
sns.violinplot(data = df, y = 'pH_Value',ax = axes[1,1])
sns.violinplot(data = df, y = 'Rainfall',ax = axes[1,2])
axes[1,3].axis('off')
plt.tight_layout()

plt.show()


# In[61]:


#The values for all the features vary in different ranges. Notable points from these plots are:

#The pH values are majorly near to the neutral value of 7
#Different levels of Nitrogen and Phosporus content have been included in the dataset
#Ptassium levels are either very low or very high
#For further analysis we visualize the variation of these features based on the crops.


# In[64]:


fig, ax = plt.subplots(figsize=(25,8))
sns.boxplot(data = df, y = 'Nitrogen', x = 'Crop')
plt.show()


# In[65]:


fig , ax = plt.subplots(figsize=(25,8))
sns.boxplot(data = df, y = 'Phosphorus' , x='Crop')
plt.show()


# In[66]:


fig , ax = plt.subplots(figsize=(25,8))
sns.boxplot(data = df, y = 'Potassium' , x='Crop')
plt.show()


# In[67]:


fig , ax = plt.subplots(figsize=(25,8))
sns.boxplot(data = df, y = 'Temperature' , x='Crop')
plt.show()


# In[68]:


fig , ax = plt.subplots(figsize=(25,8))
sns.boxplot(data = df, y = 'Humidity' , x='Crop')
plt.show()


# In[69]:


fig, ax = plt.subplots(figsize=(25,8))
sns.boxplot(data = df, y = 'pH_Value', x = 'Crop')
plt.show()


# In[70]:


fig, ax = plt.subplots(figsize=(25,8))
sns.boxplot(data = df, y = 'Rainfall', x = 'Crop')
plt.show()


# In[71]:


label_encoder = LabelEncoder()
df['Crop'] = label_encoder.fit_transform(df['Crop'])

X = df.drop(['Crop'],axis=1)
Y = df['Crop']


# In[74]:


fig , ax = plt.subplots(figsize=(15,10))
sns.heatmap(X.corr(), annot = True)
ax.set(xlabel = 'feature')
ax.set(ylabel= 'feature')

plt.title('correlation b/t different feature')
plt.show()


# In[80]:


X_train,X_test, y_train,y_test = train_test_split(X, Y, test_size = 0.2, shuffle = True, random_state = 0)


# In[81]:


model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)

y_pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_pred_lr,y_test)


# In[82]:


model_svc = SVC()
model_svc.fit(X_train,y_train)

y_pred_svc = model_svc.predict(X_test)
acc_svc = accuracy_score(y_pred_svc,y_test)


# In[83]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,y_train)

y_pred_dt = model_dt.predict(X_test)
acc_dt = accuracy_score(y_pred_dt,y_test)


# In[84]:


model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)

y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_pred_rf,y_test)


# In[86]:


print("The accuracy for Logistic Regression is : {0:0.4f}".format(acc_lr))


# In[87]:


print("The accuracy for Logistic Regression is : {0:0.4f}".format(acc_lr))
print("The accuracy for Support Vector Machine is : {0:0.4f}".format(acc_svc))
print("The accuracy for Decision Tree is : {0:0.4f}".format(acc_dt))
print("The accuracy for Random Forest is : {0:0.4f}".format(acc_rf))


# In[ ]:




