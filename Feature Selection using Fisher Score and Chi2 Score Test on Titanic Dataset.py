#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score


# In[20]:


titanic = sns.load_dataset('titanic')


# In[21]:


titanic.head()


# In[22]:


titanic.isnull().sum()


# In[23]:


titanic.drop(labels = ['age', 'deck'], axis =1, inplace = True)


# In[24]:


titanic = titanic.dropna()


# In[25]:


titanic.isnull().sum()


# In[26]:


data = titanic[['pclass', 'sex','sibsp', 'parch', 'embarked', 'who', 'alone']].copy()


# In[27]:


data.head()


# In[28]:


data.isnull().sum()


# In[29]:


sex = {'male':0, 'female': 1}
data['sex'] = data['sex'].map(sex)


# In[30]:


data.head()


# In[31]:


ports = {'S':0, 'C':1, 'Q':2}
data['embarked'] = data['embarked'].map(ports)


# In[32]:


who = {'man':0,'woman':1, 'child':2}
data['who'] = data['who'].map(who)


# In[33]:


alone = {True:1, False:0}
data['alone'] = data['alone'].map(alone)


# In[34]:


data.head()


# ### Do F-Score

# In[35]:


x = data.copy()
y = titanic['survived']


# In[36]:


x.shape, y.shape


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[40]:


f_score = chi2(x_train, y_train)


# In[41]:


f_score


# In[46]:


p_values = pd.Series(f_score[1], index = x_train.columns)
p_values.sort_values(ascending = True, inplace = True)


# In[47]:


p_values


# In[48]:


p_values.plot.bar()


# In[49]:


x_train_2 = x_train[['who', 'sex']]
x_test_2 = x_test[['who', 'sex']]


# In[51]:


def run_randomforest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy score is :' , accuracy_score(y_pred, y_test))


# In[52]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_2, x_test_2, y_train, y_test)')


# In[54]:


x_train_3 = x_train[['who', 'sex', 'pclass']]
x_test_3 = x_test[['who', 'sex', 'pclass']]


# In[55]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_3, x_test_3, y_train, y_test)')


# In[56]:


x_train_4 = x_train[['who', 'sex', 'pclass', 'embarked']]
x_test_4 = x_test[['who', 'sex', 'pclass', 'embarked']]


# In[57]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_4, x_test_4, y_train, y_test)')


# In[58]:


x_train_5 = x_train[['who', 'sex', 'pclass', 'embarked', 'alone']]
x_test_5 = x_test[['who', 'sex', 'pclass', 'embarked', 'alone']]


# In[59]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_5, x_test_5, y_train, y_test)')


# In[60]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# In[ ]:




