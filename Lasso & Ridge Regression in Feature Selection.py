#!/usr/bin/env python
# coding: utf-8

# ## Lasso & Ridge Regression In Feature Selection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score


# In[96]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel


# In[65]:


titanic = sns.load_dataset('titanic')
titanic.head()


# In[66]:


titanic.isnull().sum()


# In[67]:


titanic.drop(labels = ['age', 'deck'], axis = 1, inplace = True)


# In[68]:


titanic = titanic.dropna()


# In[69]:


titanic.isnull().sum()


# In[70]:


data = titanic[['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'who', 'alone']].copy()


# In[71]:


data.head()


# In[72]:


data.isnull().sum()


# In[73]:


sex = {'male': 0, 'female': 1}
data['sex'] = data['sex'].map(sex)


# In[74]:


data.head()


# In[75]:


ports = {'S': 0, 'C':1, 'Q':2}
data['embarked'] = data['embarked'].map(ports)


# In[76]:


who = {'man':0, 'woman': 1, 'child':2}
data['who'] = data['who'].map(who)


# In[77]:


alone = {True: 1, False:0}
data['alone'] = data['alone'].map(alone)


# In[78]:


data.head()


# In[79]:


x = data.copy()
y = titanic['survived']


# In[80]:


x.shape, y.shape


# In[81]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# ### Estimation of coeffiecnts of Linear Regression

# In[82]:


sel = SelectFromModel(LinearRegression())


# In[83]:


sel.fit(x_train, y_train)


# In[84]:


sel.get_support()


# In[85]:


sel.estimator_.coef_


# In[86]:


mean = np.mean(np.abs(sel.estimator_.coef_))


# In[87]:


mean


# In[88]:


np.abs(sel.estimator_.coef_)


# In[89]:


features = x_train.columns[sel.get_support()]
features


# In[90]:


x_train_reg = sel.transform(x_train)
x_test_reg = sel.transform(x_test)


# In[91]:


x_train_reg.shape, x_test_reg.shape


# In[93]:


def run_randomforest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy score is :' , accuracy_score(y_test, y_pred))


# In[94]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_reg, x_test_reg, y_train, y_test)')


# In[95]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# ### Logistic Regression Coefficient with L1 Regularization

# In[99]:


sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.05, solver = 'liblinear'))
sel.fit(x_train, y_train)
sel.get_support()


# In[100]:


sel.estimator_.coef_


# In[101]:


x_train_l1 = sel.transform(x_train)
x_test_l1 = sel.transform(x_test)


# In[102]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_l1, x_test_l1, y_train, y_test)')


# ### L2 Regularization

# In[103]:


sel = SelectFromModel(LogisticRegression(penalty = 'l2', C = 0.05, solver = 'liblinear'))
sel.fit(x_train, y_train)
sel.get_support()


# In[104]:


sel.estimator_.coef_


# In[105]:


x_train_l2 = sel.transform(x_train)
x_test_l2 = sel.transform(x_test)


# In[106]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_l2, x_test_l2, y_train, y_test)')


# In[ ]:




