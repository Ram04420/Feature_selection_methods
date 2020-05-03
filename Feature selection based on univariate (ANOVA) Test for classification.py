#!/usr/bin/env python
# coding: utf-8

# ## Feature selection based on  Univariate(ANOVA) Test for classification

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# In[21]:


from sklearn.feature_selection import f_classif,f_regression,VarianceThreshold
from sklearn.feature_selection import SelectKBest,SelectPercentile


# In[4]:


data = pd.read_csv('santander-train.csv', nrows = 20000)


# In[5]:


data.head()


# In[6]:


x = data.drop('TARGET', axis =1)
y = data['TARGET']

x.shape, y.shape


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)


# In[8]:


x_train.shape


# ### Removing Constant, Quasi & Duplicated features

# In[9]:


### Removing Constant, Quasi & Duplicated features


# In[10]:


constant_filter = VarianceThreshold(threshold =0.01)
constant_filter.fit(x_train)
x_train_filter = constant_filter.transform(x_train)
x_test_filter = constant_filter.transform(x_test)


# In[11]:


x_train_filter.shape


# In[12]:


x_train_T = x_train_filter.T
x_test_T = x_test_filter.T


# In[13]:


x_train_T = pd.DataFrame(x_train_T)
x_test_T = pd.DataFrame(x_test_T)


# In[14]:


x_train_T.duplicated().sum()


# In[15]:


x_train_T.shape


# In[16]:


duplicated_features = x_train_T.duplicated()


# In[17]:


duplicated_features.sum()


# In[18]:


features_to_keep = [not index for index in duplicated_features]


# In[19]:


x_train_unique = x_train_T[features_to_keep].T
x_test_unique = x_test_T[features_to_keep].T


# In[20]:


x_train_unique.shape, x_test_unique.shape


# # Now do F-Test

# In[22]:


sel = f_classif(x_train_unique, y_train)
sel


# In[26]:


p_value = pd.Series(sel[1])
p_value.index = x_train_unique.columns
p_value.sort_values(ascending = True, inplace = True)


# In[27]:


p_value.plot.bar(figsize = (16,5))


# In[30]:


p_values = p_value[p_value<0.05]


# In[31]:


p_values.index


# In[33]:


x_train_p = x_train_unique[p_values.index]
x_test_p = x_test_unique[p_values.index]


# ### Build the classifier and Compare the perfomance

# In[39]:


def run_randomforest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_pred, y_test))


# In[40]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_p, x_test_p, y_train, y_test)')


# In[41]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# In[42]:


(3.78-2.07)*100/3.78


# In[ ]:




