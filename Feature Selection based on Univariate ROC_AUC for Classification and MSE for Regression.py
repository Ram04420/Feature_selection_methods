#!/usr/bin/env python
# coding: utf-8

# # Feature Selection based on ROC_AUC for Classification and MSE for Regression

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# In[3]:


from sklearn.feature_selection import mutual_info_classif,mutual_info_regression,VarianceThreshold
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


constant_filter = VarianceThreshold(threshold =0.01)
constant_filter.fit(x_train)
x_train_filter = constant_filter.transform(x_train)
x_test_filter = constant_filter.transform(x_test)


# In[10]:


x_train_filter.shape


# In[11]:


x_train_T = x_train_filter.T
x_test_T = x_test_filter.T


# In[12]:


x_train_T = pd.DataFrame(x_train_T)
x_test_T = pd.DataFrame(x_test_T)


# In[13]:


x_train_T.duplicated().sum()


# In[14]:


x_train_T.shape


# In[15]:


duplicated_features = x_train_T.duplicated()


# In[16]:


duplicated_features.sum()


# In[17]:


features_to_keep = [not index for index in duplicated_features]


# In[18]:


x_train_unique = x_train_T[features_to_keep].T
x_test_unique = x_test_T[features_to_keep].T


# In[19]:


x_train_unique.shape, x_test_unique.shape


# ### Now Calculate ROC_AUC

# In[23]:


roc_auc =[]
for features in x_train_unique:
    clf = RandomForestClassifier(n_estimators =100, random_state = 0, n_jobs = -1)
    clf.fit(x_train_unique[features].to_frame(), y_train)
    y_pred = clf.predict(x_test_unique[features].to_frame())
    roc_auc.append(roc_auc_score(y_test, y_pred))


# In[26]:


print(roc_auc)


# In[29]:


roc_values = pd.Series(roc_auc)
roc_values.index = x_train_unique.columns
roc_values.sort_values(ascending = False, inplace = True)


# In[30]:


roc_values


# In[31]:


roc_values.plot.bar()


# In[32]:


sel = roc_values[roc_values>0.5]
sel


# In[34]:


x_train_roc = x_train_unique[sel.index]
x_test_roc = x_test_unique[sel.index]


# ### Build the model and compare the perfomance

# In[36]:


def run_randomforest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy of the score : ')
    print(accuracy_score(y_pred, y_test))


# In[37]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_roc, x_test_roc, y_train, y_test)')


# In[38]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# In[39]:


(3.68-2.33)*100/3.68


# ### feature selection using RMSE in Regression

# In[62]:


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression


# In[63]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[64]:


boston = load_boston()


# In[65]:


print(boston.DESCR)


# In[66]:


x = pd.DataFrame(data = boston.data, columns = boston.feature_names)
y= boston.target


# In[67]:


x.shape, y.shape


# In[68]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 0)


# In[70]:


mse = []
for feature in x_train.columns:
    clf = LinearRegression()
    clf.fit(x_train[feature].to_frame(), y_train)
    y_pred = clf.predict(x_test[feature].to_frame())
    mse.append(mean_squared_error(y_pred, y_test))


# In[71]:


mse


# In[72]:


mse  = pd.Series(mse, index = x_train.columns)
mse.sort_values(ascending = False, inplace = True)
mse


# In[73]:


mse.plot.bar()


# In[78]:


x_train_2 = x_train[['RM', 'LSTAT']]
x_test_2 = x_test[['RM', 'LSTAT']]


# In[80]:


x_train_2.shape, x_test_2.shape


# In[84]:


get_ipython().run_cell_magic('time', '', "model = LinearRegression()\nmodel.fit(x_train_2, y_train)\ny_pred = model.predict(x_test_2)\nprint('r2_score : ', r2_score(y_test, y_pred))\nprint('rmse :', np.sqrt(mean_squared_error(y_test, y_pred)))\nprint('sd of house Prices : ', np.std(y))")


# In[85]:


get_ipython().run_cell_magic('time', '', "model = LinearRegression()\nmodel.fit(x_train, y_train)\ny_pred = model.predict(x_test)\nprint('r2_score : ', r2_score(y_test, y_pred))\nprint('rmse :', np.sqrt(mean_squared_error(y_test, y_pred)))\nprint('sd of house Prices : ', np.std(y))")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




