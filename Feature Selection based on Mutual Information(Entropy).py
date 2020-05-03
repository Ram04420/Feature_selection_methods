#!/usr/bin/env python
# coding: utf-8

# # Feature Selection based on Mutual Information (Entropy) Gain for Classification and Regression

# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[74]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[75]:


from sklearn.feature_selection import mutual_info_classif,mutual_info_regression,VarianceThreshold
from sklearn.feature_selection import SelectKBest,SelectPercentile


# In[76]:


data = pd.read_csv('santander-train.csv', nrows = 20000)


# In[77]:


data.head()


# In[78]:


x = data.drop('TARGET', axis =1)
y = data['TARGET']

x.shape, y.shape


# In[79]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)


# In[80]:


x_train.shape


# ### Removing Constant, Quasi & Duplicated features

# In[81]:


constant_filter = VarianceThreshold(threshold =0.01)
constant_filter.fit(x_train)
x_train_filter = constant_filter.transform(x_train)
x_test_filter = constant_filter.transform(x_test)


# In[82]:


x_train_filter.shape


# In[83]:


x_train_T = x_train_filter.T
x_test_T = x_test_filter.T


# In[84]:


x_train_T = pd.DataFrame(x_train_T)
x_test_T = pd.DataFrame(x_test_T)


# In[85]:


x_train_T.duplicated().sum()


# * if drop method used for removing duplicates from train set, we have to again apply the drop method test set,
# * so we apply another method

# In[86]:


x_train_T.shape


# In[87]:


duplicated_features = x_train_T.duplicated()


# In[88]:


duplicated_features.sum()


# In[89]:


features_to_keep = [not index for index in duplicated_features]


# In[90]:


x_train_unique = x_train_T[features_to_keep].T
x_test_unique = x_test_T[features_to_keep].T


# In[91]:


x_train_unique.shape, x_test_unique.shape


# ### Calculate the MI

# In[92]:


mi = mutual_info_classif(x_train_unique, y_train)


# In[93]:


len(mi)


# In[94]:


mi


# In[95]:


mi = pd.Series(mi)


# In[96]:


mi.index = x_train_unique.columns


# In[97]:


mi.sort_values(ascending = False, inplace = True)


# In[98]:


mi.plot.bar()


# In[100]:


sel = SelectPercentile(mutual_info_classif, percentile=10).fit(x_train_unique, y_train)
x_train_unique.columns[sel.get_support()]


# In[103]:


len(x_train_unique.columns[sel.get_support()])


# In[104]:


help(sel)


# In[105]:


x_train_mi = sel.transform(x_train_unique)
x_test_mi = sel.transform(x_test_unique)


# In[107]:


x_train_mi.shape


# ### Build the model and compare the perfomance

# In[108]:


def run_randomforest(x_train,x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators = 100, n_jobs =-1, random_state = 0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('accuracy on test set : ')
    print(accuracy_score(y_test, y_pred))


# In[110]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_mi, x_test_mi, y_train, y_test)')


# In[112]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# In[113]:


(2.72-1.12)*100/2.72


# ### Mutual Inforamation Gain in Regression

# In[114]:


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[120]:


boston = load_boston()


# In[121]:


print(boston.DESCR)


# In[123]:


x = pd.DataFrame(data = boston.data, columns = boston.feature_names)


# In[124]:


x.head()


# In[125]:


y = boston.target


# In[126]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[127]:


mi = mutual_info_regression(x_train, y_train)
mi = pd.Series(mi)
mi.index = x_train.columns
mi.sort_values(ascending = False , inplace = True)


# In[128]:


mi


# In[129]:


mi.plot.bar()


# In[131]:


sel = SelectKBest(mutual_info_regression, k = 9).fit(x_train, y_train)
x_train.columns[sel.get_support()]


# In[132]:


len(x_train.columns[sel.get_support()])


# In[133]:


model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[134]:


r2_score(y_test, y_pred)


# In[135]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[136]:


np.std(y)


# In[137]:


x_train_9 = sel.transform(x_train)
x_test_9 = sel.transform(x_test)


# In[138]:


x_train_9.shape, x_test_9.shape


# In[140]:


model = LinearRegression()
model.fit(x_train_9, y_train)
y_pred = model.predict(x_test_9)
r2_score(y_test, y_pred)


# In[141]:


np.sqrt(mean_squared_error(y_test, y_pred))

