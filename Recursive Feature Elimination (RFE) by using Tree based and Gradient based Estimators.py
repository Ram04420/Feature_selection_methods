#!/usr/bin/env python
# coding: utf-8

# ## Recursive Feature Elimination (RFE) by using Tree based and Gradient based Estimators

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[5]:


data = load_breast_cancer()


# In[9]:


data.keys()


# In[10]:


print(data.DESCR)


# In[11]:


x = pd.DataFrame(data = data.data, columns = data.feature_names)
x.head()


# In[12]:


y = data.target


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[16]:


x_train.shape, x_test.shape


# ### Feature selection by Feature importance of Random Forest classifier

# In[19]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1))
sel.fit(x_train, y_train)
sel.get_support()


# In[21]:


x_train.columns


# In[22]:


features = x_train.columns[sel.get_support()]
features


# In[23]:


len(features)


# In[24]:


np.mean(sel.estimator_.feature_importances_)


# In[25]:


sel.estimator_.feature_importances_


# In[26]:


x_train_rfc = sel.transform(x_train)
x_test_rfc = sel.transform(x_test)


# In[27]:


def run_randomforest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("accuracy score is :", accuracy_score(y_pred, y_test))


# In[28]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_rfc, x_test_rfc, y_train, y_test)')


# In[29]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# #### Recursive Elimination Feature(REF)

# In[36]:


sel = RFE(RandomForestClassifier(n_estimators =100, n_jobs =-1, random_state = 0), n_features_to_select = 15)
sel.fit(x_train, y_train)
sel.get_support()


# In[38]:


features = x_train.columns[sel.get_support()]


# In[39]:


features


# In[40]:


len(features)


# In[41]:


x_train_rfe = sel.transform(x_train)
x_test_rfe = sel.transform(x_test)


# In[42]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_rfe, x_test_rfe, y_train, y_test)')


# In[43]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# #### Feature Selection by Gradient Boost Tree Importance

# In[44]:


from sklearn.ensemble import GradientBoostingClassifier


# In[45]:


sel = RFE(GradientBoostingClassifier(n_estimators =100, random_state = 0), n_features_to_select = 12)
sel.fit(x_train, y_train)
sel.get_support()


# In[46]:


features = x_train.columns[sel.get_support()]


# In[47]:


features


# In[48]:


len(features)


# In[49]:


x_train_rfg = sel.transform(x_train)
x_test_rfg = sel.transform(x_test)


# In[50]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_rfg, x_test_rfg, y_train, y_test)')


# In[51]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# In[52]:


for index in range(1,31):
    sel = RFE(GradientBoostingClassifier(n_estimators =100, random_state = 0), n_features_to_select = index)
    sel.fit(x_train, y_train)
    x_train_rfg = sel.transform(x_train)
    x_test_rfg = sel.transform(x_test)
    print('Selected Feaures : ', index)
    run_randomforest(x_train_rfg, x_test_rfg, y_train, y_test)
    print()


# In[53]:


sel = RFE(GradientBoostingClassifier(n_estimators =100, random_state = 0), n_features_to_select = 6)
sel.fit(x_train, y_train)
x_train_rfg = sel.transform(x_train)
x_test_rfg = sel.transform(x_test)
print('Selected Feaures : ', 6)
run_randomforest(x_train_rfg, x_test_rfg, y_train, y_test)
print()


# In[55]:


features = x_train.columns[sel.get_support()]
features


# In[58]:


for index in range(1,31):
    sel = RFE(RandomForestClassifier(n_estimators =100, n_jobs= -1, random_state = 0), n_features_to_select = index)
    sel.fit(x_train, y_train)
    x_train_rfg = sel.transform(x_train)
    x_test_rfg = sel.transform(x_test)
    print('Selected Feaures : ', index)
    run_randomforest(x_train_rfg, x_test_rfg, y_train, y_test)
    print()


# In[ ]:




