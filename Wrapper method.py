#!/usr/bin/env python
# coding: utf-8

# # Step Forward Selection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[5]:


from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


# In[6]:


data = load_wine()


# In[7]:


data.keys()


# In[8]:


print(data.DESCR)


# In[9]:


x = pd.DataFrame(data.data)
y = data.target


# In[10]:


x.columns =data.feature_names
x.head()


# In[11]:


x.isnull().sum()


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[13]:


x_train.shape, x_test.shape


# ### Step Forward feature Selection (SFS)

# In[17]:


sfs = SFS(RandomForestClassifier(n_estimators =100, n_jobs = -1, random_state =0),
         k_features=7,
         forward=True,
         verbose=2,
         n_jobs=-1,
         scoring='accuracy',
         cv=4,
         floating=False,
         ).fit(x_train, y_train)


# In[18]:


sfs.k_feature_names_


# In[19]:


sfs.k_feature_idx_


# In[20]:


sfs.k_score_


# In[21]:


pd.DataFrame.from_dict(sfs.get_metric_dict()).T


# In[27]:


sfs = SFS(RandomForestClassifier(n_estimators =100, n_jobs = -1, random_state =0),
         k_features=(1,8),
         forward=True,
         verbose=2,
         n_jobs=-1,
         scoring='accuracy',
         cv=4,
         floating=False,
         ).fit(x_train, y_train)


# In[23]:


sfs.k_score_


# In[24]:


sfs.k_feature_names_


# ### Step Backward Selection(SBS)

# In[28]:


sbs = SFS(RandomForestClassifier(n_estimators =100, n_jobs = -1, random_state =0),
         k_features=(1,8),
         forward=False,
         verbose=2,
         n_jobs=-1,
         scoring='accuracy',
         cv=4,
         floating=False,
         ).fit(x_train, y_train)


# In[31]:


sbs.k_score_


# In[32]:


sbs.k_feature_names_


# In[ ]:





# ### Exhaustive Feature Selection(EFS)

# In[33]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


# In[34]:


efs = EFS(RandomForestClassifier(n_estimators = 100, n_jobs=-1, random_state = 0),
         min_features=4,
         max_features=5,
         scoring='accuracy',
         cv=None,
         n_jobs=-1,
         ).fit(x_train, y_train)


# In[ ]:


c(13,4)+c(13,5) = 715+1287


# In[36]:


715+1287


# In[37]:


help(efs)


# In[38]:


efs.best_score_


# In[39]:


efs.best_feature_names_


# In[40]:


efs.best_idx_


# In[41]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[43]:


plot_sfs(efs.get_metric_dict(), kind='std_dev')


# In[ ]:




