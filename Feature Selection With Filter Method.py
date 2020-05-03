#!/usr/bin/env python
# coding: utf-8

# ## Univariate
# * Constant Removal
# * Qusi Constant Removal
# * Duplicate Feature Removal

# ### Constant Feature Removal

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold


# In[5]:


train = pd.read_csv('santander-train.csv', nrows = 20000)


# In[6]:


test = pd.read_csv('santander-test.csv')


# In[7]:


test.head()


# In[8]:


train.shape, test.shape


# In[9]:


x = train.drop('TARGET', axis =1)
y = train['TARGET']


# In[10]:


x.shape, y.shape


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# ### Constant Feature Removal

# In[12]:


constant_filter = VarianceThreshold(threshold =0)
constant_filter.fit(x_train)


# In[13]:


constant_filter.get_support().sum()


# In[14]:


constant_list = [not temp for temp in constant_filter.get_support()]
constant_list


# In[15]:


x.columns[constant_list]


# In[16]:


x_train_filter = constant_filter.transform(x_train)
x_test_filter = constant_filter.transform(x_test)


# In[17]:


x_train_filter.shape, x_test_filter.shape, x_train.shape


# ## Quesi Constant Feature Removal
# * These have large output removal from the subset
# * It's over load to Machine Learning Model

# In[18]:


quesi_constant_filter = VarianceThreshold(threshold =0.01)
quesi_constant_filter.fit(x_train_filter)


# In[19]:


quesi_constant_filter.get_support().sum()


# In[20]:


x_train_quesi_filter = quesi_constant_filter.transform(x_train_filter)
x_test_quesi_filter = quesi_constant_filter.transform(x_test_filter)


# In[21]:


x_train_quesi_filter.shape, x_test_quesi_filter.shape, x_train_filter.shape


# ### Removal  Duplicate Features

# In[22]:


x_train_T = x_train_quesi_filter.T
x_test_T = x_test_quesi_filter.T


# In[23]:


type(x_train_T)


# In[24]:


x_train_T = pd.DataFrame(x_train_T)
x_test_T = pd.DataFrame(x_test_T)


# In[25]:


x_train_T.shape, x_test_T.shape


# In[26]:


x_train_T.duplicated().sum()


# In[27]:


duplicated_features = x_train_T.duplicated()
duplicated_features


# In[28]:


feature_to_keep = [not index for index in duplicated_features]


# In[29]:


x_train_unique = x_train_T[feature_to_keep].T
x_test_unique = x_test_T[feature_to_keep].T


# In[30]:


x_train_unique.shape, x_test_unique.shape


# In[31]:


370-228


# ### Bulid ML model and compare the perfomance of the selected features

# In[32]:


def run_randomforest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators =100, random_state = 0, n_jobs =-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('accuracy score on test set : ')
    print(accuracy_score(y_test, y_pred))
    


# In[33]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_unique, x_test_unique, y_train, y_test )')


# In[34]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test )')


# * by selection of feaures we might not be increase the accuracy and reduce the accuracy
# but we can defentily decrease the complexity of model and increace the training time

# In[35]:


(2.69-2.40)*100/2.69


# ## Feature Selection with Filter Method - Correaltion feature Removal

# Summary
# * Feature Space to target correaltion is desired
# * Feature to Feature correlation is not desired
# * if Two Features are high correalted then either feature is redudant
# * correlation feature space is increases model complexity
# * Removing correlated features improving model perfomance
# * different model shows different perfomance over the correlated features

# In[36]:


corrmat = x_train_unique.corr()


# In[37]:


plt.figure(figsize = (10,10))
sns.heatmap(corrmat)


# In[46]:


def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i,j])>threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col


# In[48]:


corr_features = get_correlation(x_train_unique, 0.85)
corr_features


# In[49]:


len(corr_features)


# In[50]:


x_train_uncorr = x_train_unique.drop(labels = corr_features, axis =1)
x_test_uncorr = x_test_unique.drop(labels = corr_features, axis =1)


# In[51]:


x_train_uncorr.shape, x_test_uncorr.shape


# In[54]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_uncorr, x_test_uncorr, y_train, y_test)')


# In[55]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# In[56]:


(3.54-2.42)*100/3.54


# ### Feature Grouping and Feature Importance

# In[72]:


corrmat


# In[73]:


corrdata = corrmat.abs().stack()
corrdata


# In[74]:


corrdata = corrdata.sort_values(ascending = False)
corrdata


# In[75]:


corrdata = corrdata[corrdata>0.85]
corrdata = corrdata[corrdata<1]


# In[76]:


corrdata


# In[78]:


corrdata = pd.DataFrame(corrdata).reset_index()
corrdata.columns = ['features1', 'features2', 'corr_value']


# In[79]:


corrdata


# In[82]:


grouped_feature_list =[]
correlated_groups_list = []
for feature in corrdata.features1.unique():
    if feature not in grouped_feature_list:
        correlated_block = corrdata[corrdata.features1 == feature]
        grouped_feature_list = grouped_feature_list + list(correlated_block.features2.unique())+[feature]
        correlated_groups_list.append(correlated_block)


# In[83]:


len(correlated_groups_list)


# In[86]:


x_train_uncorr.shape, x_train.shape


# In[88]:


for groups in correlated_groups_list:
    print(groups)


# ## Feature Importance based on Tree based classifiers

# In[91]:


important_features =[]
for group in correlated_groups_list:
    features = list(group.features1.unique()) + list(group.features2.unique())
    rf = RandomForestClassifier(n_estimators =100, random_state =0)
    rf.fit(x_train_unique[features], y_train)
    
    importance = pd.concat([pd.Series(features),pd.Series(rf.feature_importances_)], axis = 1)
    importance.columns = ['features', 'importance']
    importance.sort_values(by = 'importance', ascending = False, inplace = True)
    feat = importance.iloc[0]
    important_features.append(feat)


# In[93]:


important_features


# In[97]:


important_features = pd.DataFrame(important_features)


# In[99]:


important_features.reset_index(inplace = True, drop = True)


# In[100]:


important_features


# In[104]:


features_to_consider = set(important_features['features'])


# In[105]:


features_to_discard = set(corr_features) - set(features_to_consider)


# In[106]:


features_to_discard = list(features_to_discard)


# In[109]:


x_train_grouped_unicorr = x_train_unique.drop(labels = features_to_discard, axis =1)
x_train_grouped_unicorr.shape


# In[110]:


x_test_grouped_unicorr = x_test_unique.drop(labels = features_to_discard, axis =1)
x_test_grouped_unicorr.shape


# In[111]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_grouped_unicorr,x_test_grouped_unicorr, y_train, y_test)')


# In[114]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train,x_test, y_train, y_test)')


# In[116]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_uncorr,x_test_uncorr, y_train, y_test)')


# In[113]:


(3.62-2.67)*100/3.62


# In[ ]:




