#!/usr/bin/env python
# coding: utf-8

# ## Feature Dimension Reduction by using LDA & PCA

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# In[19]:


from sklearn.feature_selection import mutual_info_classif,mutual_info_regression,VarianceThreshold
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.preprocessing import StandardScaler


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


# ### Removal Correalted Features

# In[28]:


corramat = x_train_unique.corr()


# In[29]:


def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i,j])>threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col

corr_features = get_correlation(x_train_unique, 0.70)
corr_features


# In[30]:


len(set(corr_features))


# In[31]:


x_train_uncorr = x_train_unique.drop(labels = corr_features, axis =1)
x_test_uncorr = x_test_unique.drop(labels = corr_features, axis =1)


# In[32]:


x_train_uncorr.shape, x_test_uncorr.shape


# ### Feature Dimenssion Reduction by LDA

# In[33]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[38]:


lda = LDA(n_components=1)
x_train_lda = lda.fit_transform(x_train_uncorr, y_train )


# In[39]:


x_train_lda.shape


# In[41]:


x_test_lda = lda.transform(x_test_uncorr)


# In[42]:


x_test_lda.shape


# In[43]:


def run_randomforest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy score is :' , accuracy_score(y_test, y_pred))


# In[44]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_lda, x_test_lda, y_train, y_test)')


# In[45]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train, x_test, y_train, y_test)')


# ### Feature Reduction by PCA

# In[46]:


from sklearn.decomposition import PCA


# In[47]:


pca = PCA(n_components=2, random_state= 42)
pca.fit(x_test_uncorr)


# In[48]:


x_train_pca = pca.transform(x_train_uncorr)
x_test_pca = pca.transform(x_test_uncorr)
x_train_pca.shape, x_train_uncorr.shape


# In[49]:


get_ipython().run_cell_magic('time', '', 'run_randomforest(x_train_pca, x_test_pca, y_train, y_test)')


# In[ ]:


for component in range(1, 79):
    pca = PCA(n_components=component, random_state= 42)
    pca.fit(x_test_uncorr)
    x_train_pca = pca.transform(x_train_uncorr)
    x_test_pca = pca.transform(x_test_uncorr)
    print('Selected comp : ', component )
    run_randomforest(x_train_pca, x_test_pca, y_train, y_test)
    print()


# In[ ]:





# In[ ]:




