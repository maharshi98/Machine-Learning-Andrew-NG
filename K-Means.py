#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import linalg

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
 
import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')


# In[2]:


data1 = loadmat('ex7data2.mat')
data1.keys()


# In[3]:


X1 = data1['X']
print('X1:', X1.shape)


# In[4]:


km1 = KMeans(3)
km1.fit(X1)


# In[5]:


plt.scatter(X1[:,0], X1[:,1], s=40, c=km1.labels_, cmap=plt.cm.prism) 
plt.title('K-Means Clustering Results with K=3')
plt.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);


# In[6]:


img = plt.imread('bird_small.png')
img_shape = img.shape
img_shape


# In[7]:


A = img/255


# In[8]:


AA = A.reshape(128*128,3)
AA.shape


# In[9]:


km2 = KMeans(16)
km2.fit(AA)


# In[10]:


B = km2.cluster_centers_[km2.labels_].reshape(img_shape[0], img_shape[1], 3)


# In[11]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,9))
ax1.imshow(img)
ax1.set_title('Original')
ax2.imshow(B*255)
ax2.set_title('Compressed, with 16 colors')

for ax in fig.axes:
    ax.axis('off')


# In[13]:


data2 = loadmat('ex7data1.mat')
data2.keys()


# In[14]:


X2 = data2['X']
print('X2:', X2.shape)


# In[15]:


scaler = StandardScaler()
scaler.fit(X2)


# In[16]:


U, S, V = linalg.svd(scaler.transform(X2).T)
print(U)
print(S)


# In[17]:


plt.scatter(X2[:,0], X2[:,1], s=30, edgecolors='b',facecolors='None', linewidth=1);
plt.gca().set_aspect('equal')
plt.quiver(scaler.mean_[0], scaler.mean_[1], U[0,0], U[0,1], scale=S[1], color='r')
plt.quiver(scaler.mean_[0], scaler.mean_[1], U[1,0], U[1,1], scale=S[0], color='r');


# In[ ]:




