#Linear Regression using Gradient Descent
#Author - Maharshi Doshi
#Tool used - Jupyter notebook

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d


# In[15]:

#data loaded from a text file
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = np.c_[np.ones(data.shape[0]),data[:,0]]			#first column assigned with ones
y = np.c_[data[:,1]]		#y assigned with 2nd column


# In[17]:


plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)


# In[31]:


def Cost(X, y, theta=[[0],[0]]):
    m = y.size							#size of y	
    J = 0
    h = X.dot(theta)
    J = 1/(2*m)*np.sum(np.square(h-y))		#calculate the 
    return(J)


# In[32]:


Cost(X,y)	#Cost function called


# In[35]:

#gradient descent to reach optimal solution
def gradDesc(X, y, theta=[[0],[0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_prev = np.zeros(num_iters)
    print(J_prev.shape)
    
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_prev[iter] = Cost(X, y, theta)
    return(theta, J_prev)


# In[36]:

#calculate parameter value and Cost
theta,Cost_J = gradDesc(X, y)


# In[37]:


x_plot = np.arange(5,23)
y_plot = theta[0]+theta[1]*x_plot

# plot gradient descent
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(x_plot,y_plot, label='Linear regression')

