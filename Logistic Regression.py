#Logistic Regression
#Author - Maharshi Doshi
#Tool used - Jupyter notebook

# In[3]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[4]:

# function to load data from text file
def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:6,:])
    return(data)


# In[5]:


data = loaddata('ex2data1.txt', ',')


# In[86]:

#defining arrays X and Y from data
X = data[:,:2]
y = data[:,2]
print(X)


# In[60]:


lr=0.2		#learning rate (more learning rate might get faster results
num_iter=5000


# In[116]:


def sigmoid(z):				#equation of sigmoid
    return 1 / (1 + np.exp(-z))
	
def loss(h, y):				#formula for loss function
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def log_fit(X, y):				#determine optimum values of theta
	intercept = np.ones((X.shape[0], 1))
	X = np.concatenate((intercept, X), axis=1)
        
    # weights initialization
    theta = np.zeros(X.shape[1])
    print(theta.shape)
	
    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= lr * gradient		

        z = np.dot(X, theta)
        h = sigmoid(z)
        loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

        if(i % 10000 == 0):
            print(f'loss: {loss} \t')

#probability/accuracy of model			
def predict_prob(X):
	intercept = np.ones((X.shape[0], 1))
	X = np.concatenate((intercept, X), axis=1)
	print(X.shape)
	return sigmoid(np.dot(X,np.array(theta.shape,1)))

def predict(X):
    return predict_prob(X).round()


# In[117]:


log_fit(X,y)


# In[118]:


preds = predict(X)
(preds == y).mean()


# In[85]:

#prints the accuracy
print('Train accuracy {}%'.format(100*sum(preds == y.ravel())/preds.size))


# In[74]:


theta


# In[75]:

#plot graph
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = predict_prob(grid,theta).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');


# In[ ]:





# In[ ]:




