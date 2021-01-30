#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Internet-Firewall
# dataset link : http://archive.ics.uci.edu/ml/datasets/Internet+Firewall+Data
# email : amirsh.nll@gmail.com


# In[2]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('firewall.csv')
df


# In[3]:


df.info() 


# In[4]:


y = df['Action'].values
y = y.reshape(-1,1)
x_data = df.drop(['Action'],axis = 1)
print(x_data)


# In[5]:


sns.countplot(x='Action',data=df,palette='hls')
plt.show()


# In[6]:


#normalize data

x = (x_data - np.min(x_data)) / (np.max(x_data) / np.min(x_data)).values
x.head()


# In[7]:


#data train & data test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state= 300)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# In[8]:


#decision tree classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier()
dt = dt.fit(x_train,y_train)
y_forecast=dt.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, dt.predict(x_test)))
print('accuracy:{:.4f}'.format(dt.score(x_test, y_test)))

from sklearn import tree
plt.figure(figsize=(30,30))
temp = tree.plot_tree(dt.fit(x,y), fontsize=10)
plt.show()


# In[9]:


#Nave Bayes Classifier

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb = nb.fit(x_train, y_train.ravel())
y_forecast=nb.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, nb.predict(x_test)))
print('Nave Bayes Classifier{:.4f}'.format(nb.score(x_test, y_test)))


# In[10]:


#Logistic Regreession Classifier

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs')
lr = lr.fit(x_train, y_train.ravel())
y_forecast=lr.predict(x_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, lr.predict(x_test)))
print('accuracy:{:.4f}'.format(lr.score(x_test, y_test)))



# In[ ]:


#Knn Classifier
from sklearn.neighbors import KNeighborsClassifier
K = 1
knn = KNeighborsClassifier(n_neighbors=K)
knn = knn.fit(x_train,y_train.ravel())
print("k = {}neighbors , knn test:{}".format(K, knn.score(x_test, y_test)))
print("knn = {}neighbors , knn train:{}".format(K, knn.score(x_train, y_train)))

ran = np.arange(1,40)
train_list = []
test_list = []
for i,each in enumerate(ran):
    knn = KNeighborsClassifier(n_neighbors=each)
    knn = knn.fit(x_train, y_train.ravel())
    test_list.append(knn.score(x_test, y_test))
    train_list.append(knn.score(x_train, y_train))
    
print("best test {} , k={}".format(np.max(test_list),test_list.index(np.max(test_list))+1))
print("best train {} , k={}".format(np.max(train_list),train_list.index(np.max(train_list))+1))


# In[ ]:


#mlp classifier
from sklearn.neural_network import MLPClassifier
clfm = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
clfm.fit(x_train, y_train.ravel())
y_predm = clfm.predict(x_test)
print("ACCTURACY:", metrics.accurecy_score(y_test, y_predm))
print(classification_report(y_test, clfk.predict(x_test)))
print("mlp:", clfk.score(x_test, y_test))

