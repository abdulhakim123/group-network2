# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 15:46:54 2023

@author: admin
"""



from sklearn.linear_model import LogisticRegressionCV
print ("edit")
iris=sns.load_dataset("iris")
X=iris.values[:,:4]
y=iris.values[:,4]
train_X,test_X,train_y,test_y=train_test_split(X,y,train_size=0.5,test_size=0.5,random_state=0)
lr=LogisticRegressionCV()
lr.fit(train_X,train_y)
print("Accuracy={:2f}".format(lr.score(test_X,test_y)))
