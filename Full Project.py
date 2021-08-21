#!/usr/bin/env python
# coding: utf-8

# In[255]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime as dt
import itertools
import csv
import os
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from IPython.display import display
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
    
from __future__ import division
from __future__ import print_function
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from importlib import reload

print ("pandas version ",pd.__version__)


# In[3]:


# Load ucldata into python
ucldata= pd.read_csv(r'/Users/andrea/Downloads/ucl-soccer-1955-2019.csv')


# In[4]:


# make sure data was uploaded and is correct
ucldata.head()
ucldata.tail()


# In[430]:


# Download champs data 
champs= pd.read_csv(r'/Users/andrea/Downloads/champs.csv', encoding='unicode_escape')
champs.head()
champs.tail()


# In[431]:


# assess champs data 
champs.shape
champs.info()
champs.isnull().sum()


# In[432]:


# Data cleaning
champs.isnull()
champs["round"].fillna("No round", inplace = True) 
champs["pens"].fillna("No penalties", inplace = True)
champs["aet"].fillna("No extra time", inplace = True)
champs["HT"].fillna("No info", inplace = True)
champs["leg"].fillna("No info", inplace = True)
champs["aethgoal"].fillna("No info", inplace = True)
champs["aetvgoal"].fillna("No info", inplace = True)
champs["tiewinner"].fillna("No tie", inplace = True)
# Confirming the data does not have NA values
champs.isnull().sum()


# In[433]:


# Check type for each column after cleaning
champs.info()


# In[113]:


# Check missing values for ucldata
ucldata.isnull()
ucldata.isnull().sum()


# In[10]:


# Exploratory Data Analysis UCL Data
ucldata.info()
ucldata.shape


# In[11]:


# EDA cont.
ucldata.describe()
ucldata['club'].value_counts().head(5)


# In[12]:


# Top 5 nations that teams come from
ucldata['nation'].value_counts().head(5)


# In[13]:


# Top 10 teams in the final 
ucldata['club'].value_counts().head(10)


# In[14]:


# EDA champs data
champs['home'].value_counts().head(10)


# In[15]:


#Top 10 visiting teams
champs['visitor'].value_counts().head(10)


# In[244]:


# Top 5 score outcomes for all games
champs['FT'].value_counts().head(5).sort_values().plot(kind = 'barh', color='lightblue')
plt.title('Top 5 Score Outcomes')

plt.show()


# In[237]:


# Countries represented in tournament since 1955
champs.hcountry.value_counts(normalize=True)
champs.hcountry.value_counts(normalize=True).plot.pie()

plt.show()


# In[18]:


# Taking a look at the aggregate values for home teams
champs.FTagg_home.describe()


# In[19]:


# Aggregate values for visitor teams which has a lower mean
champs.FTagg_visitor.describe()


# In[32]:


# scatter plt of aggregate goals for home and visitor
plt.scatter(champs.FTagg_home,champs.FTagg_visitor)
plt.show()


# In[37]:


from pandas import DataFrame
from numpy import array
from sklearn.ensemble import RandomForestClassifier


# In[455]:


import matplotlib as mp
from matplotlib import colors as mcolors

# Plot producing the top 5 teams in the final which were either winner or runner up
res=ucldata.groupby(by='club', as_index=False).count()
res = res.sort_values(by='position', ascending=False).head()
plt.title('Teams Based on Winner and Runner Up')
ax = sns.barplot(x='club', y='position', data=res, color ='thistle')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


plt.show()


# In[434]:


# In order to run the model these columns needed to be deleted to not disrupt it. They are object types like data
champs = champs.drop(champs.columns[[0,1,2,3,9,20]], axis=1)


# In[462]:


# Checking to see the new dataset 
champs.head()


# In[463]:


### Preparing Data


# In[199]:




x_all = champs.drop(['FT', 'home', 'visitor', 'hcountry', 'vcountry'], 1)
y_all = champs['FT']


# In[200]:


from sklearn.preprocessing import scale


# In[201]:


cols=[['hgoal', 'vgoal', 'totvgoal', 'tothgoal', 'totagg_home', 'totagg_visitor']]
for col in cols:
    x_all[col] = scale(x_all[col])


# In[202]:




def preprocess_features(X):
    '''Preprocesses the soccer data and converts catagorical variables into dummy variables.'''
    
    output = pd.DataFrame(index = X.index)
    
    for col, col_data in X.iteritems():
        
        
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
    

            
        output = output.join(col_data)
    return output 

x_all = preprocess_features(x_all)
print ("Processed feature columns ({} total features):\n{}".format(len(x_all.columns), list(x_all.columns)))


# In[203]:


print ("\nFeature values:")
display(x_all.head())


# In[351]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 100, random_state = 2)


# In[352]:


from time import time

from sklearn.metrics import f1_score

def train_classifier(clf, x_train, y_train):
    ''' Fits a classifier to the training data.'''
    
    start = time()
    clf.fit(x_train, y_train)
    end = time()
    
    print ("Trained model in {:.4f} seconds".format(end - start))

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score.'''
    
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, pos_label='H', average='micro'), sum(target == y_pred) / float(len(y_pred))

def train_predict(clf, x_train, y_train, x_test, y_test):
    '''Train and predict using a classifier based on F1 score.'''
    
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(x_train)))
    
    train_classifier(clf, x_train, y_train)
    
    f1, acc = predict_labels(clf, x_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for test set: {:.4f}, {:.4f}.".format(f1, acc))


# In[353]:


import xgboost as xgb

clf_A = LogisticRegression(random_state = 42, max_iter=2000)
clf_B = SVC(random_state = 912, kernel='rbf')

clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, x_train, y_train, x_test, y_test)
print ('')
train_predict(clf_B, x_train, y_train, x_test, y_test)
print('')
train_predict(clf_C, x_train, y_train, x_test, y_test)
print('')


# In[212]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer

parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth' : [3],
               'min_child_weight' : [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }

clf = xgb.XGBClassifier(seed=2)

f1_scorer = make_scorer(f1_score,pos_label='H', average='micro')

grid_obj = GridSearchCV(clf, scoring=f1_scorer, 
                             param_grid=parameters,
                             cv=5)
grid_obj = grid_obj.fit(x_train,y_train)

clf = grid_obj.best_estimator_
print (clf)

f1, acc = predict_labels(clf, x_train, y_train)
print ("F1 score and accuracy score for trainig set: {:.4f} , {:.4f}.".format(f1 , acc))

f1, acc = predict_labels(clf, x_test, y_test)
print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


# In[214]:


prediction=clf.predict(x_train)


# In[215]:


print(prediction)


# In[216]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer

parameters = { 'learning_rate' : [0.03],
               'n_estimators' : [20],
               'max_depth' : [5],
               'min_child_weight' : [5],
               'gamma':[0.2],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-2]
             }

clf = xgb.XGBClassifier(seed=2)

f1_scorer = make_scorer(f1_score,pos_label='H', average='micro')

grid_obj = GridSearchCV(clf, 
                        scoring=f1_scorer, 
                             param_grid=parameters,
                             cv=5)

grid_obj = grid_obj.fit(x_all,y_all)

clf = grid_obj.best_estimator_
print (clf)

f1, acc = predict_labels(clf, x_train, y_train)
print ("F1 score and accuracy score for trainig set: {:.4f} , {:.4f}.".format(f1 , acc))


# In[454]:


# Save Model using joblib
import joblib

filename = 'finalized_model.sav'
joblib.dump(clf_C, filename)


loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print(result)


# In[428]:


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

numpy.vstack((y_train, yhat_binary)).T


# In[441]:


yhat = clf_A.predict(x_test)
yhat


# In[442]:


home = champs.loc[ :99, 'home']
visitor = champs.loc [:99, 'visitor']


print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)


# In[443]:


from sklearn import metrics

yhat_binary =clf_A.predict(x_train)
print("Accuracy:",metrics.accuracy_score(y_train, yhat_binary))


# In[444]:


import sys
import numpy

yhat_binary = clf_A.predict(x_train)
print('Accuracy:', metrics.accuracy_score(y_train, yhat_binary))


# In[445]:


df = pd.DataFrame({'home': home, 'Predictions':yhat, 'visitor': visitor})
df


# In[446]:


yhat_prob = clf_A.predict(x_test)
yhat_prob


# In[448]:


df2 = pd.DataFrame ({'home': home, 'Prediction': yhat_prob[:,1], 'visitor': visitor})


# In[375]:


champs.columns


# In[376]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:




