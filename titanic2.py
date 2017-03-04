#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:51:15 2017

@author: Cino
"""

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

"""Function to update the age"""
def age_impute(train,test):
    for i in [test,train]:
        mean = i['Age'].mean()
        i['Age'] = i['Age'].apply(lambda x: int(mean) if x !=x else x)
        #print   train['Age']
    return train,test
              
"""Function to update th cabin number """
def cabin_number(train,test):
    #print train['Cabin']
    for i in [test,train]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        #train['Cabin_lettter'] =train['Cabin'].apply(lambda x: x[0])
    return train,test

def name_title(test,train):
    for i in [train,test]:
             i['Name_title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    return train,test

def drop(train, test, cols = ['Ticket', 'SibSp', 'Parch']):
    for i in [train, test]:
        for z in cols:
            del i[z]
    return train, test

def trim_sex(test,train):
    for i in [train,test]:
        i['Sex'] = i['Sex'].apply(lambda x: x[0])
    return train,test             

train = pd.read_csv(os.path.join("data","train.csv"))
test = pd.read_csv(os.path.join("data","test.csv"))
train,test = age_impute(train,test)
#train,test = cabin_number(train,test)
#train,test  = name_title(train,test)
#train,test = trim_sex(train,test)
#train,test = drop(train,test,cols=['Ticket','Name','Parch','Cabin'])
#333333print train

rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=50,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
#print train.iloc[:,2:]
y = train.pop('Survived')
numeric_variables = list(train.dtypes[train.dtypes != "object"].index)
print y
rf.fit(train[numeric_variables], y)
print "%.4f" % rf.oob_score_ 
"""
rf = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1,5,10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train[numeric_variables], train.iloc[:, 1])


# #### Inspect best parameters


print(gs.best_score_)
print(gs.best_params_)

"""
print train.dtypes.index
print test.dtypes.index
numeric_variables_test= list(test.dtypes[test.dtypes != "object"].index)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
#print test
predictions = rf.predict(test[numeric_variables_test])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('data', 'y_test51.csv'), sep=",", index = False)


#


                        
