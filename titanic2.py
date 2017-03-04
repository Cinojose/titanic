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
import numpy as np

"""Function to update the age"""
def age_impute(train,test):
    for i in [test,train]:
        mean = i['Age'].mean()
        i['Age'] = i['Age'].apply(lambda x: int(mean) if x !=x else x)
        #print   train['Age']
    return train,test

def embark_impute(train,test):
    for i in (train,test):
        i['Embarked'] = i['Embarked'].apply(lambda x: 'P' if x!=x else x)
        i['Embarked'] =  np.where((i['Embarked']) == "S" , 1,
                           np.where((i["Embarked"]) == "C", 2,
                                    np.where((i['Embarked'])=="P",3,4)))
    return train,test
    
              
"""Function to update th cabin number """
def cabin_number(train,test):
    #print train['Cabin']
    for i in [test,train]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        #train['Cabin_lettter'] =train['Cabin'].apply(lambda x: x[0])
    return train,test

""" TODO: fix the bug """
def name_title(train,test):
    for i in [train,test]:
             i['Name_title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    pd.concat((test, pd.get_dummies(test['Name_title'])),axis = 1)
    pd.concat((train, pd.get_dummies(train['Name_title'])),axis = 1)
    return train,test

def drop(train, test, cols = ['Ticket', 'SibSp', 'Parch']):
    for i in [train, test]:
        for z in cols:
            del i[z]
    return train, test

def famsize(train,test):
    for i in [train,test]:
        #i['Sibsp'] = i['Sibsp'].apply(lambda x: x)
        i['famsize'] =  np.where((i['SibSp']+i['Parch']) == 0 , 1,
                           np.where((i['SibSp']+i['Parch']) <= 3, 2, 3))
        del i['SibSp']
        del i['Parch']
    return train,test

def trim_sex(train,test):
    for i in [train,test]:
        i['Sex'] = i['Sex'].apply(lambda x: x[0])
        i['sex_num'] = i['Sex'].apply(lambda x: 1 if x =='m' else 0)
        del i['Sex']
    return train,test  

"""TODO : continue to catgorize """
def do_fare(train,test):
    for i in [train,test]:
        mean = i['Fare'].mean()
        i['Fare'] = i['Fare'].apply(lambda x: mean if x !=x else x)        
    train['Fare_num'] = pd.cut(train['Fare'],4,labels=[0,1,2,3])
    test['Fare_num'] = pd.cut(test['Fare'],4,labels=[0,1,2,3])
    return train,test
          

train = pd.read_csv(os.path.join("data","train.csv"))
test = pd.read_csv(os.path.join("data","test.csv"))
train,test = age_impute(train,test)
train,test = famsize(train,test)
#train,test = cabin_number(train,test)
train,test  = name_title(train,test)
train,test = trim_sex(train,test)
#train,test = drop(train,test,cols=['Ticket','Name','Parch','Cabin'])
train,test = embark_impute(train,test)
train,test = do_fare(train,test)
#print train['Fare']
######3333333print train['Name_title']

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
##print y
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


                        
