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
import seaborn as sns
import matplotlib.pyplot as plt

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
    name_titles = ["Miss.","Mr.","Mrs.","Ms.","Col.","Dr.","Master."]
    for i in [train,test]:
             name_title = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: "Mr." if x.split()[0] not in name_titles else x.split()[0])
             i['Name_title'] = name_title
    test = pd.concat([test,pd.get_dummies(test['Name_title'], prefix="Name_title")],axis=1)
    train = pd.concat([train,pd.get_dummies(train['Name_title'], prefix="Name_title")],axis=1)
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

def do_visualize(train):
    avg_survived = train[['Name_title','Survived']].groupby(['Name_title'], as_index=False).mean()
    fig,axis1= plt.subplots(1,1,figsize=(18,4))
    sns.barplot(x="Name_title",y="Survived",data=avg_survived)

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
do_visualize(train)
#print train['Fare
print train.groupby(['Name_title'])
avg_survived = train[['Name_title','Survived']].groupby(['Name_title'], as_index=False).mean()
print avg_survived

"""
Criterion
n_estimators
min_samples_split
min_samples_leaf
max_features
oob-score
random_stttttttate
n_jobs
"""
rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=400,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
#print train.iloc[:,2:]
#print train[['Name_title','Survived']].groupby(['Name_title'], as_index=False).mean()
y = train.pop('Survived')
numeric_variables = list(train.dtypes[train.dtypes != "object"].index)
##print y
rf.fit(train[numeric_variables], y)
print "%.4f" % rf.oob_score_ 
#sns



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

gs = gs.fit(train[numeric_variables], y)


# #### Inspect best parameters


print(gs.best_score_)
print(gs.best_params_)


#print train.dtypes.index
##33333333print test.dtypes.index
numeric_variables_test= list(test.dtypes[test.dtypes != "object"].index)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
#print test
predictions = rf.predict(test[numeric_variables_test])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('data', 'titanic_result.csv'), sep=",", index = False)


#


                        
