#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:59:08 2017

@author: Cino
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def cabin_number(train,test):
    #print train['Cabin']
    for i in [test,train]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        #train['Cabin_lettter'] =train['Cabin'].apply(lambda x: x[0])
    return train,test

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


