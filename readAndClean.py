# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import re
import sys

def getGroupCount(room, dict):    
    if dict[room] > 1:
        return dict[room]
    else:    
        return 0

def getGroupID(room, dict):    
    if dict[room] > 1:
        return room
    else:    
        return 0
        
def getSurname(name):
    matchObj = re.match(r'(.*),', name)
    if matchObj:
        return matchObj.group(1)
    else:    
        return ""
        
def getRoom(ticket):
    matchObj = re.match(r'([0-9]+)', ticket)
    if matchObj:
        return matchObj.group(1)
    else:    
        return 0
    
def getTitle(name):    
    matchObj = re.match(r'(.+), ([a-z]+).', name,re.IGNORECASE)        
    if matchObj:    
        return matchObj.group(2)    
    else:    
        return ""

def getDeck(ticketNumber):    
    matchObj = re.match(r'^([A-Z])', ticketNumber,re.IGNORECASE)     
    if matchObj:    
        return matchObj.group(1)    
    else:    
        return ""
        
def getData():
    #Open the csv train file
    
    data = pd.read_csv('Data/train.csv', header=0)
    
    test_data = pd.read_csv('Data/test.csv', header=0)
    data = data.append(test_data)
    
    #Set 'Gender'=0 for women and 1 for men
    data['Gender'] = data['Sex'].map({'female':0, 'male':1} ).astype(int)
    
    """
    #Compute information about ports of embarcation
    embarked = np.zeros((2, 3, 3))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                if k==0:                
                    embarked[i, j, k] = len(data[(data['Gender'] == i) & (data['Embarked'] == 'C') & (data['Pclass'] == j+1)])
                elif k==1:
                    embarked[i, j, k] = len(data[(data['Gender'] == i) & (data['Embarked'] == 'Q') & (data['Pclass'] == j+1)])
                else:
                    embarked[i, j, k] = len(data[(data['Gender'] == i) & (data['Embarked'] == 'S') & (data['Pclass'] == j+1)]) 
                    
    #This shows that most passengers embarked in Southampton
    #print embarked
    """
    
    #Set the embarcation port to Southampton when it is null
    data.loc[ data.Embarked.isnull(), 'Embarked'] = 'S'  
    
    #Transform the embarcation information into an integer
    data['StartingPort'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2} )  
    
    #Fill in the age median information
    median_ages = np.zeros((2,3))
    
    for i in range(2):
        for j in range(3):
            median_ages[i,j] = data[(data['Gender'] == i) & \
                                  (data['Pclass'] == j+1)]['Age'].dropna().median()
    
    data['AgeIsNull'] = pd.isnull(data.Age)
    
    #Compute ages when it is missing
    for i in range(2):
        for j in range(3):
            data.loc[ (data.Age.isnull()) & (data.Gender == i) & (data.Pclass == j+1), 'Age' ] = median_ages[i, j]
    
    data['Age']  = data['Age'] .astype(int)
    
    #Set Fares
    faresMedian = np.zeros(3)
    for j in range(3):
        faresMedian[j] = data[data.Pclass == j+1]['Fare'].dropna().median()
        
    for j in range(3):
        data.loc[ (data.Fare.isnull()) & (data.Pclass == j+1), 'Fare'] = faresMedian[j]
    
    #Compute a few features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    data["Surname"] = data["Name"].apply(lambda name: getSurname(name))
    data["Title"] = data["Name"].apply(lambda name: getTitle(name))
   
    
    """
    print pd.Series(data.Title).unique()
    This shows the following : 
    ['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
     'Sir' 'Mlle' 'Col' 'Capt' 'the' 'Jonkheer' 'Dona']
     "the" is actually "the Countess"
    """
       
    data["Title"] = data["Title"].replace(["Mme", "the", "Dona", "Lady"], "Mrs")
    data["Title"] = data["Title"].replace(["Ms", "Mlle"], "Miss")
    data.loc[(data.Gender == 0) & (data.Title == "Dr"), "Title" ] = "Mrs"
    data["Title"] = data["Title"].replace(["Dr", "Don", "Jonkheer", "Col", "Sir", "Rev", "Major", "Capt"], "Mr")
    
    titles = pd.get_dummies(data['Title']).rename(columns=lambda x: 'Title_' + str(x))
    data = pd.concat([data, titles], axis=1)
    
    """Stats on title
    statsTitle = data[(data.Title == "Miss") & (data.Survived.notnull())]
    print statsTitle.describe()  
    #16% of Mr survived
    #57% of Master survived
    #70% of Miss Survived
    #79% of Mrs survived
    """
    
    data['Title'] = data['Title'].map({'Mrs':0, 'Miss':1, 'Master':2, 'Mr':3})

    data.loc[(data.FamilySize == 0), "Surname"] = "Single"
    #This is used to compare different families with the same name but different family sizes
    data['Surname'] = data.FamilySize.map(str) + data.Surname    
    data.Surname = pd.factorize(data.Surname)[0]
    
    data.loc[(data.Cabin.isnull()) | (data.Cabin == ''), "Cabin"] = "Z0000"     
    data["Deck"] = data["Cabin"].apply(lambda ticket: getDeck(ticket))
    data["Deck"] = data["Deck"].replace("T", "Z")    
    decks = pd.get_dummies(data['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
    
    data['Deck'] = data['Deck'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'Z':7})
    data = pd.concat([data, decks], axis=1)
    
    data["Room"] = data["Ticket"].apply(lambda ticket: getRoom(ticket)).astype(int)
    data["GroupID"] = data["Room"]
    counts = data["Room"].value_counts()
    nroom = dict(counts)
    data['GroupCount'] = data['Room'].map(nroom)
    data.loc[ (data.Room == 0), 'GroupCount'] = 1 #For the passengers without rooms attributed
    
    #data["GroupCount"] = data["Room"].apply(lambda room: getGroupCount(room, nroom)).astype(int)
    data["GroupID"] = data["Room"].apply(lambda room: getGroupID(room, nroom)).astype(int)
    
    data.GroupID = pd.factorize(data.GroupID)[0]
    
    data = data.drop(['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket'], axis=1)
    
    labels=['PassengerId', 'Survived', 'Gender', 'Title', 'Title_Mrs', 'Title_Miss', 'Title_Master', 'Title_Mr', 'Pclass', 'Fare', 
    'Age', 'AgeIsNull', 'StartingPort',  'Parch', 
    'SibSp', 'Deck_A', 'Deck_B', 
     'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_Z', 'Deck', 'Room', 'GroupCount', 'GroupID', 'FamilySize', 'Surname']
    
    data = data[labels]  
   
    data_train = data[data.Survived.notnull()].astype(int)
    data_test = data[data.Survived.isnull()]
    data_test = data_test.drop('Survived', axis=1).astype(int)
    
    #print data.info()
        
    return data_train.values, data_test.values, labels, data

d, b, c, data = getData()

    
    
   