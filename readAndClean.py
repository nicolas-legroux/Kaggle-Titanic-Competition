# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

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
        
def getTicketNumber(ticket):
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
        
def elementInList(elementID, IDlist):
    if (elementID in IDlist):
        return 1
    else:
        return 0      
        
def getData():
    
    #Open the csv files  
    data = pd.read_csv('Data/train.csv', header=0)    
    test_data = pd.read_csv('Data/test.csv', header=0)
    #Append the test data to the train data
    data = data.append(test_data)
    
    # --------------------------------
    
    #Set 'Sex'=0 for women and 1 for men
    data['Sex'] = data['Sex'].map({'female':0, 'male':1} ).astype(int)   
    
    # -------------------------------
    
    #Create dummies for Pclass   
    classes = pd.get_dummies(data['Pclass']).rename(columns=lambda x: 'Class_' + str(x))
    data = pd.concat([data, classes], axis=1)
    
    # -------------------------------
    
    #Set the embarcation port to Southampton when it is null
    data.loc[ data.Embarked.isnull(), 'Embarked'] = 'S'  
    
    #Create dummies for the embarked feature
    embarkedDummies = pd.get_dummies(data['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarkedDummies], axis=1)    
    #Transform the embarcation information into an integer
    data['Embarked'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2} )
    
    # ---------------------------------------
        
    #Retrieve family last name
    data["Surname"] = data["Name"].apply(lambda name: getSurname(name))
    
    # --------------------------------------
    
    #Retrieve Title and transform the titles into either 'Mr', 'Mrs', 'Master', or 'Miss'
    data["Title"] = data["Name"].apply(lambda name: getTitle(name))       
      
    """
    # Print the titles
    print pd.Series(data.Title).unique()
    # This shows the following : 
    #  ['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
    # 'Sir' 'Mlle' 'Col' 'Capt' 'the' 'Jonkheer' 'Dona']
    # "the" is actually "the Countess"
    """       
    data["Title"] = data["Title"].replace(["Mme", "the", "Dona", "Lady"], "Mrs")
    data["Title"] = data["Title"].replace(["Ms", "Mlle"], "Miss")
    data.loc[(data.Sex == 0) & (data.Title == "Dr"), "Title" ] = "Mrs"
    data["Title"] = data["Title"].replace(["Dr", "Don", "Jonkheer", "Col", "Sir", "Rev", "Major", "Capt"], "Mr")    
    
    #Create dummies for the Title
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
    
    #Map the title to integers    
    data['Title'] = data['Title'].map({'Mrs':0, 'Miss':1, 'Master':2, 'Mr':3})
    
    # ---------------------------------------
    
    #Fill in the age median information using Title and Class information
    median_ages = np.zeros((4,3))    
    for i in range(4):
        for j in range(3):
            median_ages[i,j] = data[(data['Title'] == i) & \
                                  (data['Pclass'] == j+1)]['Age'].dropna().median()
    
    # Create a feature to know when the age was Null
    data['AgeIsNull'] = pd.isnull(data.Age).astype(int)
    
    #Compute ages when it is missing using median information
    for i in range(4):
        for j in range(3):
            data.loc[ (data.Age.isnull()) & (data.Title == i) & (data.Pclass == j+1), 'Age' ] = median_ages[i, j]
            
    # -------------------------------------
            
    #Set missing Fares using Class information
    faresMedian = np.zeros(3)
    for j in range(3):
        faresMedian[j] = data[data.Pclass == j+1]['Fare'].dropna().median()
        
    for j in range(3):
        data.loc[ (data.Fare.isnull()) & (data.Pclass == j+1), 'Fare'] = faresMedian[j]
        
    # -------------------------------------
    
    #Create some family features
    
    #Compute family size
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    #Compute fare per person
    data['FarePerPerson'] = data['Fare'] / data['FamilySize'].map(float)
    
    #Drop surname when person has no family
    data.loc[(data.FamilySize == 0), "Surname"] = "Single"
    #Set surname = concat(family size, surname) [To avoid mixing families with same surname but different family sizes]
    data['Surname'] = data.FamilySize.map(str) + data.Surname  
    #Transform surnames into Integers
    data.Surname = pd.factorize(data.Surname)[0]
    
    # ---------------------------------------
    
    # Set Cabin = Z0000 when there is no Info
    data.loc[(data.Cabin.isnull()) | (data.Cabin == ''), "Cabin"] = "Z0000"  
    # Get the Deck (first letter of cabin number)
    data["Deck"] = data["Cabin"].apply(lambda cabin: getDeck(cabin))
    # For some reason there is one Deck 'T' in the Data, set it to 'Unknown Deck'
    data["Deck"] = data["Deck"].replace("T", "Z") 
    
    #Create dummies for the Deck feature
    decks = pd.get_dummies(data['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
    data = pd.concat([data, decks], axis=1)
    
    #Map the Deck Letter to Integers
    data['Deck'] = data['Deck'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'Z':7})
    
    # ----------------------------------------
    
    #Retrieve ticket number
    data["TicketNumber"] = data["Ticket"].apply(lambda ticket: getTicketNumber(ticket)).astype(int)
    
    #Create groups (ie People with same ticket number --> not necessarily in the same family !)
    data["GroupID"] = data["TicketNumber"]
    counts = data["TicketNumber"].value_counts()
    nroom = dict(counts)
       
    # Create feature 'how many people with same ticket number'
    data['GroupCount'] = data['TicketNumber'].map(nroom)
    data.loc[ (data.TicketNumber == 0), 'GroupCount'] = 1 #For the passengers alone in a group
    
    #Create Group IDs
    data["GroupID"] = data["TicketNumber"].apply(lambda room: getGroupID(room, nroom)).astype(int)    
    data.GroupID = pd.factorize(data.GroupID)[0]
    
    # -----------------------------------------
    
    #To know whether a passenger is in a group or family
    
    data['TravellingAlone'] = 1
    data.loc[((data.GroupCount>1) | (data.FamilySize>1)), 'TravellingAlone'] = 0
    

    # -------------------------------------------------
    
    
    #Drop columns which are not numbers    
    data = data.drop(["Cabin", "Name", "Ticket"], axis=1)
    
       
    #Divide into train and test data
    data_train = data[data.Survived.notnull()].astype(int)
    data_test = data[data.Survived.isnull()]
    
    #Get Survived
    survived = data_train["Survived"] 
    
    #Get test IDS
    IDS = data_test["PassengerId"] 
    
    data_test = data_test.drop(["PassengerId"], axis=1)
    data_train = data_train.drop(["PassengerId"], axis=1)
    
    return data_train, survived, data_test, IDS

data_train, survived, data_test, IDS = getData()



def computeSecondaryFeatures(data_test, data_train, isTest):
        # -----------------------------------------
    
    #Create 2 features to know, for each person :
    # 1. Whether, in the same Family or Group, a Female or Child Died ('should have survived but died')
    # 2. Whether, in the same Family or Group, a Man survived ('should have died but survived')
    # This allows us to identify outliers.
    
    data_test = data_test.drop('Survived', axis=1).astype(int)  
    
    if(isTest):
        data = data_train.append(data_test)
    else:
        data = data_train
    
    shouldHaveSurvivedButDiedGroupID = pd.Series(data[(((data.Sex == 0) & (data.Survived == 0)) | ((data.Title == 2) & (data.Survived == 0) & (data.Age<15))) & ((data.GroupCount>1) | (data.FamilySize>1))]["GroupID"]).unique()
    # GroupID = 0 means the person has no Group, therefore remove 0
    shouldHaveSurvivedButDiedGroupID = shouldHaveSurvivedButDiedGroupID[shouldHaveSurvivedButDiedGroupID != 0]
    shouldHaveSurvivedButDiedFamilyID = pd.Series(data[(((data.Sex == 0) & (data.Survived == 0)) | ((data.Title == 2) & (data.Survived == 0) & (data.Age<15))) & ((data.GroupCount>1) | (data.FamilySize>1))]["Surname"]).unique()
    
    shouldHaveDiedButSurvivedGroupID = (pd.Series(data[((data.Title == 3) & (data.Survived == 1) & (data.Age>18)) & ((data.GroupCount>1) | (data.FamilySize>1))]['GroupID'])).unique()
    shouldHaveDiedButSurvivedGroupID = shouldHaveDiedButSurvivedGroupID[shouldHaveDiedButSurvivedGroupID != 0]
    shouldHaveDiedButSurvivedFamilyID = (pd.Series(data[((data.Title == 3) & (data.Survived == 1) & (data.Age>18)) & ((data.GroupCount>1) | (data.FamilySize>1))]['Surname'])).unique()
        
    data["hasFamilyThatShouldHaveSurvived"] = 0
    data["hasFamilyThatShouldHaveSurvived"] = data["Surname"].apply(lambda familyID : elementInList(familyID.astype(int), shouldHaveSurvivedButDiedFamilyID))
    
    data["hasFriendThatShouldHaveSurvived"] = 0
    data["hasFriendThatShouldHaveSurvived"] = data["GroupID"].apply(lambda groupID : elementInList(groupID.astype(int), shouldHaveSurvivedButDiedGroupID))
    
    data["inGroupWithOutlierDeath"] = 0
    data.loc[(data.hasFamilyThatShouldHaveSurvived == 1) | (data.hasFriendThatShouldHaveSurvived == 1), "inGroupWithOutlierDeath"] = 1
    
    data["hasFamilyThatShouldHaveDied"] = 0
    data["hasFamilyThatShouldHaveDied"] = data["Surname"].apply(lambda familyID : elementInList(familyID.astype(int), shouldHaveDiedButSurvivedFamilyID))
    
    data["hasFriendThatShouldHaveDied"] = 0
    data["hasFriendThatShouldHaveDied"] = data["GroupID"].apply(lambda groupID : elementInList(groupID.astype(int), shouldHaveDiedButSurvivedGroupID))
    
    data["inGroupWithOutlierSurvival"] = 0
    data.loc[(data.hasFamilyThatShouldHaveDied == 1) | (data.hasFriendThatShouldHaveDied == 1), "inGroupWithOutlierSurvival"] = 1
    
    data = data.drop(["hasFamilyThatShouldHaveSurvived", "hasFriendThatShouldHaveSurvived", "hasFamilyThatShouldHaveDied", "hasFriendThatShouldHaveDied"], axis=1)
     
        
    
    
    data_train = data[data.Survived.notnull()].astype(int)
    data_test = data[data.Survived.isnull()]  
    
    data_test = data_test.drop('Survived', axis=1).astype(int)  
    data_train = data_train.drop('Survived', axis=1).astype(int)  
        
    if(isTest):
        return data_test
    else:
        return data_train
      
def keepLabels(data, onlyBinary=False):
     #Re-order the labels   
    
    #labels=['PassengerId', 'Survived', 'Title_Mr', 'Title_Miss', 'Title_Mrs', 'Title_Master', 'Age', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F',
    #        'Deck_G', 'Deck_Z', 'TicketNumber', 'Pclass', 'GroupCount', 'FamilySize', 'inGroupWithOutlierDeath', 'inGroupWithOutlierSurvival']
    
    labels = ['Title_Master', 'Age', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F',
    'Deck_G', 'Deck_Z', 'Deck', 'TicketNumber', 'Pclass', 'GroupCount', 'FamilySize', 'inGroupWithOutlierDeath', 'inGroupWithOutlierSurvival',
    'Title_Mr', 'Title_Miss', 'Title_Mrs', 'FarePerPerson', 'Class_1', 'Class_2', 'Class_3']

    labels_not_boolean = ['Pclass', 'Age', 'TicketNumber', 'GroupCount', 'FamilySize', 'FarePerPerson', 'Deck']
    
    data = data[labels]
    
    if(onlyBinary):        
        data = data.drop(labels_not_boolean, axis=1)
        labels = labels_not_boolean
        
    return data, labels
    