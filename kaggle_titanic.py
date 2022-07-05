# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:39:38 2022

@author: BSeng
"""

#no clue how to do this right now so uhhh
# titnanic. 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import tensorflow as tf


testData = pd.read_csv("C:\\Users\BSeng\\OneDrive - Minitab, LLC\\Documents\\titanic\\test.csv")
trainData = pd.read_csv("C:\\Users\BSeng\\OneDrive - Minitab, LLC\\Documents\\titanic\\train.csv")

#EDA time
trainData.shape #gets us the size of the dataframe. 
trainData.head() #gives us first 5 rows.
trainData.info() #Tells us object data types and how many values. Note that Cabin and Age have definite missings, and Embarked has 2 missings.
#remove unnecessary variables. These are unique identifiers.
trainData=trainData.drop(labels=['Name','Ticket'],axis=1)
testData=testData.drop(labels=['Name','Ticket'],axis=1)
uniqueCabin = pd.Series(trainData['Cabin']).drop_duplicates().tolist() #look to see unique identifiers in the cabin. Most have A - G cabin names and will require further cleanup to use.
#Electing to remove due to the high volume of missing data from .info().
trainData=trainData.drop(labels=['Cabin'],axis=1)
testData=testData.drop(labels=['Cabin'],axis=1)


#save these for later
testPassengerIds = testData['PassengerId']
trainPassengerIds = trainData['PassengerId']

sns.pairplot(trainData,hue='Survived') #a lot of people who died were low fare, 3rd class.
plt.hist(trainData['Age']) #skewed
trainData.isna().sum()/len(trainData) #Age has ~20% missing data.

sns.countplot(x=trainData['SibSp']) 
sns.countplot(x=trainData['Parch'])

#Categorical variables: passenger ID, PClass, Sex, Embarked, Survived, SibSp, Parch
#Continuous variables: Age, Fare

#describe the data and look for further evidence of skew
trainData['Age'].describe() #skew
trainData['Fare'].describe() #v skew - replace with median.

#Categorical data
trainData['Pclass'].value_counts() #most were 3rd class
trainData['Embarked'].value_counts() #most departed from southhampton
trainData['Sex'].value_counts() #Most were men
trainData['SibSp'].value_counts()
trainData['Parch'].value_counts() #skew

#Data cleaning section. Will want to clean the Cabin data. Will also want to redo continuous analysis for SibSp and Parch because those look more like categorical variables.
#fill in blanks. 
medianTrain = trainData
medianTest = testData
medianTrain['Age'].fillna(trainData['Age'].mean(skipna=True),inplace=True)
medianTrain['Parch'].fillna(trainData['Parch'].mean(skipna=True),inplace=True)
medianTrain['SibSp'].fillna(trainData['SibSp'].mean(skipna=True),inplace=True)
medianTrain['Fare'].fillna(trainData['Fare'].median(skipna=True),inplace=True)

medianTest['Age'].fillna(testData['Age'].mean(skipna=True),inplace=True)
medianTest['Parch'].fillna(testData['Parch'].mean(skipna=True),inplace=True)
medianTest['SibSp'].fillna(testData['SibSp'].mean(skipna=True),inplace=True)
medianTest['Fare'].fillna(testData['Fare'].median(skipna=True),inplace=True)

#encode categorical variables
medianTrain["Sex"].replace({'male': 0, 'female': 1},inplace=True)
medianTest["Sex"].replace({'male': 0, 'female': 1},inplace=True)

#encode embarked
medianTrain['Embarked'].fillna("S",inplace=True) #Since Southhampton is most common
medianTest['Embarked'].fillna("S",inplace=True)
medianTrain["Embarked"].replace({'S': 0, 'C': 1, 'Q':2},inplace=True)
medianTest["Embarked"].replace({'S': 0, 'C': 1, 'Q':2},inplace=True)

#encode SibSp
medianTrain['SibSp'].fillna(trainData['SibSp'].mode(),inplace=True)
medianTest['SibSp'].fillna(testData['SibSp'].mode(),inplace=True)

#encode Parch
medianTrain['Parch'].fillna(trainData['Parch'].mode(),inplace=True)
medianTest['Parch'].fillna(testData['Parch'].mode(),inplace=True)

#MODEL TIME OH BOY

y = medianTrain['Survived']
X = medianTrain.drop('Survived',axis=1)
XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=2)

decisionTreeModel=sklearn.tree.DecisionTreeClassifier(criterion='entropy')
decisionTreeModel.fit(XTrain,yTrain)
yPredict = decisionTreeModel.predict(XTest)
decisionTreeModelAccuracy = sklearn.metrics.accuracy_score(yTest,yPredict)

#Let's try an EXTRA TREE
extraTreeModel=sklearn.tree.ExtraTreeClassifier(criterion='gini')
extraTreeModel.fit(XTrain,yTrain)
yPredict = extraTreeModel.predict(XTest)
extraTreeModelAccuracy = sklearn.metrics.accuracy_score(yTest,yPredict)

#ok that didn't work RANDOM FOREST TIME
RFModel = RandomForestClassifier(n_estimators = 2000, criterion='entropy',max_depth=20)
RFModel.fit(XTrain,yTrain)
yPredict=RFModel.predict(XTest)
RFModelAccuracy = sklearn.metrics.accuracy_score(yTest,yPredict)

#tensorflow. Credit to https://towardsdatascience.com/how-to-train-a-classification-model-with-tensorflow-in-10-minutes-fd2b7cfba86 for tutorial
tf.random.set_seed(1198)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[
                  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall')
                  ],
              )
#get some metrics and see
numEpoch = 300
history=model.fit(XTrain, yTrain, epochs=numEpoch)
plt.plot(np.arange(1,numEpoch+1),history.history['loss'],label="Loss")
plt.plot(np.arange(1,numEpoch+1),history.history['accuracy'],label="Accuracy")
plt.plot(np.arange(1,numEpoch+1),history.history['precision'],label="Precision")
plt.plot(np.arange(1,numEpoch+1),history.history['recall'],label="Recall")
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend();

#Let's actually optimize our model.

actualPredictions=model.predict(medianTest)
testPredictions = model.predict(XTest)
predictionClassTest =[1 if prob > 0.5 else 0 for prob in np.ravel(testPredictions)]
print(sklearn.metrics.confusion_matrix(yTest,predictionClassTest)) #more false negatives than false positives

predictionClass = [1 if prob > 0.5 else 0 for prob in np.ravel(actualPredictions)]
pd.DataFrame({"PassengerID": testPassengerIds, "Predictions": predictionClass})

