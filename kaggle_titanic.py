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
trainData.head()
#remove unnecessary variables
trainData=trainData.drop(labels=['Name','Ticket','Cabin'],axis=1)
testData=testData.drop(labels=['Name','Ticket','Cabin'],axis=1)

#save these for later
testPassengerIds = testData['PassengerId']
trainPassengerIds = trainData['PassengerId']

sns.pairplot(trainData,hue='Survived') #a lot of people who died were low fare, 3rd class.
plt.hist(trainData['Age']) #skewed
trainData.isna().sum()/len(trainData) #Age has ~20% missing data.

#Categorical variables: passenger ID, PClass, Sex, Embarked, Survived
#Continuous variables: Age, SibSp, Parch, Fare

#describe the data and look for further evidence of skew
trainData['Age'].describe() #skew
trainData['Parch'].describe() #skew
trainData['SibSp'].describe() #skew
trainData['Fare'].describe() #v skew

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
medianTrain['Embarked'].fillna("S",inplace=True)
medianTest['Embarked'].fillna("S",inplace=True)
medianTrain["Embarked"].replace({'S': 0, 'C': 1, 'Q':2},inplace=True)
medianTest["Embarked"].replace({'S': 0, 'C': 1, 'Q':2},inplace=True)

#the other categorical variables are already in a good state.

#MODEL TIME OH BOY

y = medianTrain['Survived']
X = medianTrain.drop('Survived',axis=1)
XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, test_size = 0.15, random_state=2)

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
    ])

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

actualPredictions=model.predict(medianTest)
testPredictions = model.predict(XTest)
predictionClassTest =[1 if prob > 0.5 else 0 for prob in np.ravel(testPredictions)]
print(sklearn.metrics.confusion_matrix(yTest,predictionClassTest)) #more false negatives than false positives

predictionClass = [1 if prob > 0.5 else 0 for prob in np.ravel(actualPredictions)]
pd.DataFrame({"PassengerID": testPassengerIds, "Predictions": predictionClass})

