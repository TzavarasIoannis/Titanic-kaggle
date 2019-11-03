
'''
 My first approach on Titanic set 😊. 
 short, my moves:
 1)DROP values(PassengerId,Name,Cabin , Ticket)  
 2)map the values(Sex-Embarked) 
 3)Standardization  
 4)Create and fit SVM(kernel= 'rbf')
'''

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score

#names of attributes
names= ["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
#import train dtset
dataset=pd.read_csv("titanicTrain.txt", names=names , skiprows = 1)
print(dataset.head(10)) #view the first 10 rows of dataset.
print(dataset.info()) #info

# search for missing values.
print(dataset.info())
#plot missing values
sns.heatmap(dataset.isnull(), cbar=False, cmap="YlGnBu_r") #null values 
plt.show() 
#1rst approach 
#cabin has many Nan values 
#drop the att/es : PassengerID, Name, Cabin , Ticket.
dataset = dataset.drop(["PassengerId","Name","Cabin" , "Ticket"  ] , axis = 1 )

#analyze survival rate , Sex-Survival rate bar plot
sns.barplot(x='Sex' , y = 'Survived', data = dataset )
plt.show()

#Embarked - Survived relationship 
sns.barplot(x='Embarked' , y = 'Survived', data = dataset )
plt.show()

#bar plot for Embarked values
dataset.Embarked.value_counts().plot(kind="bar")
plt.show()

#map the sex values. 
d = {"male": 2  , "female": 1}
dataset['Sex'] = dataset['Sex'].map(d)
#view the sex values after mapping.
dataset['Sex'].head(5)

#corr 
#corr map
sns.heatmap(dataset.corr(), square=True)
plt.show()

# correlation with numbers. 
corr = dataset.corr()

print( corr["Age"])
print( corr["Fare"])
print( corr['Sex'])

#map the values.
x = {  'S' : 3 , 'Q' : 2 , 'C': 1}
dataset['Embarked'] = dataset['Embarked'].map(x)
#view the embarked values after mapping.
dataset['Embarked'].head(5)

#filll missing 
#print(dataset['Age'].isnull().sum()) #result: 177
#mean of AGE
Agemean = dataset['Age'].mean()
dataset['Age']=dataset['Age'].fillna(Agemean)

#mean of Embarked.
EmbarkedMean = dataset['Embarked'].mean()
dataset['Embarked']=dataset['Embarked'].fillna(EmbarkedMean) # fill missing values

print(dataset.info()) #get a view of dataset 

#convert values of dataset to float
dataset = dataset.astype('float')

#create X_train, Y_train sets 
X_train = dataset.drop(['Survived'], axis = 1)
Y_train = dataset['Survived']

#prin 5 first rows
print(X_train.head(5))
print(Y_train.head(5))

#insert test dataset
names2 = ["PassengerId1","Pclass1","Name1","Sex1","Age1","SibSp1","Parch1","Ticket1","Fare1","Cabin1","Embarked1"]
dataset_test = pd.read_csv( "testTitanic.txt", names= names2 , skiprows = 1 ) 

#Y test- true values 
Y_test = pd.read_csv( "Y_TEST.txt" )

#map the values
dataset_test["Embarked1"]=dataset_test["Embarked1"].map(x)

#print(dataset_test["Embarked1"].head())
#MAP Sex values
print(dataset_test['Sex1'].head(5))
dataset_test['Sex1'] = dataset_test['Sex1'].map(d)
print( dataset_test['Sex1'].head(5))
#print('embarked test set', dataset_test['Embarked1'].head(5))

'''
before drop PassengerID, we create an array with ID's for knowing who passenger live/die 
'''
#create PassengerID table 
PassengerID =np.array(dataset_test['PassengerId1'])

#drop values from Test set.
dataset_test = dataset_test.drop(["PassengerId1","Name1","Cabin1" , "Ticket1"  ], axis=1)

dataset_test['Age1']= dataset_test['Age1'].fillna(Agemean) #fill missing values from Age- test set

Faremean = dataset['Fare'].mean() # mean of Fare 

dataset_test['Fare1']= dataset_test['Fare1'].fillna(Faremean) # fill Fare (test set ) with mean

#!!!!
dataset_test = dataset_test.astype('float')
print(dataset_test.info()) #info of test set 

#import svm
from sklearn.svm import SVC
#import metrics
import sklearn.metrics as m

svclassifier = SVC(kernel='rbf').fit(X_train,Y_train) # train Gaussian SVM on setrain set
scores = cross_val_score(svclassifier, X_train, Y_train, cv=5 ) #Evaluating of estimator performance with k-cross validation

print('Accuracy of SVM with k-cross validation (k = 5 ) -without standardization- :\n',scores.mean()) # print mean of validation scores = model accuracy 
predict = svclassifier.predict(dataset_test)  #predict values

acc = m.accuracy_score(Y_test,predict,normalize = True) # accuracy of estimator 

#standardization of X_train/(train set), dataset_set/(test_set)
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

xs = ss.fit_transform(X_train)   #transform train set - > xs
xss = ss.fit_transform(dataset_test)  #transform test set -> xss

svclassifier = SVC(kernel='rbf').fit(xs,Y_train) # train Gaussian SVM on train set 
predict = svclassifier.predict(xss) #model prediction

scores = cross_val_score(svclassifier, xs, Y_train, cv=5 ) #Evaluating of estimator performance with k-cross validation
res = scores.mean() # mean of validation scores (evaluate the model)

acc = m.accuracy_score(Y_test,predict,normalize = True) # accuracy score of predictions
print("Accuracy of SVM with k-cross validation (k = 5 ) - after standardization-: \n"  , + res)
print("Accuracy score of prediction \n", + acc)

#create a Dataframe with Passenders Id and prediction (survived or not)
#visualize the first 5 rows

#create submission
submission = pd.DataFrame()
submission['PassengerID'] = PassengerID
submission['Survived'] = predict.reshape((predict.shape[0]))
#view first 5 rows of submission dataframe
print(submission.head())

#convert dataframe to csv file
submission.to_csv('TitanicPredictionS2 ',index=False)
#plot id - survived
plt.plot(submission['PassengerID'],submission['Survived'])
plt.show()




