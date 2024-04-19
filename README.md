# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ABIJITH SHAAN 
RegisterNumber:  212223080002
*/
import pandas as pd
data = pd.read_csv('Employee.csv')
data.head()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
DATA HEAD:

![image](https://github.com/Shaan2803/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568486/96d79af6-a3ed-4b4e-816c-2e2a35136fa7)


NULL VALUES:

![image](https://github.com/Shaan2803/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568486/7d755b14-9bcd-4e2c-a5a6-8ae00226830f)


ASSIGNMENT OF X VALUES:



![image](https://github.com/Shaan2803/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568486/5b46d034-bde5-46bf-840c-e53caf94bff4)

ASSIGNMENT OF Y VALUES:


![image](https://github.com/Shaan2803/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568486/a8d6ec9a-0390-4775-b603-cbbf298327c4)

Converting string literals to numerical values using label encoder :


![image](https://github.com/Shaan2803/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568486/1ec95e02-bbd5-4a0c-b6b3-60f3d1435c30)

Accuracy :


![image](https://github.com/Shaan2803/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568486/e8f76ceb-01d5-4f71-a601-02fbde22720d)

Prediction :

![image](https://github.com/Shaan2803/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568486/db9e0c21-8583-410d-8a53-a616ada7ec25)]

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
