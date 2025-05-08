# EX-09:Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries and load the dataset.
2. Encode the 'Position' column using LabelEncoder to convert text to numbers.
3. Select 'Position' and 'Level' as input features (X) and 'Salary' as the target (Y).
4. Split the data into training and testing sets.
5. Train a Decision Tree Regressor model using the training data and predict on the test data.
6. Evaluate the model using Mean Absolute Error and Mean Squared Error, and predict a salary for a new input.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Ashwin Akash M
RegisterNumber: 212223230024 
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("Salary (1).csv")
df
df.head()
df.info()
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()
x=df[['Position','Level']]
y=df['Salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
de=DecisionTreeRegressor()
de.fit(x_train,y_train)
ypred=de.predict(x_test)
ypred
mas=mean_absolute_error(y_test,ypred)
print("Mean Absolute Error:",mas)
mse=mean_squared_error(y_test,ypred)
print("Mean Squared Error:",mse)
de.predict([[4,7]])
```

## Output:
![image](https://github.com/user-attachments/assets/a650126a-55ba-463e-ad49-1e0e1d49f42c)<br>
![image](https://github.com/user-attachments/assets/0f8d34de-912e-45b9-9c71-9939a6ac9e31)<br>
![image](https://github.com/user-attachments/assets/026b5afe-3428-4a78-bba5-6eb2418ffb13)<br>
![image](https://github.com/user-attachments/assets/4826167b-53ea-461a-991b-5f4a88e9c5ef)<br>
![image](https://github.com/user-attachments/assets/4d112a7d-9153-4113-bd2f-68fe91b2107b)<br>
![image](https://github.com/user-attachments/assets/a50ee29a-2eed-43f8-8a56-21bbcbbdf943)<br>
![image](https://github.com/user-attachments/assets/dd19f03a-6c64-4029-8f24-de7136628950)<br>
![image](https://github.com/user-attachments/assets/3dea8b73-9ada-4f84-a7ab-b0ac66d7381b)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
