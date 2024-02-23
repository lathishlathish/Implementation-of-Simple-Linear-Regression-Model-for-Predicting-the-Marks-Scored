# EXP :2 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

# Program:
python
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LATHISH KANNA.M
RegisterNumber: 212222230073

import pandas as pd
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
x=(df.iloc[:,:-1]).values
x
y=(df.iloc[:,1]).values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

# Output:

## df.head() and df.tail()
![Screenshot 2024-02-20 102308](https://github.com/Hariveeraprasad-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145049988/b8510e62-cdb5-4bcb-934f-97e0be965990)
## Array values of X
![Screenshot 2024-02-20 102507](https://github.com/Hariveeraprasad-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145049988/27467469-ee6d-451b-b62c-1f5dfc544a5c)
## Array values of Y
![Screenshot 2024-02-20 102549](https://github.com/Hariveeraprasad-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145049988/80d0ebe8-14c1-41bd-aa1d-05b32670d003)
## Values of Y prediction
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145049988/ba471633-9e4e-4f6b-8a35-3c917453006f)
## Values of Y test
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145049988/c9ef0661-3d82-4b6a-9cdc-a3e41f69fca6)
## Training set graph
![Screenshot 2024-02-20 102647](https://github.com/Hariveeraprasad-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145049988/f09332bc-b092-4ea9-80ae-de7494dd1076)
## Testing set graph
![Screenshot 2024-02-20 103319](https://github.com/Hariveeraprasad-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145049988/21fbef8c-faa4-41b6-9947-2d9f1d3fcab4)
## Value of MSE,MAE & RMSE

![image](https://github.com/lathishlathish/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120359170/993aa8d6-1448-4b58-b4fe-dfd85fc52009)

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
