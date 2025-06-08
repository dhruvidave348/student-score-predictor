import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


#LOAD THE DATASET
data=pd.read_csv("data.csv")
X=data[["Hours"]]
y=data["Score"]


#SPLIT INTO TRAINING AND TESTING DATA
X_train,X_test,y_train,y_test,=train_test_split(X,y,test_size=0.2,random_state=0)


#TRAIN THE MODEL
model=LinearRegression()
model.fit(X_train,y_train)

#PREDICT THE PLOT
y_pred = model.predict(X_test)


#COMPARE ACTUAL vs PREDICTED
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)

#PLOT
plt.scatter(X,y,color='blue',label='Actual Data')
plt.plot(X,model.predict(X),color='red',label='Regression line')
plt.title("Hours vs Score")
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.legend()
plt.show()
