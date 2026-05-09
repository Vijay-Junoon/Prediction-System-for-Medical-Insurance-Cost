import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

dataset = pd.read_csv("insurance.csv")
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# Encoding categorical data - sex, smoker, region
ct = ColumnTransformer(transformers=[('encoding',OneHotEncoder(),[1,4,5])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Splitting training, testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Fitting Model
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print(f"MAE: {mean_absolute_error(y_pred,y_test)}")
print(f"MSE: {mean_squared_error(y_pred,y_test)}")
print(f"R2 Score: {r2_score(y_pred,y_test)}")

