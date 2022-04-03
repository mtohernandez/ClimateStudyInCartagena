import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error

"""
This is an in-depth analysis of linear regression

Made by: Mateo Hernandez

02/04/2022
"""

# Read csv and extract relevant data
data  = pd.read_csv('climatedata.csv', sep=';', encoding='ISO-8859-1')
df = pd.DataFrame(data, columns=['T2M', 'TS', 'PRECTOTCORR', 'QV2M'])

# These three variables wok the best
x = df[['T2M','PRECTOTCORR', 'TS']].values
y = df['QV2M'].values

# Split into a training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=100)

# Define the pipeline for scaling and model fitting
pipeline = Pipeline([
    #("MinMax Scaling", MinMaxScaler()),
    ("Scaling", StandardScaler()),
    ("SGD Regression", SGDRegressor())
])

# Scale the data and fit the model
pipeline.fit(X_train, Y_train)

# Evaluate the model
Y_pred = pipeline.predict(X_test)

print(pipeline.named_steps['SGD Regression'].coef_)
print(pipeline.named_steps['SGD Regression'].intercept_)
print("\n\n")
print('Mean Absolute Error: ', mean_absolute_error(Y_pred, Y_test))
print('Score Test', pipeline.score(X_test, Y_test))
print('Score Train', pipeline.score(X_train, Y_train))
print(f"RMSE: {sqrt(mean_squared_error(Y_test, Y_pred))}")
print(f"MSE: {mean_squared_error(Y_test, Y_pred)}")
