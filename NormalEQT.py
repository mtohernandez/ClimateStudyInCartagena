
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# Read the CSV file
data  = pd.read_csv('climatedata.csv', sep=';', encoding='ISO-8859-1')
df = pd.DataFrame(data, columns=['T2M', 'TS', 'PRECTOTCORR', 'QV2M'])

X = df['T2M'].values
y = df['QV2M'].values
theta = [0,0]

def find_theta(X, y):
    
    m = X.shape[0] # Number of training examples. 
    # Appending a cloumn of ones in X to add the bias term.
    X = np.append(X, np.ones((m,1)), axis=1)    
    # reshaping y to (m,1)
    y = y.reshape(m,1)
    
    # The Normal Equation
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    
    return theta


def predict(X):
    
    # Appending a cloumn of ones in X to add the bias term.
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)
    
    # preds is y_hat which is the dot product of X and theta.
    preds = np.dot(X, theta)
    
    return preds


theta = find_theta(X[:, np.newaxis], y)
pred = predict(X[:, np.newaxis])

print(f"Theta: {theta}")
print(f"Mean Squared error: {mean_squared_error(y, pred)}")
print(f"R2: {r2_score(y, pred)}")