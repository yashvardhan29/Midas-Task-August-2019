# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def MakeDataframe():
	# Importing the dataset.
	arr = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model year", "origin", "car name"]
	dataset = pd.read_csv('auto-mpg.data', header = None, names = arr, delim_whitespace = True)

	X = dataset.iloc[:, 1:-1].values
	y = dataset.iloc[:,0].values

	for i in range(len(X)):
	    for j in range(len(X[i])):
	        if X[i][j] == '?':
	            X[i][j] = 'NaN'
	return X,y            

def ReplaceMissing(X):
	# Replacing missing values. 
	imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0) 
	imputer = imputer.fit(X[:,:])
	X[:,:] = imputer.transform(X[:,:])

	X = np.vstack(X[:,:]).astype(np.float)

	return X

def EncodeCategoricalData(X):
    # Encoding categorical data using dummy variables.
    labelencoder = LabelEncoder()
    X[:, 6] = labelencoder.fit_transform(X[:, 6])
    onehotencoder = OneHotEncoder(categorical_features = [6])
    X = onehotencoder.fit_transform(X).toarray()
    # Removing first column.
    X = X[:,1:]
    return X


X,y = MakeDataframe()
X = ReplaceMissing(X)
X = EncodeCategoricalData(X)

val = 0 

for i in range(100):
    
    #Splitting dataset into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    # Fitting Multiple Linear Regression to the Training set.
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # Predicting the Test set results.
    y_pred = regressor.predict(X_test)


    val += regressor.score(X_test,y_test)
    
print("The accuracy of the model is " + str(val/100))


