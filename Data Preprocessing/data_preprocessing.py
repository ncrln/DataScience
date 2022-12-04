import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing the dataset
dataset = pd.read_csv('Data.csv')

#We have two entities that are called features (independent variables) and dependent variables (what will be predicted)
#The features are the columns in the dataset and the dependent variable vector is the last one (communly)
#locate indexes of the column, the ':' means that we want to locate all the rows, after the ',' we specify that will be located all the columns except the last one ':–1'
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)


#Substitute the missing data by the average of the other values of the same column:
#Instantiating the object imputer in the class SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#Selecting the numerical columns with the missing values '1:3', and considering all the rows ':' from X (features):
imputer.fit(X[:, 1:3])

#This method will replace into the missing points the mean value:
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)


#Transform the categorized columns in numbers - Independent Variable:
#Instantiating an object:

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') #transformer for the type of encoding data, specify the column to be transformed, and remainder for passthrough the columns that we dont want to encode
X = np.array(ct.fit_transform(X))
print(X)
#The result will bring the first three columns as binary IDs for each country (OneHotEncoder), or each non-numerical category, and keep the others columns the same.


#Transform the categorized columns in numbers - Dependent Variable:
le = LabelEncoder()
y = le.fit_transform(y)

print('Dependent variables:\n', y)


#Splitting the dataset into Training set and Test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) #test_size is the percentage to split to the training set and random_state is to always get the same result from the splitting
print('X_traing\n', X_train) #it will return 8 observations of the dataset, which corresponds to the values of the columns and rows splitted through the dataset.
print('X_test\n', X_test) #it will return 2 observations from the same dataset taked randomly
print('X_traing\n', y_train) #it will return 8 purchased decisions, correspondent to the same observations that returned from X_train
print('X_traing\n', y_test) #it will return 2 purchased decisions for the X_test observations


#Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])

print('X train scalled',X_train)
print('X test scalled',X_test)