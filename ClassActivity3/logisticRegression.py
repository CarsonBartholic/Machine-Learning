# Class Activity 3
# Carson Bartholic

import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load CSV
dataSet = pandas.read_csv('../dataset.csv',index_col=0)

# Data Preprocessing
dataSet.rename(index = lambda x: 0 if x.startswith("brca") else 1 if x.startswith("prad") else 2 if x.startswith("luad") else x, inplace=True) # relabels the data set to 0,1,2 based on cancer type
dataSet = dataSet.rename(columns = {'Unnamed: 0': 'Targets'})                       # rename index column to label
dataSet['Targets'] = dataSet.index                                                  # set 'Targets' column as index
dataSet = dataSet.dropna()                                                          # drop rows with missing values

scaler = MinMaxScaler()                                                             # creates a scaler object for min max scaling

xVal = dataSet.drop(columns = 'Targets')                                            # drop target column
xVal = pandas.DataFrame(scaler.fit_transform(xVal))                                 # performs min max scaling on the data set not including the label column
xVal = xVal.loc[:, xVal.nunique() > 1]                                              # drop columns with only one unique value

yVal = dataSet['Targets']                                                           # sets y values to the label column

# Split Data ---> x_train and x_test are the features, y_train and y_test are the labels
x_train, x_test, y_train, y_test = train_test_split(xVal, yVal, test_size=0.2, random_state=42) # splits data into training and testing data sets 

# Create a logistic regression model
model = LogisticRegression()                                                        # creates a logistic regression model
model.fit(x_train, y_train)                                                         # fits the model to the training data

# Predict
predictions = model.predict(x_test)                                                 # makes predictions on the test data
print("Predictions: \n", predictions)                                               # prints the predictions

Accuracy = accuracy_score(y_test, predictions) * 100                                # calculates the accuracy of the model by comparing predected values to the known values
print("Accuracy: ", Accuracy, "%")                                                  # prints the accuracy of the model