# Class Activity 4
# Carson Bartholic

import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
# Number of features to select
# Logistic NS, KNN NS, Logistic S, KNN S
# 10000 --> 79%, 34%, 74%, 37%
# 1000  --> 76%, 52%, 75%, 54%
# 100   --> 67%, 57%, 71%, 60%
# 10    --> 69%, 62%, 66%, 64%
KVAL = 100

# load CSV
dataSet = pandas.read_csv('dataset.csv',index_col=0)

# Data Preprocessing
dataSet.rename(index = lambda x: 0 if x.startswith("brca") else 1 if x.startswith("prad") else 2 if x.startswith("luad") else x, inplace=True) # relabels the data set to 0,1,2 based on cancer type
dataSet = dataSet.rename(columns = {'Unnamed: 0': 'Targets'})                       # rename index column to label
dataSet['Targets'] = dataSet.index                                                  # set 'Targets' column as index
dataSet = dataSet.dropna()                                                          # drop rows with missing values

#########################
#### WITHOUT SCALING ####
#########################
xVal = dataSet.drop(columns = 'Targets')                                            # drop target column
xVal = xVal.loc[:, xVal.nunique() > 1]                                              # drop columns with only one unique value
yVal = dataSet['Targets']                                                           # sets y values to the label column

featureSelection = SelectKBest(k=KVAL).fit_transform(xVal, yVal)                    # selects the top features

x_train, x_test, y_train, y_test = train_test_split(featureSelection, yVal, test_size=0.2, random_state=42) # splits data into training and testing data sets 

# Posible parameters for the KNN Classifier, used in a grid search
param_grid = {
    'n_neighbors' : [i for i in range(1, (int(len(x_train) ** 0.5)) + 1, 2)],
    'weights' : ['uniform', 'distance'],
    'metric' :  ['manhattan', 'euclidean', 'minkowski'],
    'n_jobs' : [-1]
}

KNNModel = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5) # performs grid search to find the best number of neighbors

KNNModel.fit(x_train, y_train)                                    # fits the model to the training data
KNNPrediction = KNNModel.predict(x_test)                          # makes predictions on the test data

Logisticmodel = LogisticRegression()                                                  # creates a logistic regression model
Logisticmodel.fit(x_train, y_train)                                                   # fits the model to the training data
predictions = Logisticmodel.predict(x_test)                                           # makes predictions on the test data
logisticAccuracy = accuracy_score(y_test, predictions) * 100                          # calculates the accuracy of the model by comparing predected values to the known values

print("\n######################VALUES WITHOUT SCALING######################")
print("Logistic Accuracy: ", logisticAccuracy, "%")                                   # prints the accuracy of the model
print("Best KNN Score:", (KNNModel.best_score_)*100, "%")                             # prints the best score found by the grid search
print("Best KNN Parameters:", KNNModel.best_params_, "\n")                            # prints the best parameters found by the grid search


######################
#### WITH SCALING ####
######################
scaler = MinMaxScaler()                                                             # creates a scaler object for min max scaling
xVal_SCALED = pandas.DataFrame(scaler.fit_transform(xVal))                          # performs min max scaling on the data set not including the label column
xVal_SCALED = xVal_SCALED.loc[:, xVal_SCALED.nunique() > 1]                         # drop columns with only one unique value

featureSelection_SCALED = SelectKBest(k=KVAL).fit_transform(xVal_SCALED, yVal)      # selects the top features

x_train_SCALED, x_test_SCALED, y_train_SCALED, y_test_SCALED = train_test_split(featureSelection_SCALED, yVal, test_size=0.2, random_state=42) # splits data into training and testing data sets 

# Posible parameters for the KNN Classifier, used in a grid search
param_grid = {
    'n_neighbors' : [i for i in range(1, (int(len(x_train) ** 0.5)) + 1, 2)],
    'weights' : ['uniform', 'distance'],
    'metric' :  ['manhattan', 'euclidean', 'minkowski'],
    'n_jobs' : [-1]
}

KNNModel = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5) # performs grid search to find the best number of neighbors

KNNModel.fit(x_train_SCALED, y_train_SCALED)                      # fits the model to the training data
KNNPrediction = KNNModel.predict(x_test_SCALED)                   # makes predictions on the test data

Logisticmodel = LogisticRegression()                                                  # creates a logistic regression model
Logisticmodel.fit(x_train_SCALED, y_train_SCALED)                                     # fits the model to the training data
predictions = Logisticmodel.predict(x_test_SCALED)                                    # makes predictions on the test data
logisticAccuracy = accuracy_score(y_test_SCALED, predictions) * 100                   # calculates the accuracy of the model by comparing predected values to the known values

print("\n######################VALUES WITH SCALING######################")
print("Logistic Accuracy: ", logisticAccuracy, "%")                                   # prints the accuracy of the model
print("Best KNN Accuracy:", (KNNModel.best_score_)*100, "%")                          # prints the best score found by the grid search
print("Best KNN Parameters:", KNNModel.best_params_, "\n")                            # prints the best parameters found by the grid search