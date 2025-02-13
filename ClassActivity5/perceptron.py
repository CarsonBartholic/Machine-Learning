# Class Activity 5
# Carson Bartholic

import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
# Number of features to select
KVAL = 10000

# load CSV
dataSet = pandas.read_csv('../dataset.csv',index_col=0)

# Data Preprocessing
dataSet.rename(index = lambda x: 0 if x.startswith("brca") else 1 if x.startswith("prad") else 2 if x.startswith("luad") else x, inplace=True) # relabels the data set to 0,1,2 based on cancer type
dataSet = dataSet.rename(columns = {'Unnamed: 0': 'Targets'})                           # rename index column to label
dataSet['Targets'] = dataSet.index                                                      # set 'Targets' column as index
dataSet = dataSet.dropna()                                                              # drop rows with missing values


xVal = dataSet.drop(columns = 'Targets')                                                # drop target column
xVal = xVal.loc[:, xVal.nunique() > 1]                                                  # drop columns with only one unique value
yVal = dataSet['Targets']                                                               # sets y values to the label column

# feature selection and binarization
featureSelection = SelectKBest(k=KVAL).fit_transform(xVal, yVal)                        # selects the top features
binarizer = Binarizer(threshold=0)
featureSelection = binarizer.fit_transform(featureSelection)

x_train, x_test, y_train, y_test = train_test_split(featureSelection, yVal, test_size=0.2, random_state=35) # splits data into training and testing data sets 


# Posible parameters for the perceptron, used in a grid search
param_grid = {
    'penalty': ['l1', 'l2'],
    'alpha': [0.0001, 0.001],
    'max_iter': [100, 200, 300],
    'n_jobs' : [-1]
}

perceptronModel = GridSearchCV(Perceptron(), param_grid, cv=5)                          # performs grid search to find the best number of neighbors
perceptronModel.fit(x_train, y_train)                                                   # fits the model to the training data
predictions = perceptronModel.predict(x_test)                                           # makes predictions on the test data
print("\nBest Perceptron Parameters:", perceptronModel.best_params_)                    # prints the best parameters found by the grid search
print("Best Perceptron Score From Grid Search:", perceptronModel.best_score_ * 100, "%")# prints the best parameters found by the grid search


##Output best accuracy using best params##
bestPerceptronModel = Perceptron(penalty='l1', alpha=0.0001, max_iter=100, n_jobs=-1)   # sets the model to the best parameters found by the grid search
bestPerceptronModel.fit(x_train, y_train)                                               # fits the model to the training data
bestPredictions = bestPerceptronModel.predict(x_test)                                   # makes predictions on the test data
bestPerceptronAccuracy = accuracy_score(y_test, bestPredictions) * 100                  # calculates the accuracy of the model by comparing predected values to the known values

print("\nPerceptron Accuracy Using Best Parameters:", bestPerceptronAccuracy, "%\n")     # prints the accuracy of the model