# Class Activity 6
# Carson Bartholic

import pandas
from sklearn import svm
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, classification_report

# Color Codes
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

KVAL = 300      # Number of features to select
sampleRS = 420  # Random state for sampling
testSize = 0.2  # Percentage of data to use for testing

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

# Choose the same number of samples from each class
class0 = sum(yVal == 0)                                                                 #find numbr of breast cancer samples
class1 = sum(yVal == 1)                                                                 # find numbr of prostate cancer samples
class2 = sum(yVal == 2)                                                                 # find numbr of lung cancer samples
sampleNumber = min(class0, class1, class2)                                              # finds the minimum number of samples from each class


xVal = pandas.concat([xVal[yVal == 0].sample(sampleNumber, random_state = sampleRS), xVal[yVal == 1].sample(sampleNumber, random_state = sampleRS), xVal[yVal == 2].sample(sampleNumber, random_state = sampleRS)]) # samples the data set to have an even number of samples from each class
yVal = pandas.concat([yVal[yVal == 0].sample(sampleNumber, random_state = sampleRS), yVal[yVal == 1].sample(sampleNumber, random_state = sampleRS), yVal[yVal == 2].sample(sampleNumber, random_state = sampleRS)]) # samples the data set to have an even number of samples from each class

# feature selection and binarization
featureSelection = SelectKBest(k=KVAL).fit_transform(xVal, yVal)                        # selects the top features
binarizer = Binarizer(threshold=0)
featureSelection = binarizer.fit_transform(featureSelection)


x_train, x_test, y_train, y_test = train_test_split(featureSelection, yVal, test_size = testSize, random_state = 420) # splits data into training and testing data sets 


# Support Vector Classifier
svc = svm.SVC(kernel = 'rbf', C = 1, probability = True, gamma = 'scale')               # select parameters for SVC
svc.fit(x_train, y_train)                                                               # fit SVC to training data
y_pred = svc.predict(x_test)                                                            # predict on test data

# Output
print("Support Vector Classifier using ", sampleNumber, " samples from each class and ", KVAL, " features") 
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print(classification_report(y_test, y_pred))


# Test Results
print("Test Results from ", len(y_pred), " samples")
actualValues = y_test.to_numpy()                                                        # convert y_test to array of only values and no labels
for i in range(len(y_pred)):                                                            # loop through all the predicted values and print comparison
    if y_pred[i] != actualValues[i]:
        print(RED, f"{i:>3}. Predicted:", y_pred[i],  "Actual:",  actualValues[i], "INCORRECT", RESET)
    else:
        print(GREEN, f"{i:>3}. Predicted:", y_pred[i],  "Actual:",  actualValues[i], "Correct", RESET)
        