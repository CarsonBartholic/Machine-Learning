# Class Activity 2
# Carson Bartholic

import pandas
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# load CSV
dataSet = pandas.read_csv('dataset.csv')

# Dara Preprocessing
dataSet = dataSet.drop(columns='Unnamed: 0', axis=1)    # drop index column
dataSet = dataSet.dropna()                              # drop rows with missing values
dataSet = dataSet.loc[:, dataSet.nunique() > 1]         # drop columns with only one unique value
scaler = MinMaxScaler()                                 # creates a scaler object for min max scaling
scaledDataSet = pandas.DataFrame(scaler.fit_transform(dataSet[dataSet.columns])) # performs min max scaling on the data set

# Display Data
print("Scaled Processed Data Set") # display scaled data set gens as columns and people as rows
print(scaledDataSet.head())        # shows first five patients

print("Processed Data Set")        # displays genes as columns(with labels) and people as rows
print(dataSet.head())              # shows first five patients

transposedSet = dataSet.T
print("Transposed Data Set")       # displays people as columns and genes as rows
print(transposedSet.head())        # shows first five genes

# for i in range(1410, 1435): 
#     mutationCount = dataSet.iloc[i].sum()
#     print(mutationCount)
    
# mutationCount = dataSet.iloc[1427].sum()
# print(mutationCount)

# create a list of the sums of each person's mutations
mutationCounts = dataSet.sum(axis=1)
print("Mutation Count Per Person")
print(mutationCounts)

# create a bar graph of the mutation counts
plt.bar(range(len(mutationCounts)), mutationCounts, label="Bars 1", color="black")
plt.xlabel('People')
plt.ylabel('Mutation Count')
plt.title('Mutation Count by Person')
plt.show()

# line 1429 if CSV file has the largest number of mutations with 9345 mutations