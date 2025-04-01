# Class Activity 7
# Carson Bartholic

import pandas
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, mutual_info_score
import matplotlib.pyplot as plt

# Color Codes
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

KVAL0 = 10      # Number of features to select
KVAL1 = 100     # Number of features to select
KVAL2 = 500     # Number of features to select
KVAL3 = 1000    # Number of features to select
KVAL4 = 10000   # Number of features to select
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

random_state = 0
####################################################################
#KVAL0 = 10 Processing##############################################
####################################################################
# feature selection and binarization
xVal0 = SelectKBest(k=KVAL0).fit_transform(xVal, yVal)                                                         # selects the top k features

kmeans0 = KMeans(n_clusters=3, random_state=random_state)
kmeans0.fit(xVal0)
kmeans0_pred = kmeans0.predict(xVal0)


####################################################################
#KVAL1 = 100 Processing#############################################
####################################################################
# feature selection and binarization

xVal1 = SelectKBest(k=KVAL0).fit_transform(xVal, yVal)                        # selects the top k features

kmeans1 = KMeans(n_clusters=3, random_state=random_state)
kmeans1.fit(xVal1)
kmeans1_pred = kmeans1.predict(xVal1)


####################################################################
#KVAL2 = 500 Processing#############################################
####################################################################
# feature selection and binarization

xVal2 = SelectKBest(k=KVAL2).fit_transform(xVal, yVal)                        # selects the top k features

kmeans2 = KMeans(n_clusters=3, random_state=random_state)
kmeans2.fit(xVal2)
kmeans2_pred = kmeans2.predict(xVal2)


####################################################################
#KVAL3 = 1000 Processing############################################
####################################################################
# feature selection and binarization
xVal3 = SelectKBest(k=KVAL3).fit_transform(xVal, yVal)                        # selects the top k features

kmeans3 = KMeans(n_clusters=3, random_state=random_state)
kmeans3.fit(xVal3)
kmeans3_pred = kmeans3.predict(xVal3)


####################################################################
#KVAL4 = 10000 Processes############################################
####################################################################
# feature selection and binarization
xVal4 = SelectKBest(k=KVAL4).fit_transform(xVal, yVal)                        # selects the top k features

kmeans4 = KMeans(n_clusters=3, random_state=random_state)
kmeans4.fit(xVal4)
kmeans4_pred = kmeans4.predict(xVal4)


####################################
#######Score Calculation############
####################################
# Calculate scores for each k value
# Pass in the different test splits to get outputs
def calculate_scores(xVal, yVal, kmeans_pred):
    scores = {}
    scores['silhouette'] =  silhouette_score(xVal, kmeans_pred)
    scores['calinski'] =    calinski_harabasz_score(xVal, kmeans_pred)
    scores['davies'] =      davies_bouldin_score(xVal, kmeans_pred)
    scores['rand'] =        rand_score(yVal, kmeans_pred)
    scores['ari'] =         adjusted_rand_score(yVal, kmeans_pred)
    scores['mi'] =          mutual_info_score(yVal, kmeans_pred)
    scores['nmi'] =         normalized_mutual_info_score(yVal, kmeans_pred)
    return scores

# Store scores for each K value
scores_dict = {
    str(KVAL0): calculate_scores(xVal0, yVal, kmeans0_pred),
    str(KVAL1): calculate_scores(xVal1, yVal, kmeans1_pred),
    str(KVAL2): calculate_scores(xVal2, yVal, kmeans2_pred),
    str(KVAL3): calculate_scores(xVal3, yVal, kmeans3_pred),
    str(KVAL4): calculate_scores(xVal4, yVal, kmeans4_pred)
}

# Create a DataFrame to store the scores
scores_df = pandas.DataFrame(scores_dict).T

# Save the scores to a CSV file
scores_df.to_csv('clustering_scores.csv')

####################################
####Visualization of the scores#####
####################################
# Plot the scores on different graphs
fig, axes = plt.subplots(nrows=len(scores_df.columns), ncols=1, figsize=(10, 10))

for ax, score in zip(axes, scores_df.columns): #pair each axis with a score
    ax.plot(scores_df.index, scores_df[score], marker='o', label=score)
    ax.set_title(f'{score} Score for Different K Values')
    ax.set_xlabel('K Values')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('clustering_scores_plot.png')  # Save the plot as a PNG file
plt.show()
plt.close() 
