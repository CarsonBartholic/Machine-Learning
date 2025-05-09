import pandas as pd
import numpy as np
import umap
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, mutual_info_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import AgglomerativeClustering, DBSCAN

# load the cancer data from sklearn using load_cancer_data
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
target = pd.Series(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)
data.index = data.index.astype(str)


y = target
X = data

# generate a umap from the cluster for dimensionality reduction
xumap = umap.UMAP(min_dist=0.5, n_components=2, random_state = 42).fit_transform(X, y)
plt.scatter(x=xumap[:, 0], y=xumap[:, 1], c=y, cmap='Spectral', s=5)
plt.savefig('DBSCAN_umap.png')  # Save the plot as a PNG file
plt.show()


# caculate the required score functions and return a dictionary
def calculate_scores(xVal, yVal, predicted):
    scores = {}
    scores['silhouette'] =  silhouette_score(xVal, predicted)
    scores['calinski'] =    calinski_harabasz_score(xVal, predicted)
    scores['davies'] =      davies_bouldin_score(xVal, predicted)
    scores['rand'] =        rand_score(yVal, predicted)
    scores['ari'] =         adjusted_rand_score(yVal, predicted)
    scores['mi'] =          mutual_info_score(yVal, predicted)
    scores['nmi'] =         normalized_mutual_info_score(yVal, predicted)
    return scores


# DBSCAN Clustering
clustering = DBSCAN(eps=1, min_samples=5)
dbscan_labels = clustering.fit_predict(xumap)
calculated_scores = calculate_scores(xumap, y, dbscan_labels)  # Calculate the scores before clustering

####################################
####Visualization of the scores#####
####################################
categories = ['silhouette', 'calinski', 'davies', 'rand', 'ari', 'mi', 'nmi']
values = [calculated_scores[categories[0]], calculated_scores[categories[1]], calculated_scores[categories[2]], calculated_scores[categories[3]], calculated_scores[categories[4]], calculated_scores[categories[5]], calculated_scores[categories[6]]]

scores_df = pd.DataFrame(list(calculated_scores.items()), columns=['Score Function', 'Values']) # Create data frame
scores_df.to_csv('clustering_scores.csv', index=False)  # output to csv

# Creating a vertical bar graph
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink']
plt.bar(categories, values, color=colors)
plt.ylim(0, 1)  # Limit y-axis to 0-1
plt.xlabel('Score Function')
plt.ylabel('Values')
plt.title('Clusterig Scores')
plt.savefig('DBSCAN_clustering_scores_bar.png')  # Save the plot as a PNG file
plt.show()
