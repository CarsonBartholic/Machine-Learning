import pandas as pd
import numpy as np
import umap
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, mutual_info_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import AgglomerativeClustering


# load the cancer data from sklearn using load_cancer_data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

target = pd.Series(data.target)

data = pd.DataFrame(data.data, columns=data.feature_names)
data.index = data.index.astype(str)

y = target
X = data

selector = SelectKBest(score_func=f_classif, k=15)  # Initialize SelectKBest with f_classif and k=10
X = selector.fit_transform(X, y)  # Fit and transform the data

# Clustering
clustering = AgglomerativeClustering()
clustering.fit(X)
labels = clustering.fit_predict(X)

# generate a umap from the cluster
xumap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(X)
# plot the umap
plt.figure(figsize=(10, 8))
sns.scatterplot(x=xumap[:, 0], y=xumap[:, 1], hue=clustering.labels_, palette='viridis', alpha=0.7)
plt.title('UMAP of Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
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


calculated_scores = calculate_scores(X, y, clustering.labels_)


####################################
####Visualization of the scores#####
####################################
categories = ['silhouette', 'calinski', 'davies', 'rand', 'ari', 'mi', 'nmi']
values = [calculated_scores[categories[0]], calculated_scores[categories[1]], calculated_scores[categories[2]], calculated_scores[categories[3]], calculated_scores[categories[4]], calculated_scores[categories[5]], calculated_scores[categories[6]]]

scores_df = pd.DataFrame(list(calculated_scores.items()), columns=['Score Function', 'Values']) # Create data frame
scores_df.to_csv('clustering_scores.csv', index=False)  # output to csv

# Creating a vertical bar graph
plt.bar(categories, values, color='pink')
plt.ylim(0, 1) #limit y axis to 0-1
plt.xlabel('Score Function')
plt.ylabel('Values')
plt.title('Clusterig Scores')
plt.savefig('clustering_scores_bar.png')  # Save the plot as a PNG file
plt.show()