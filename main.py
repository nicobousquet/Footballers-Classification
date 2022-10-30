# file main.py
# Nicolas Bousquet

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from functions import *

# Load the data from the csv file into a Pandas Dataframe
X = pd.read_csv('processed_data.csv', index_col='name')

# Standardize the data
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means algorithm ####################################################################################################

# Elbow method to determine number of clusters
# Run a number of tests, for 1, 2, ... num_clusters
num_clusters = 50
kmeans_tests = [KMeans(n_clusters=i, init='random', n_init=10) for i in range(1, num_clusters)]
score = [kmeans_tests[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans_tests))]

# Plot the curve
plt.plot(range(1, num_clusters), score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# Create a k-means clustering model
num_clusters = 9
# int(input("Choose the number of clusters: "))
kmeans = KMeans(init='random', n_clusters=num_clusters, n_init=10)

# Fit the data to the model
kmeans.fit(X_scaled)

# Create a PCA model to reduce our data to 3 dimensions for visualisation
pca = PCA(n_components=3)
pca.fit(X_scaled)
#
# Transform the scaled data to the new PCA space
X_reduced = pca.transform(X_scaled)

# Convert to a data frame
X_reduceddf = pd.DataFrame(X_reduced, index=X.index, columns=['PC1', 'PC2', 'PC3'])
# Add cluster column to which cluster is each player in
clusters = kmeans.predict(X_scaled)
X_reduceddf['cluster'] = clusters
# Sort players by cluster they belong
X_reduceddf_sorted = X_reduceddf.sort_values('cluster')
# Save the dataframe
X_reduceddf_sorted.to_csv('k_means_PCA.csv')
# Plot data in 3D
centroids = pca.transform(kmeans.cluster_centers_)  # Get centroids coordinates
ax = display_factorial_planes(X_reduced, 3, pca, [(0, 1, 2)], illustrative_var=clusters,
                              alpha=0.8)  # Plot the players points
ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=169, linewidths=3,  # Plot the centroids
             color='r', zorder=10)
plt.show()

for i in range(1, 6):
    # Create a data frame containing our centroids
    centroids_df = pd.DataFrame(kmeans.cluster_centers_[:, 10 * (i - 1):10 * i], columns=X.columns[10 * (i - 1):10 * i])
    centroids_df['cluster'] = centroids_df.index
    # Parallel plot of centroids values
    display_parallel_coordinates_centroids(centroids_df, num_clusters)

plt.show()

# Gaussian Mixture Model ###############################################################################################
# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(X_scaled)

# Transform the scaled data to the new PCA space
X_reduced = pca.transform(X_scaled)

gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', n_init=10).fit(X_reduced)

# Convert to a data frame
X_reduceddf = pd.DataFrame(X_reduced, index=X.index, columns=['PC1', 'PC2'])
# Add cluster column to which cluster is each player in
clusters = kmeans.predict(X_scaled)
X_reduceddf['cluster'] = clusters
# Sort players by cluster they belong
X_reduceddf_sorted = X_reduceddf.sort_values('cluster')
# Save the dataframe
X_reduceddf_sorted.to_csv('gmm_PCA.csv')

# visualize
plot_results(X_reduced, gmm.predict(X_reduced), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")
plt.show()

gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', n_init=10).fit(X_scaled)
# Get density of each Gaussian component for each sample in X_reduced
proba_array = gmm.predict_proba(X_scaled)
proba_arraydf = pd.DataFrame(proba_array, index=X.index,
                             columns=['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5',
                                      'cluster 6', 'cluster 7', 'cluster 8'])

export_3D_array("covariances.csv", gmm.covariances_)

# Add cluster column in dataframe
clusters = gmm.predict(X_scaled)
proba_arraydf['final cluster'] = clusters
# Sort players by cluster they belong
proba_arraydf_sorted = proba_arraydf.sort_values('final cluster')
# Save array
proba_arraydf_sorted.to_csv('gmm_clusters.csv')

# Create a data frame containing our centroids
for i in range(1, 6):
    centroids_gmm = pd.DataFrame(gmm.means_[:, 10 * (i - 1):10 * i], columns=X.columns[10 * (i - 1):10 * i])
    centroids_gmm['cluster'] = centroids_gmm.index

    display_parallel_coordinates_centroids(centroids_gmm, num_clusters)

plt.show()

