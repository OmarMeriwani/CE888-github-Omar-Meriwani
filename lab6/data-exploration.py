import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, metrics
import seaborn as sns

n_samples = 1500 # no.of.data points

# The dataset function is avialable in sklearn package
noisy_moons,moon_labels = datasets.make_moons(n_samples=n_samples, noise=.1) # Generate Moon Toy Dataset
noisy_circles,circle_labels = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05) # Generate Circle Toy Dataset

# noisy_moons.shape
# moon_labels.shape

# Put in Array
noisy_moons=np.array(noisy_moons)
noisy_circles = np.array(noisy_circles)

# Plot Half-moon data
plt.figure(figsize=(8,5))
plt.title("Half-moon shaped data", fontsize=18)
plt.grid(True)
plt.scatter(noisy_moons[:,0],noisy_moons[:,1])
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/HALF_MOON.png', dpi=300)
plt.show()

# Plot Circle data
plt.figure(figsize=(8,5))
plt.title("Concentric circles of data points", fontsize=18)
plt.grid(True)
plt.scatter(noisy_circles[:,0],noisy_circles[:,1])
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/CIRCLE.png', dpi=300)
plt.show()

# Fit K-Means Clustering on noise moon data
km=cluster.KMeans(n_clusters=2)
km.fit(noisy_moons)
km.labels_

print("Completeness: %0.3f" % metrics.completeness_score(moon_labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(noisy_moons, km.labels_))


plt.figure(figsize=(8,5))
plt.title("Half-moon shaped data", fontsize=18)
plt.grid(True)
plt.scatter(noisy_moons[:,0],noisy_moons[:,1],c=km.labels_)
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/CLUSTER_MOON.png', dpi=300)
plt.show()

# Fit K-Means Clustering on noise Circle data
km.fit(noisy_circles)

print("Completeness: %0.3f" % metrics.completeness_score(circle_labels, km.labels_))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(noisy_circles, km.labels_))

plt.figure(figsize=(8,5))
plt.title("Concentric circles of data points", fontsize=18)
plt.grid(True)
plt.scatter(noisy_circles[:,0],noisy_circles[:,1],c=km.labels_)
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/CLUSTER_CIRCLE.png', dpi=300)
plt.show()

# Fit DBSCAN Clustering on noise moon data
dbs = cluster.DBSCAN(eps=0.1) # The maximum distance between two samples for them to be considered as in the same neighborhood.
dbs.fit(noisy_moons)
dbs.labels_

plt.figure(figsize=(8,5))
plt.title("Half-moon shaped data", fontsize=18)
plt.grid(True)
plt.scatter(noisy_moons[:,0],noisy_moons[:,1],c=dbs.labels_)
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/DBSCAN_MOON.png', dpi=300)
plt.show()

# Fit DBSCAN Clustering on noise Circle data
dbs.fit(noisy_circles)
dbs.labels_

plt.figure(figsize=(8,5))
plt.title("Concentric circles of data points", fontsize=18)
plt.grid(True)
plt.scatter(noisy_circles[:,0],noisy_circles[:,1],c=dbs.labels_)
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/DBSCAN_CIRCLE.png', dpi=300)
plt.show()


from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
noisy_moons_spca = scikit_pca.fit_transform(noisy_moons)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))

ax[0].scatter(noisy_moons_spca[moon_labels == 0, 0], noisy_moons_spca[moon_labels == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(noisy_moons_spca[moon_labels == 1, 0], noisy_moons_spca[moon_labels == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(noisy_moons_spca[moon_labels == 0, 0], np.zeros((750, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(noisy_moons_spca[moon_labels == 1, 0], np.zeros((750, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/PCA_MOON.png', dpi=300)
plt.show()


# Compute the correlation matrix before doing PCA
nm=pd.DataFrame(noisy_moons)
pca_corr=nm.corr()
print(pca_corr)
sns.heatmap(pca_corr, vmax=1, center=0, square=True)
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/CORR_PCA.png', dpi=300)

# Compute the correlation matrix after doing PCA
nmp=pd.DataFrame(noisy_moons_spca)
corr_pca=nmp.corr()
print(corr_pca)
sns.heatmap(corr_pca, vmax=1, center=0, square=True)
plt.savefig('C:/Users/hr17576/OneDrive - University of Essex/Teaching/CE888/ce888-master_HR/ce888-master/graphics/lec6/PCA_CORR.png', dpi=300)