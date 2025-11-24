import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import joblib
from google.colab import files

# Mengambil X yang sudah distandarisasi di pipeline
X_scaler = kmeans_pipeline.named_steps['scaler'].transform(X)
sample_size = 200000

idx = np.random.choice(len(X_scaler), size=sample_size, replace=False)
X_sample = X_scaler[idx].astype(np.float32)
labels_sample = cluster_labels[idx]

sil = silhouette_score(X_sample, labels_sample)
print("Silhoutte Score :", sil)

dbi = davies_bouldin_score(X_sample, labels_sample)
print("Davies_Bouldin_Score:", dbi)

#Tampilan 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaler)

plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels)
plt.title('PCA 2D Clustering')
plt.show()

#Tampilan 3D
pca = PCA(n_components=3)
X_pca3 = pca.fit_transform(X_scaler)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=cluster_labels, s=2)
plt.title("PCA 3D Clustering")
plt.show()