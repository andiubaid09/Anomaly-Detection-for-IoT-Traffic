from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

kmeans_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(
        n_clusters=2,
        init='k-means++',
        n_init=50,
        max_iter=500,
        tol=1e-5,
        algorithm='elkan',
        random_state=42
    ))
])
kmeans_pipeline.fit(X)

# Ambil Cluster

cluster_labels = kmeans_pipeline.named_steps['kmeans'].labels_
df['cluster'] = cluster_labels

X_cluster = X.copy()
X_cluster['cluster'] = cluster_labels

df.head(5)
X_cluster.head(5)
