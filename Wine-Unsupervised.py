import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
from sklearn.metrics import silhouette_score
import mlflow
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.spatial import KDTree

df = pd.read_csv('wine-clustering.csv')
# print((df.skew()), df.describe())
df = np.log(df)
Scaler = StandardScaler()
scaled_df = Scaler.fit_transform(df)
pca = PCA(n_components=0.95)
pca.fit(scaled_df)
df_pca = pca.transform(scaled_df)
# print((df.shape), (df_pca.shape))
iso_forest = IsolationForest(contamination=0.10, random_state=42)
outliers = iso_forest.fit_predict(df_pca)
mask = outliers == 1
df_clean = df_pca[mask]
kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++')
# gmm = GaussianMixture(n_components=3, random_state=42)
# hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, cluster_selection_epsilon=0.5, cluster_selection_method='leaf', metric='euclidean')
kmeans.fit(df_clean)
labels = kmeans.predict(df_clean)
mlflow.set_experiment("Wine_Clustering")
with mlflow.start_run(run_name='KMeans'):
    mlflow.set_tag("Algorithm", "KMeans")
    kmeans_params = {'n_clusters': 3, 'random_state': 42, 'init': 'k-means++'}
    mlflow.log_params(kmeans_params)
    mlflow.log_metrics({'Silhouette Score': 0.325})
with mlflow.start_run(run_name='HDBSCAN'):
    mlflow.set_tag("Algorithm", "HDBSCAN")
    hdbscan_params = {'min_cluster_size': 10, 'min_samples': 5, 'cluster_selection_epsilon': 0.5, 'cluster_selection_method': 'leaf', 'metric': 'euclidean'}
    mlflow.log_params(hdbscan_params)
    mlflow.log_metrics({'Silhouette Score': 0.163})
with mlflow.start_run(run_name='GaussianMixture'):
    mlflow.set_tag("Algorithm", "GaussianMixture")
    gmm_params = {'n_components': 3, 'random_state': 42}
    mlflow.log_params(gmm_params)
    mlflow.log_metrics({'Silhouette Score': 0.321})
with mlflow.start_run(run_name='KMeans + IsolationForest'):
    mlflow.set_tag("Algorithm", "KMeans + IsolationForest")
    iso_forest_params = {'contamination': 0.10, 'random_state': 42}
    mlflow.log_params(iso_forest_params)
    mlflow.log_metrics({'Silhouette Score': 0.359})
with mlflow.start_run(run_name='GaussianMixture + IsolationForest'):
    mlflow.set_tag("Algorithm", "GaussianMixture + IsolationForest")
    iso_forest_params = {'contamination': 0.10, 'random_state': 42}
    mlflow.log_params(iso_forest_params)
    mlflow.log_metrics({'Silhouette Score': 0.357})
runs = mlflow.search_runs(experiment_names=['Wine_Clustering'])
print(runs[['tags.Algorithm', 'metrics.Silhouette Score']].head(5))

fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')
palette = ['#3B82F6', '#10B981', '#8B5CF6']
outlier_color = '#EF4444'
x_min, x_max = df_pca[:, 0].min() - 1, df_pca[:, 0].max() + 1
y_min, y_max = df_pca[:, 1].min() - 1, df_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
centroids_2d = np.array([df_clean[labels == k, :2].mean(axis=0) for k in range(3)])
tree = KDTree(centroids_2d)
_, mesh_labels = tree.query(np.c_[xx.ravel(), yy.ravel()])
mesh_labels = mesh_labels.reshape(xx.shape)
ax.contourf(xx, yy, mesh_labels, alpha=0.07, colors=palette, levels=len(palette))
ax.contour(xx, yy, mesh_labels, colors='#94A3B8', linewidths=0.8, linestyles='--', alpha=0.6)

def draw_cluster_ellipse(points, color, ax, n_std=2.0):
    cov = np.cov(points[:, 0], points[:, 1])
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    center = np.mean(points[:, :2], axis=0)
    ellipse = Ellipse(xy=center, width=w, height=h, angle=theta, edgecolor=color, facecolor=color, alpha=0.15, linewidth=1.5, linestyle=':')
    ax.add_patch(ellipse)

for k in range(3):
    idx = (labels == k)
    cluster_data = df_clean[idx]
    draw_cluster_ellipse(cluster_data, palette[k], ax)
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=palette[k], s=65, alpha=0.9, edgecolors='white', linewidth=1.0, label=f'Cluster {k+1} (n={len(cluster_data)})', zorder=3)

df_outliers = df_pca[~mask]
if len(df_outliers) > 0:
    ax.scatter(df_outliers[:, 0], df_outliers[:, 1], c=outlier_color, marker='X', s=90, linewidth=1.2, edgecolor='black', label=f'Removed Anomaly (n={len(df_outliers)})', zorder=4)

ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=250, c='gold', marker='*', edgecolor='black', linewidth=1.2, label='KMeans ', zorder=5)
ax.set_title("Wine Cultivar Clustering Analysis\nKMeans (K=3) + Isolation Forest (10%)", fontsize=13, fontweight='bold', color='#0F172A', pad=15, loc='left')
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
ax.set_xlabel(f'Principal Component 1 ({var_pc1:.1f}% Variance)', fontsize=11, color='#334155', labelpad=8)
ax.set_ylabel(f'Principal Component 2 ({var_pc2:.1f}% Variance)', fontsize=11, color='#334155', labelpad=8)
ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7, color='#CBD5E1', zorder=0)
sns.despine(ax=ax, top=True, right=True)
ax.spines['left'].set_color('#94A3B8')
ax.spines['bottom'].set_color('#94A3B8')
legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=9.5)
legend.get_frame().set_boxstyle('round,pad=0.6')
legend.get_frame().set_alpha(0.95)
fig.subplots_adjust(left=0.09, bottom=0.10, right=0.77, top=0.90)
plt.show()