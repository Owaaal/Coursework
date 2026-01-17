import pandas
import numpy
import sklearn
import seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist, squareform
RANDOM_SEED = 42
np.random.seed(seed=RANDOM_SEED)


X1 = np.random.randn(50, 2) + np.array([1, 1])
X2 = np.random.randn(50, 2) + np.array([-1, -1])
X = np.vstack([X1, X2])
print(X)
labels = np.array([0]*50 + [1]*50)

kmeans = KMeans(n_clusters=2, n_init=10, verbose=1, random_state=42)
kmeans.fit(X)
print(kmeans.cluster_centers_)


#Labels =  True lables, kmeans.labels = pred labels
ct = pandas.crosstab(labels, kmeans.labels_, rownames=['True'], colnames=['KMeans'])
print(ct)
mapping = {}
for cluster in sorted(ct.columns):
    majority_label = ct[cluster].idxmax()
    mapping[cluster] = majority_label

pred = kmeans.labels_
pred_mapped = pd.Series(pred).map(mapping).to_numpy()
incorrect_mask = (pred_mapped != labels)
incorrect_indices = np.where(incorrect_mask)[0]
print(f"Number incorrectly clustered: {incorrect_indices.size}")
if incorrect_indices.size > 0:
    # Build a DataFrame for easy inspection
    mis_df = pd.DataFrame({
        'index': incorrect_indices,
        'x': X[incorrect_indices, 0],
        'y': X[incorrect_indices, 1],
        'true_label': labels[incorrect_indices],
        'kmeans_label': pred[incorrect_indices],
        'mapped_pred': pred_mapped[incorrect_indices]
    })
    print('Incorrectly clustered samples (index, x, y, true_label, kmeans_label, mapped_pred):')
    print(mis_df.to_string(index=False))





# Create subplots: 3 panels now
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (i) Scatter plot without labels
axes[0].scatter(X[:, 0], X[:, 1], c='gray', edgecolor='k')
axes[0].set_title('Scatter plot: All points')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# (ii) Scatter plot with ground-truth labels
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='bwr', edgecolor='k')
axes[1].set_title('Scatter plot: Coloured by ground-truth')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

# (iii) Scatter plot with kmeans labels and centroids
klabels = kmeans.labels_
scatter = axes[2].scatter(X[:, 0], X[:, 1], c=klabels, cmap='bwr', edgecolor='k')
# plot centroids
centers = kmeans.cluster_centers_
axes[2].scatter(centers[:, 0], centers[:, 1], c='yellow', s=200, marker='X', edgecolor='k', label='centroids')
axes[2].set_title('Scatter plot: Coloured by kmeans labels + centroids')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')
axes[2].legend()

plt.tight_layout()
plt.show()


# --- Elbow method: compute RSS (inertia) for K=2..20 ---
Ks = list(range(2, 21))
inertias = []
for K in Ks:
    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_SEED)
    km.fit(X)
    inertias.append(km.inertia_)

print('\nK vs Inertia (RSS):')
for k, val in zip(Ks, inertias):
    print(f'K={k}: inertia={val:.4f}')

# Plot RSS vs K
plt.figure(figsize=(8, 4))
plt.plot(Ks, inertias, marker='o')
plt.xticks(Ks)
plt.xlabel('Number of clusters K')
plt.ylabel('Total within-cluster RSS (inertia)')
plt.title('Elbow plot: RSS vs K')
plt.grid(True)
plt.show()