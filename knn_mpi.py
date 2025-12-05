from mpi4py import MPI
import numpy as np
from collections import Counter
import sys

# Rank 0 
if MPI.COMM_WORLD.Get_rank() == 0:
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------- Parámetros  -----------------

train_size   = int(sys.argv[1]) if len(sys.argv) >= 2 else 1000
test_size    = int(sys.argv[2]) if len(sys.argv) >= 3 else 200
k            = int(sys.argv[3]) if len(sys.argv) >= 4 else 3
num_features = int(sys.argv[4]) if len(sys.argv) >= 5 else 10
# --------------------------------------------------------

# data 
if rank == 0:
    X, y = make_classification(
        n_samples=train_size + test_size,
        n_features=num_features,
        n_informative=6, 
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
else:
    X_train = y_train = X_test = y_test = None

X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# training data
local_train_size = train_size // size
local_X = np.empty((local_train_size, num_features), dtype='float64')
local_y = np.empty(local_train_size, dtype='int')

t_start = MPI.Wtime()
comm.Scatter(X_train, local_X, root=0)
comm.Scatter(y_train, local_y, root=0)
t_dist = MPI.Wtime()

local_predictions = []
for x in X_test:
    dists = euclidean_distance(local_X, x)
    k_indices = dists.argsort()[:k]
    k_labels = local_y[k_indices]
    local_predictions.append((dists[k_indices], k_labels))
t_comp = MPI.Wtime()

all_dists = comm.gather(local_predictions, root=0)
t_gather = MPI.Wtime()

if rank == 0:
    final_preds = []
    for i in range(test_size):
        all_neighbors = []
        for proc_preds in all_dists:
            all_neighbors.extend(zip(proc_preds[i][0], proc_preds[i][1]))
        all_neighbors.sort(key=lambda x: x[0])
        top_k = [label for _, label in all_neighbors[:k]]
        final_pred = Counter(top_k).most_common(1)[0][0]
        final_preds.append(final_pred)

    final_preds = np.array(final_preds)
    accuracy = np.mean(final_preds == y_test)

    print(f"[Process Count: {size}] Train Size: {train_size} | Test Size: {test_size} | k={k} | d={num_features}")
    print(f"Total Time       : {t_gather - t_start:.4f} sec")
    print(f"  - Distribution : {t_dist - t_start:.4f} sec")
    print(f"  - Computation  : {t_comp - t_dist:.4f} sec")
    print(f"  - Gathering    : {t_gather - t_comp:.4f} sec")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Etiquetas reales : {y_test}")
    print(f"Predicciones     : {final_preds}")

    pca = PCA(n_components=2)
    X_test_2D = pca.fit_transform(X_test)