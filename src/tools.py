import numpy as np
from sklearn.decomposition import PCA


def remove_unclassified(preds, labels):
    preds_cpy = preds.copy()
    zeros_idx = np.argwhere(labels.flatten() == 0).flatten()
    preds_cpy += 1
    preds_flatten = preds_cpy.flatten()
    preds_flatten[zeros_idx] = 0
    final_preds = preds_flatten.reshape((preds.shape[0], preds.shape[1]))
    return final_preds

def get_number_components(X):
    # first PCA with by keeping all features
    print("Fitting the PCA")
    pca_model = PCA()
    pca_model.fit(X)
    var_cumsum = pca_model.explained_variance_ratio_.cumsum()
    return len(var_cumsum[var_cumsum <= 0.9991])

def extract_features(X, n_components):
    pca_model = PCA(n_components=n_components)
    pca_model.fit(X)
    return pca_model.transform(X)

def copy_without_outlier(X, outlier_pos):
    l = []
    X_positions = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if [i, j] in outlier_pos:
                continue
            l.append(X[i, j])
            X_positions.append([i, j])
    return np.array(l), np.array(X_positions)

def rebuild_data_with_outliers(salinas_preds, X_positions, outlier_pos, target_shape):
    preds = np.empty(target_shape)
    for i in range(salinas_preds.shape[0]):
        pos = X_positions[i]
        preds[pos[0], pos[1]] = salinas_preds[i]
    for j in range(len(outlier_pos)):
        pos = outlier_pos[j]
        preds[pos[0], pos[1]] = 0
    return preds

def get_label(labels, x):
    return np.where(labels == x, labels, 0)

def compute_labels_correspondence(labels, preds, n_cluster):
    l2, c2 = np.unique(labels, return_counts=True)
    c2 = c2.argsort()
    l, c = np.unique(preds, return_counts=True)
    c = c.argsort()

    res_labels = labels.copy().flatten()
    res_preds = preds.copy().flatten()

    j = 0
    offset = len(l2) - len(l)

    for i in range(n_cluster):
        if i in l:
            idx = np.argwhere(preds.flatten() == l[c[j]]).flatten()
            res_preds[idx] = j + offset + 1
            j += 1

        idx2 = np.argwhere(labels.flatten() == l2[c2[i]]).flatten()
        res_labels[idx2] = i + 1

    zeros_idx = np.argwhere(labels.flatten() == 0).flatten()
    res_preds[zeros_idx] = 0
    res_preds = res_preds.reshape((preds.shape[0], preds.shape[1]))
    res_labels = res_labels.reshape((labels.shape[0], labels.shape[1]))

    return res_labels, res_preds
