import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#import seaborn as sn

def plot_two_figures(fig1, fig2, title_fig1, title_fig2, figsize=(10, 10), cmap='viridis'):
    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    plt.imshow(fig1, cmap=cmap)
    plt.title(title_fig1)

    plt.subplot(2, 2, 2)
    plt.imshow(fig2, cmap=cmap)
    plt.title(title_fig2)

    plt.show()


#def plot_confusion_matrix(labels, preds, class_names):
#    matrix = confusion_matrix(labels.flatten(), preds.flatten())
#    matrix = matrix[1:, 1:]
#    df = pd.DataFrame({class_names[i]:matrix[:,i] for i in range(16)}, index=class_names)
#    fig = plt.figure(figsize = (10,6))
#    ax = fig.add_subplot()
#    ax.xaxis.set_ticks_position('top')
#    sn.heatmap(df, annot=True, cmap="OrRd", fmt='g')
#
#    plt.show()

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


