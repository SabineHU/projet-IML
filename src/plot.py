import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from matplotlib.colors import LogNorm

from tools import get_label

def plot_two_figures(fig1, fig2, title_fig1='', title_fig2='', figsize=(10, 10), cmap='viridis', suptitle=''):
    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    plt.imshow(fig1, cmap=cmap)
    plt.title(title_fig1)

    plt.subplot(2, 2, 2)
    plt.imshow(fig2, cmap=cmap)
    plt.title(title_fig2)

    plt.suptitle(suptitle, fontsize=16)
    plt.show()

def plot_confusion_matrix(labels, preds, class_names, title='Confusion matrix', log=False):
    # Plot confusion matrix using log / linear scale

    # Get confusion matrix and skip background (label 0)
    matrix = confusion_matrix(labels.flatten(), preds.flatten())
    matrix = matrix[1:, 1:]

    df = pd.DataFrame({class_names[i]:matrix[:,i] for i in range(len(class_names))}, index=class_names)
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot()
    ax.xaxis.set_ticks_position('top')
    if log:
        sn.heatmap(df, annot=True, cmap="OrRd", fmt='g', norm=LogNorm())
    else:
        sn.heatmap(df, annot=True, cmap="OrRd", fmt='g')

    plt.title(title)
    plt.show()

def plot_each_labels(labels, predictions, nb_labels, title_lab='Expected', title_pred='Predicted', figsize=(10, 10), cmap='viridis', begin=0, end=0):

    # Plot each label from begin to end
    for x in range(begin, nb_labels - end, 2):
        plt.figure(figsize=figsize)
        plt.subplot(4, 4, 1)
        plt.imshow(get_label(labels, x + 1), cmap=cmap)
        plt.title("{} label {}".format(title_lab, x + 1))

        plt.subplot(4, 4, 2)
        plt.imshow(get_label(predictions, x + 1), cmap=cmap)
        plt.title("{} label {}".format(title_pred, x + 1))

        plt.subplot(4, 4, 3)
        plt.imshow(get_label(labels, x + 2), cmap=cmap)
        plt.title("{} label {}".format(title_lab, x + 2))

        plt.subplot(4, 4, 4)
        plt.imshow(get_label(predictions, x + 2), cmap=cmap)
        plt.title("{} label {}".format(title_pred, x + 2))
        plt.show()

    if nb_labels % 2 == 1 and end >= nb_labels:
        plt.figure(figsize=(20, 20))
        plt.subplot(4, 4, 1)
        plt.imshow(get_label(labels, nb_labels), cmap=cmap)
        plt.title("{} label {}".format(title_lab, nb_labels))

        plt.subplot(4, 4, 2)
        plt.imshow(get_label(predictions, nb_labels), cmap=cmap)
        plt.title("{} label {}".format(title_pred, nb_labels))
        plt.show()

def plot_histogram(arr, figsize=(10,4), title='Histogram per classes'):
    '''
    Plot of the histogram of the different classes

    params:
    ----------
    arr: Flatten labels array-like of shape (n_samples,)

    figsize: matplotlib figsize of the plot

    title: Title of the matplotlib plot
    '''
    fig, axs = plt.subplots(figsize=figsize)
    bar_x, bar_count = np.unique(arr, return_counts=True)

    bar = axs.bar(bar_x, bar_count, 0.6)
    axs.bar_label(bar, padding=2)

    axs.set_xticks(np.arange(len(bar_x)))
    axs.set_xticklabels(bar_x.astype(int))

    axs.set_title(title)
    plt.show()

def plot_bands(hsi_img, figsize=(20,10), title='Spectral Bands for specific pixels'):
    '''
    Plot bands for specific pixels of the image
    Check if step bands are correlated between each other
    Check if different pixels have similar bands

    params:
    ----------
    hsi_img: Array-like of shape (n_pixels_row, n_pixels_col, n_features)
        Hyperspectral image -> n_features == number of bands

    figsize: matplotlib figsize of the plot

    title: Title of the matplotlib plot
    '''
    dim = hsi_img.shape[-1]
    plt.figure(figsize=figsize)
    plt.subplot(122)
    plt.plot(np.arange(1, dim + 1), hsi_img[100, 100, :], 'b')
    plt.plot(np.arange(1, dim + 1), hsi_img[100, 120, :], 'r')
    plt.plot(np.arange(1, dim + 1), hsi_img[120, 100, :], 'k')
    plt.plot(np.arange(1, dim + 1), hsi_img[120, 120, :], 'g')
    plt.xlim(1, dim + 1)
    plt.legend(['Pixel (100, 100)', 'Pixel (100, 120)', 'Pixel (120, 100)', 'Pixel (120, 120)'], loc='upper right')
    plt.title(title)
    plt.show()

def plot_correlation(X, figsize=(6, 6), title='Correlation plot as image'):
    '''
    Plot correlation between features as an image

    params:
    ----------
    X: array-like of shape (n_samples, n_features)

    figsize: matplotlib figsize of the plot

    title: Title of the matplotlib plot
    '''
    mat_coef = np.corrcoef((X), rowvar=False)
    plt.figure(figsize=(6,6))
    plt.imshow(mat_coef, cmap='gray')
    plt.title(title)
    plt.show()

def plot_pca_components(X, figsize=(16, 6), title='Explained variance per PC'):
    '''
    Compute PCA on input data
    Display the variance according to each components

    params:
    ----------
    X: array-like of shape (n_samples, n_features)

    figsize: matplotlib figsize of the plot

    title: Title of the matplotlib plot


    returns:
    ----------
    pca_model: PCA model after the fit
    '''
    dim = X.shape[-1]

    # first PCA with by keeping all features
    pca_model = PCA()
    pca_model.fit(X)

    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.title(title)
    plt.plot(np.arange(1, dim + 1), pca_model.explained_variance_, 'b')
    plt.xlabel('PC number')
    plt.xlim(1, dim + 1)
    plt.subplot(122)
    plt.title(title + ' (in log scale)')
    plt.plot(np.arange(1, dim + 1), pca_model.explained_variance_,'b')
    plt.xlabel('PC number')
    plt.xlim(1, dim+1)
    plt.yscale('log')
    plt.show()

    return pca_model
