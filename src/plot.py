import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm

from tools import get_label

def plot_two_figures(fig1, fig2, title_fig1='', title_fig2='', figsize=(10, 10), cmap='viridis'):
    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    plt.imshow(fig1, cmap=cmap)
    plt.title(title_fig1)

    plt.subplot(2, 2, 2)
    plt.imshow(fig2, cmap=cmap)
    plt.title(title_fig2)

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
