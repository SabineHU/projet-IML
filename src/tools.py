import numpy as np
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
