import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cf_matrix, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix for a given set of classes
    """
    ax = plt.subplot()
    
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap=cmap, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    
    plt.title(title)
    
    plt.show()