
import numpy as np
from sklearn.decomposition import PCA

def pcp(X):
    """
    Performs Principal Component Pursuit (PCP) on a matrix.

    Args:
        X (numpy.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The low-rank component of the matrix.
    """
    # Set the tuning parameter lambda
    lmbda = 1 / np.sqrt(max(X.shape))

    # Set the maximum number of iterations
    max_iter = 100

    # Initialize the low-rank and sparse components
    L = np.zeros(X.shape)
    S = np.zeros(X.shape)

    # Iterate until convergence
    for i in range(max_iter):
        # Update the low-rank and sparse components using PCA and the soft thresholding operator
        U, s, Vt = np.linalg.svd(X - S, full_matrices=False)
        L = U @ np.diag(soft_threshold(s, lmbda)) @ Vt
        S = soft_threshold(X - L, lmbda)

    return L

def soft_threshold(x, lmbda):
    """
    Computes the soft thresholding operator.

    Args:
        x (numpy.ndarray): The input vector or matrix.
        lmbda (float): The threshold.

    Returns:
        numpy.ndarray: The thresholded vector or matrix.
    """
    return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_auc(scores, labels, title=None):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve' if title is None else title)
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc
