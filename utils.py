import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score


def eval(metric, y_test, y_test_hat):

    # Accuracy metric
    global result
    if metric == 'acc':
        result = accuracy_score(np.argmax(y_test, axis=1),
                                np.argmax(y_test_hat, axis=1))
    # AUROC metric
    elif metric == 'auc':
        result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])

    # F1 metric
    elif metric == 'f1':
        result = f1_score(np.argmax(y_test, axis=1),
                          np.argmax(y_test_hat, axis=1))
    # Rec
    elif metric == 'rec':
        result = recall_score(np.argmax(y_test, axis=1),
                              np.argmax(y_test_hat, axis=1))

    return result


def matrix2vec(matrix):

    no = matrix.shape[0]
    dim = matrix.shape[1]
    # Define output
    vector = np.zeros([no, ])

    # Convert matrix to vector
    for i in range(dim):
        idx = np.where(matrix[:, i] == 1)
        vector[idx] = i

    return vector

