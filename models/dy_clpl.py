'''
Easy-Curriculum-Labeling for tabular data
'''
import numpy as np
from lightgbm import LGBMClassifier
from dataload import load_diabetes_data
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

class DYCLPL():

    def __init__(self, model, th_rate=10, iter=10, random_seed = 7, factor = 2, std_bound = 0.12):

        self.model = model
        self.th_rate = th_rate
        self.Tr = 100 - th_rate
        self.iter = iter
        self.random_seed = random_seed
        self.factor = factor
        self.std_bound = std_bound

    def fit(self, d_l, y_l, d_ul):
        d_l = d_l
        y_l = y_l
        self.model.fit(d_l, y_l)
        # iteration
        for i in range(self.iter):
            if len(d_ul)  == 0:
                break
            max_values = []
            index = []
            pseudo_labels_prob = self.model.predict_proba(d_ul)
            # get max probability
            for j in range(pseudo_labels_prob.shape[0]):
                label = pseudo_labels_prob[j].argmax()
                max_values.append(pseudo_labels_prob[j][label])

            if self.Tr < 0:
                self.Tr = 0
            # get threshold
            th = np.percentile(max_values, self.Tr)
            print('iter: {} - threshold: {} - percentile: {}'.format(i, th, self.Tr))

            std = np.std(max_values)
            if std > self.std_bound:
                self.th_rate = self.th_rate / self.factor
            else:
                self.th_rate = self.th_rate * self.factor
            # update dataset to training

            for k in range(pseudo_labels_prob.shape[0]):
                label = pseudo_labels_prob[k].argmax()
                if pseudo_labels_prob[k][label] >= th:
                    d_l = np.vstack((d_l, d_ul[k]))
                    y_l = np.hstack((y_l, np.asarray(label)))
                    index.append(k)

            # update d_ul
            d_ul = np.delete(d_ul, index, axis=0)
            self.model.fit(d_l, y_l)
            # update threshold
            self.Tr = self.Tr - self.th_rate

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

if __name__=='__main__':

    d_l, y_l, d_ul, d_test, y_test, _ , _ = load_diabetes_data(label_data_rate=0.1)
    # build model
    clpl_cls = DYCLPL(model = LGBMClassifier())
    clpl_cls.fit(d_l, y_l, d_ul)
    # predict
    y_pre = clpl_cls.predict_proba(d_test)
    # metrics
    acc = accuracy_score(y_test, np.argmax(y_pre, axis=1))
    auc = roc_auc_score(y_test, y_pre[:, 1])
    f1 = f1_score(y_test, np.argmax(y_pre, axis=1))
    recall = recall_score(y_test, np.argmax(y_pre, axis=1))
    print('accuracy: {} - auc: {} - f1 score: {} - recall: {}'.format(acc, auc, f1, recall))


