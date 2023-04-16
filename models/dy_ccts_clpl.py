import numpy as np
from lightgbm import LGBMClassifier
from dataload import load_diabetes_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

class DYCCTSCLPL():

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

        ## randomly partition the unlabeled data
        d_ul_1, d_ul_2 = train_test_split(d_ul, test_size=1/2, random_state=self.random_seed)

        # iteration
        for i in range(self.iter):
            if d_ul_1.shape[0] == 0 or d_ul_2.shape[0] == 0:
                break
            ## odd or even
            if i % 2 == 1:
                # predict d_ul_1
                pseudo_labels_prob = self.model.predict_proba(d_ul_1)
                d_ul_new = d_ul_1
            else:
                # predict d_ul_2
                pseudo_labels_prob = self.model.predict_proba(d_ul_2)
                d_ul_new = d_ul_2

            max_values = []
            index = []
            # get max probability
            for j in range(pseudo_labels_prob.shape[0]):
                label = pseudo_labels_prob[j].argmax()
                max_values.append(pseudo_labels_prob[j][label])

            if self.Tr < 0:
                self.Tr = 0
            # get threshold
            th = np.percentile(max_values, self.Tr)
            print('iter: {} - threshold: {} - percentile: {}'.format(i, th, self.Tr))
            # get pace
            std = np.std(max_values)
            if std > self.std_bound:
                self.th_rate = self.th_rate / self.factor
            else:
                self.th_rate = self.th_rate * self.factor

            # update dataset to training
            for k in range(pseudo_labels_prob.shape[0]):
                label = pseudo_labels_prob[k].argmax()
                if pseudo_labels_prob[k][label] >= th:

                    d_l = np.vstack((d_l, d_ul_new[k]))
                    y_l = np.hstack((y_l, np.asarray(label)))
                    index.append(k)

            if i % 2 == 1:
                # update d_ul_1
                d_ul_1 = np.delete(d_ul_new, index, axis=0)
                self.model.fit(d_l, y_l)
            else:
                # update d_ul_2
                d_ul_2 = np.delete(d_ul_new, index, axis=0)
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
    clpl_cls = DYCCTSCLPL(model = LGBMClassifier(random_state = 666))
    clpl_cls.fit(d_l, y_l, d_ul)
    # predict
    y_pre = clpl_cls.predict_proba(d_test)
    # metrics
    acc = accuracy_score(y_test, np.argmax(y_pre, axis=1))
    auc = roc_auc_score(y_test, y_pre[:, 1])
    f1 = f1_score(y_test, np.argmax(y_pre, axis=1))
    recall = recall_score(y_test, np.argmax(y_pre, axis=1))
    print('accuracy: {} - auc: {} - f1 score: {} - recall: {}'.format(acc, auc, f1, recall))