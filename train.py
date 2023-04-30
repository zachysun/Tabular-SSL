import pandas as pd
from dataload import load_diabetes_data

from models.clpl import CLPL
from models.cts_clpl import CTSCLPL
from models.cts_bi_clpl import CTSBICLPL
from models.dy_cts_clpl import DYCTSCLPL
from models.dy_ccts_clpl import DYCCTSCLPL
from models.ccts_clpl import CCTSCLPL
from models.dy_clpl import DYCLPL
from models.npl import confi

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix)

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

import warnings

warnings.filterwarnings("ignore")


def binary_cls(label_rate=0.1, thrate=12, iters=5):
    x_lab, y_lab, x_unlabel, x_test, y_test, x_train, y_train = load_diabetes_data(label_data_rate=label_rate)

    accuracy, precision, recall, f1, auc, conf_mat = [], [], [], [], [], []

    random_state = 5

    rf = RandomForestClassifier(random_state=random_state)
    xg = XGBClassifier(random_state=random_state)
    lgb = LGBMClassifier(random_state=random_state)
    ee = VotingClassifier(estimators=[('rf', rf), ('xg', xg), ('lgb', lgb), ], voting='soft', weights=[1, 1, 1])

    scls = []

    scls.append(SVC(random_state=random_state, probability=True))
    scls.append(RandomForestClassifier(random_state=random_state))
    scls.append(KNeighborsClassifier())
    scls.append(LogisticRegression(random_state=random_state))
    scls.append(XGBClassifier(random_state=random_state))
    scls.append(LGBMClassifier(random_state=random_state))
    scls.append(VotingClassifier(estimators=[('rf', rf), ('xg', xg), ('lgb', lgb), ], voting='soft', weights=[1, 1, 1]))

    npls = []

    npls.append(confi(ee, x_unlabel, sample_rate=0.2, verbose=True))
    npls.append(confi(RandomForestClassifier(random_state=random_state), x_unlabel, sample_rate=0.2, verbose=True))

    sslcls = []
    sslcls.append(CLPL(model=RandomForestClassifier(random_state=random_state), th_rate=thrate, iter=iters))
    sslcls.append(CTSCLPL(model=RandomForestClassifier(random_state=random_state), th_rate=thrate, iter=iters))

    ves = []
    ves.append(CLPL(model=ee, th_rate=thrate, iter=iters))
    ves.append(CTSCLPL(model=ee, th_rate=thrate, iter=iters, random_seed=4))
    ves.append(CTSBICLPL(model=ee, th_rate=thrate, iter=iters, random_seed=4, start_th=95))
    ves.append(DYCTSCLPL(model=ee, th_rate=thrate, iter=iters, random_seed=4, factor=2, std_bound=0.12))
    ves.append(DYCCTSCLPL(model=ee, th_rate=thrate, iter=iters, random_seed=4, factor=2, std_bound=0.12))
    ves.append(
        DYCCTSCLPL(model=RandomForestClassifier(random_state=random_state), th_rate=thrate, iter=iters, random_seed=4,
                   factor=2, std_bound=0.12))
    ves.append(CCTSCLPL(model=ee, th_rate=thrate, iter=iters, random_seed=4))
    ves.append(DYCLPL(model=ee, th_rate=thrate, iter=iters, random_seed=4, factor=2, std_bound=0.12))

    # supervised
    for clsf in scls:
        clf = clsf
        clf.fit(x_train, y_train)

        y_preds = clf.predict(x_test)
        y_probs = clf.predict_proba(x_test)

        accuracy.append(accuracy_score(y_test, y_preds))
        precision.append(precision_score(y_test, y_preds))
        recall.append((recall_score(y_test, y_preds)))
        f1.append((f1_score(y_test, y_preds)))
        auc.append((roc_auc_score(y_test, y_probs[:, 1])))
        conf_mat.append(confusion_matrix(y_test, y_preds))
    # npls
    for clsf in npls:
        clf = clsf
        clf.fit(x_lab, y_lab)

        y_preds = clf.predict(x_test)
        y_probs = clf.predict_proba(x_test)

        accuracy.append(accuracy_score(y_test, y_preds))
        precision.append(precision_score(y_test, y_preds))
        recall.append((recall_score(y_test, y_preds)))
        f1.append((f1_score(y_test, y_preds)))
        auc.append((roc_auc_score(y_test, y_probs[:, 1])))
        conf_mat.append(confusion_matrix(y_test, y_preds))
    # cts
    for csslcls in sslcls:
        clf = csslcls
        clf.fit(x_lab, y_lab, x_unlabel)

        y_preds = clf.predict(x_test)
        y_probs = clf.predict_proba(x_test)

        accuracy.append(accuracy_score(y_test, y_preds))
        precision.append(precision_score(y_test, y_preds))
        recall.append((recall_score(y_test, y_preds)))
        f1.append((f1_score(y_test, y_preds)))
        auc.append((roc_auc_score(y_test, y_probs[:, 1])))
        conf_mat.append(confusion_matrix(y_test, y_preds))
    # ve-cts
    for ve in ves:
        clf = ve
        clf.fit(x_lab, y_lab, x_unlabel)

        y_preds = clf.predict(x_test)
        y_probs = clf.predict_proba(x_test)

        accuracy.append(accuracy_score(y_test, y_preds))
        precision.append(precision_score(y_test, y_preds))
        recall.append((recall_score(y_test, y_preds)))
        f1.append((f1_score(y_test, y_preds)))
        auc.append((roc_auc_score(y_test, y_probs[:, 1])))
        conf_mat.append(confusion_matrix(y_test, y_preds))

    results_df = pd.DataFrame(
        {"Accuracy Score": accuracy,
         "Precision Score": precision,
         "Recall Score": recall,
         "f1 Score": f1,
         "AUC Score": auc,
         "Confusion Matrix": conf_mat,
         "Algos": ["SVC",
                   "RandomForest",
                   "KNeighbours",
                   "LogisticRegression",
                   "XGBoost",
                   "LightGBM",
                   "ve",
                   "venpl",
                   "npl",
                   "clpl",
                   "ctsclpl",
                   "veclpl",
                   "vectsclpl",
                   "vectsbiclpl",
                   "dyvectsclpl",
                   "dyvecctsclpl",
                   "dycctsclpl",
                   "vecctsclpl",
                   "dyveclpl"]})

    results = results_df
    # results = (results_df.sort_values(by = ['AUC Score','f1 Score'], ascending = False).reset_index(drop =  True))

    return scls, sslcls, results


def ran_train(epoch=20, lr=0.1, thr=12, itr=5):
    _, _, results_1 = binary_cls(label_rate=lr, thrate=thr, iters=itr)
    total_results = results_1
    for i in range(epoch - 1):
        _, _, results = binary_cls(label_rate=lr)
        total_results.iloc[:, 0:6] = total_results.iloc[:, 0:6] + results.iloc[:, 0:6]

    total_results.iloc[:, 0:6] = total_results.iloc[:, 0:6] / epoch
    total_results = (total_results.sort_values(by=['AUC Score', 'f1 Score', 'Accuracy Score', ], ascending=False)
                     .reset_index(drop=True))

    return total_results


if __name__ == '__main__':
    results = ran_train(epoch=30, lr=0.1, thr=12, itr=5)
    print(results)
