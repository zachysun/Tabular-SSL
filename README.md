### Tabular-SSL Testing Lib:

- Tabular data 

- Semi-supervised learning 

- Curriculum learning 

- Pseudo label

- Classical machine learning 

---

#### Overview:

| Supervised Methods  |             Description             |
| :-----------------: | :---------------------------------: |
|         SVC         |                  -                  |
|     KNeighbours     |                  -                  |
| Logistic Regression |                  -                  |
|       XGBoost       |                  -                  |
|      LightGBM       |                  -                  |
|    RandomForest     |                  -                  |
|  VotingClassifier   | RandomForest \| XGBoost \| LightGBM |

| Semi-Supervised Methods |                         Description                          |
| :---------------------: | :----------------------------------------------------------: |
|           npl           |                 Normal pseudo label learning                 |
|         ve-npl          |       VotingClassifier \| Normal pseudo label learning       |
|          clpl           |             Curriculum learning \| Pseudo label              |
|         ve-clpl         |   VotingClassifier \| Curriculum learning \| Pseudo label    |
|        cts-clpl         | Cross-training strategy \| Curriculum learning \| Pseudo label |
|       ve-cts-clpl       | VotingClassifier \| Cross-training strategy \| Curriculum learning \| Pseudo label |
|      ve-ccts-clpl       | VotingClassifier \| Cross-training strategy \| Curriculum learning \| Pseudo label |
|     ve-cts-bi-clpl      |                                                              |
|       dy-ve-clpl        |                                                              |
|      dy-ccts-clpl       |                                                              |
|     dy-ve-cts-clpl      |                                                              |
|     dy-ve-ccts-clpl     |                                                              |

---

#### Notes:

- Difference between 'cts' and 'ccts':
- Data link for test [PIMA](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

#### Results:

