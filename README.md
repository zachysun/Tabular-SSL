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
|           npl           |              SVC, Normal pseudo label learning               |
|         ve-npl          |        VotingClassifier, Normal pseudo label learning        |
|          clpl           |              Curriculum learning, Pseudo label               |
|         ve-clpl         |     VotingClassifier, Curriculum learning, Pseudo label      |
|        cts-clpl         |  Cross-training strategy, Curriculum learning, Pseudo label  |
|       ve-cts-clpl       | VotingClassifier, Cross-training strategy, Curriculum learning, Pseudo label |
|      ve-ccts-clpl       | VotingClassifier, Cross-training strategy, Curriculum learning, Pseudo label |
|     ve-cts-bi-clpl      | VotingClassifier, Cross-training strategy, Curriculum learning, Pseudo label |
|       dy-ve-clpl        | Dynamic confidence level change step size, VotingClassifier, Curriculum learning, Pseudo label |
|      dy-ccts-clpl       | Dynamic confidence level change step size, Cross-training strategy, Curriculum learning, Pseudo label |
|     dy-ve-cts-clpl      | Dynamic confidence level change step size, VotingClassifier, Cross-training strategy, Curriculum learning, Pseudo label |
|     dy-ve-ccts-clpl     | Dynamic confidence level change step size, VotingClassifier, Cross-training strategy, Curriculum learning, Pseudo label |



---

#### Notes:

- Data link for test [PIMA](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- "**ccts**" means "labeled samples are trained together"; "**cts**" means "labeled samples are divided into two parts and trained separately."

---

#### Results:

|      | Accuracy Score | Precision Score | Recall Score | f1 Score | AUC Score | Confusion Matrix                      | Algos              |
| :--- | :------------- | :-------------- | :----------- | :------- | :-------- | :------------------------------------ | :----------------- |
| 0    | 0.864583       | 0.805970        | 0.805970     | 0.805970 | 0.956657  | \[\[112.0, 13.0\], \[13.0, 54.0\]\]   | ve                 |
| 1    | 0.864583       | 0.797101        | 0.820896     | 0.808824 | 0.954627  | \[\[111.0, 14.0\], \[12.0, 55.0\]\]   | RandomForest       |
| 2    | 0.859375       | 0.794118        | 0.805970     | 0.800000 | 0.954507  | \[\[111.0, 14.0\], \[13.0, 54.0\]\]   | LightGBM           |
| 3    | 0.875000       | 0.820896        | 0.820896     | 0.820896 | 0.953194  | \[\[113.0, 12.0\], \[12.0, 55.0\]\]   | XGBoost            |
| 4    | 0.833333       | 0.701149        | 0.910448     | 0.792208 | 0.923701  | \[\[99.0, 26.0\], \[6.0, 61.0\]\]     | dyvecctsclpl       |
| 5    | 0.828125       | 0.697674        | 0.895522     | 0.784314 | 0.921672  | \[\[99.0, 26.0\], \[7.0, 60.0\]\]     | vectsbiclpl        |
| 6    | 0.822917       | 0.689655        | 0.895522     | 0.779221 | 0.917015  | \[\[98.0, 27.0\], \[7.0, 60.0\]\]     | ctsclpl            |
| 7    | 0.826875       | 0.695753        | 0.895522     | 0.783093 | 0.914716  | \[\[98.76, 26.24\], \[7.0, 60.0\]\]   | npl                |
| 8    | 0.833333       | 0.701149        | 0.910448     | 0.792208 | 0.914030  | \[\[99.0, 26.0\], \[6.0, 61.0\]\]     | veclpl             |
| 9    | 0.828125       | 0.697674        | 0.895522     | 0.784314 | 0.913552  | \[\[99.0, 26.0\], \[7.0, 60.0\]\]     | dycctsclpl         |
| 10   | 0.827708       | 0.696851        | 0.896119     | 0.784013 | 0.913344  | \[\[98.88, 26.12\], \[6.96, 60.04\]\] | venpl              |
| 11   | 0.828125       | 0.697674        | 0.895522     | 0.784314 | 0.910925  | \[\[99.0, 26.0\], \[7.0, 60.0\]\]     | vecctsclpl         |
| 12   | 0.828125       | 0.697674        | 0.895522     | 0.784314 | 0.909493  | \[\[99.0, 26.0\], \[7.0, 60.0\]\]     | dyvectsclpl        |
| 13   | 0.822917       | 0.689655        | 0.895522     | 0.779221 | 0.908657  | \[\[98.0, 27.0\], \[7.0, 60.0\]\]     | vectsclpl          |
| 14   | 0.822917       | 0.689655        | 0.895522     | 0.779221 | 0.905075  | \[\[98.0, 27.0\], \[7.0, 60.0\]\]     | clpl               |
| 15   | 0.822917       | 0.770492        | 0.701493     | 0.734375 | 0.902448  | \[\[111.0, 14.0\], \[20.0, 47.0\]\]   | SVC                |
| 16   | 0.822917       | 0.689655        | 0.895522     | 0.779221 | 0.898866  | \[\[98.0, 27.0\], \[7.0, 60.0\]\]     | dyveclpl           |
| 17   | 0.770833       | 0.725490        | 0.552239     | 0.627119 | 0.868060  | \[\[111.0, 14.0\], \[30.0, 37.0\]\]   | LogisticRegression |
| 18   | 0.781250       | 0.676056        | 0.716418     | 0.695652 | 0.846090  | \[\[102.0, 23.0\], \[19.0, 48.0\]\]   | KNeighbours        |

---

#### Reference:

- “Curriculum labeling: Revisiting pseudo-labeling for semi-supervised learning” AAAI, 2021 [Link](https://ojs.aaai.org/index.php/AAAI/article/view/16852) 
- "Pseudo-Labeled Auto-Curriculum Learning for Semi-Supervised Keypoint Localization" ICLR, 2022 [Link](https://arxiv.org/abs/2201.08613)



