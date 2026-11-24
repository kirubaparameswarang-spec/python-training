\# Task 4 – Evaluation and Tuning



\## Objective



Evaluate classification and regression models, compare their performance, tune hyperparameters, select the best classification model, and perform final evaluation on an unseen test dataset.



\## Dataset



\- File: `data/loan\_preprocessing.csv`

\- Total records: 600

\- Classification target: `loan\_status`

\- Regression target: `loan\_amount`

\- Test set: 20% of the dataset

\- Random state: 42



The dataset was split into:

\- Training: 60% (360 records)

\- Validation: 20% (120 records)

\- Testing: 20% (120 records)



The test set was kept separate during model selection and tuning.



\## Classification Evaluation



Models evaluated:

\- Logistic Regression

\- Decision Tree

\- Random Forest

\- XGBoost



Metrics:

\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- ROC-AUC

\- Confusion Matrix



ROC curves were plotted for Logistic Regression, Random Forest, and XGBoost.



A classification threshold was selected using validation data to achieve at least 75% recall.



\## Hyperparameter Tuning



\### Random Forest



\- Method: GridSearchCV

\- Parameters: n\_estimators, max\_depth, min\_samples\_split

\- Combinations: 12

\- Cross-validation: 5 folds

\- Best CV ROC-AUC: 0.6501



\### XGBoost



\- Method: RandomizedSearchCV

\- Iterations: 15

\- Cross-validation: 5 folds

\- Best CV ROC-AUC: 0.6626



\## Regression Evaluation



Models:

\- Linear Regression

\- Ridge Regression

\- Lasso Regression

\- Mean Baseline



Metrics:

\- MAE

\- RMSE

\- R²



Best regression model based on validation RMSE: Ridge Regression.



Validation RMSE: 86896.4894



\## Final Classification Model



Selected model: Logistic Regression



Selection criterion: Highest validation ROC-AUC.



Validation ROC-AUC: 0.6919



Selected probability threshold: 0.2904



\### Final Test Results



| Metric | Score |

|---|---:|

| Accuracy | 0.5583 |

| Precision | 0.4091 |

| Recall | 0.6585 |

| F1 Score | 0.5047 |

| ROC-AUC | 0.6505 |



The final model was refitted using the training and validation datasets and evaluated once on the held-out test set.



The selected threshold achieved the required recall on validation data, but test recall was 65.85%. This difference demonstrates that validation performance does not guarantee identical performance on unseen data.



\## Output Files



\- `classification\_metrics.csv` – Classification evaluation metrics

\- `classification\_models.joblib` – Baseline classification models

\- `data\_splits.npz` – Saved dataset split indices

\- `roc\_curves.png` – ROC curve comparison

\- `roc\_results.csv` – ROC-AUC results

\- `threshold\_results.csv` – Threshold selection results

\- `random\_forest\_tuning.csv` – Random Forest GridSearchCV results

\- `xgboost\_tuning.csv` – XGBoost RandomizedSearchCV results

\- `tuning\_summary.csv` – Hyperparameter tuning summary

\- `tuned\_classification\_models.joblib` – Tuned classification models

\- `regression\_metrics.csv` – Regression evaluation metrics

\- `regression\_models.joblib` – Trained regression models

\- `validation\_model\_comparison.csv` – Validation model comparison

\- `final\_classification\_model.joblib` – Selected final classifier

\- `final\_test\_results.csv` – Final test evaluation



\## Conclusion



Logistic Regression achieved the highest validation ROC-AUC among the evaluated classification candidates and was selected for final testing.



Ridge Regression achieved the lowest validation RMSE among the regression models, although all regression models had negative R² values.



The assessment demonstrates classification evaluation, ROC analysis, threshold selection, cross-validated hyperparameter tuning, regression comparison, and final evaluation on an independent test set.

