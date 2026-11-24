\# Task 3: Models — Supervised and Unsupervised Learning



\## Project Overview



This project applies supervised and unsupervised machine learning techniques to a loan dataset containing 600 records and 9 columns.



The project covers classification, regression, regularisation, feature importance, and K-Means clustering.



\## Dataset



File: `data/loan\_preprocessing.csv`



Important columns:

\- `annual\_income`

\- `loan\_amount`

\- `credit\_score`

\- `employment\_type`

\- `education\_level`

\- `property\_area`

\- `dependents`

\- `loan\_status`



Missing numeric values are handled using median imputation. Missing categorical values are handled using most-frequent imputation.



\## 1. Classification Models



Target: `loan\_status`



Four classifiers were trained using preprocessing pipelines:

\- Logistic Regression

\- Decision Tree

\- Random Forest

\- XGBoost



The data was split into 80% training and 20% validation using stratification and `random\_state=42`.



\### Validation Results



| Model | Validation Accuracy |

|---|---:|

| Logistic Regression | 0.7083 |

| Decision Tree | 0.6583 |

| Random Forest | 0.6583 |

| XGBoost | 0.6833 |



Random Forest OOB score: 0.6625.



Logistic Regression achieved the highest validation accuracy of 70.83%. Its recall for class 1 was 0.24, showing that accuracy alone does not fully describe model performance.



The project also compares Logistic Regression predicted probabilities with predicted classes and displays the five most important features from Random Forest and XGBoost.



Output: `outputs/classification\_results.csv`



\## 2. Regression Models



Target: `loan\_amount`



The following regression models were trained:

\- Linear Regression

\- Ridge Regression

\- Lasso Regression



\### Validation Results



| Model | R² Score | MAE |

|---|---:|---:|

| Linear Regression | -0.0146 | 59493.77 |

| Ridge Regression | -0.0145 | 59492.38 |

| Lasso Regression | -0.0073 | 59481.45 |



Lasso produced 8 zero coefficients out of 15 features, demonstrating feature selection through regularisation.



All three models had negative R² scores on the validation data, indicating weak predictive performance.



Outputs:

\- `outputs/regression\_results.csv`

\- `outputs/regression\_coefficients.csv`



\## 3. K-Means Clustering



Three numeric features were selected:

\- Annual income

\- Loan amount

\- Credit score



Missing values were imputed using the median, and the features were standardised before clustering.



K-Means was tested for k values from 2 to 8.



The highest silhouette score was 0.3206 at k = 4.



\### Cluster Summary



| Cluster | Customers | Average Income | Average Loan | Average Credit Score |

|---|---:|---:|---:|---:|

| 0 | 249 | 46476.48 | 158220.97 | 579.49 |

| 1 | 189 | 44039.24 | 154725.63 | 772.14 |

| 2 | 83 | 50241.82 | 336524.30 | 670.51 |

| 3 | 79 | 116771.46 | 154389.43 | 653.71 |



The four clusters show differences in income, loan amount, and credit score.



Outputs:

\- `outputs/elbow\_silhouette.png`

\- `outputs/cluster\_segment\_summary.csv`

\- `outputs/loan\_customer\_clusters.csv`



\## Project Structure



```text

Task - 3 Models, Supervised and Unsupervised/

│

├── data/

│   └── loan\_preprocessing.csv

│

├── notebooks/

│

├── outputs/

│   ├── classification\_results.csv

│   ├── regression\_results.csv

│   ├── regression\_coefficients.csv

│   ├── elbow\_silhouette.png

│   ├── cluster\_segment\_summary.csv

│   └── loan\_customer\_clusters.csv

│

├── src/

│   ├── 01\_classification\_models.py

│   ├── 02\_regression\_models.py

│   └── 03\_kmeans\_clustering.py

│

├── .gitignore

├── requirements.txt

└── README.md

```



\## How to Run



Install the dependencies:



```bash

pip install -r requirements.txt

```



Run the classification script:



```bash

python src/01\_classification\_models.py

```



Run the regression script:



```bash

python src/02\_regression\_models.py

```



Run the clustering script:



```bash

python src/03\_kmeans\_clustering.py

```



\## Concepts Covered



\- Logistic Regression

\- Linear Regression

\- Decision Trees

\- Random Forest and OOB score

\- XGBoost and early stopping

\- Ridge and Lasso regularisation

\- Feature importance

\- K-Means clustering

\- Elbow method

\- Silhouette score

\- Data preprocessing and pipelines

