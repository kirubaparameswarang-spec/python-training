
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    classification_report
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = "task2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("CUSTOMER CHURN PREDICTION - TASK 2")
print("=" * 65)

n_samples = 5000

tenure = np.random.randint(1, 73, n_samples)

monthly_charges = np.random.uniform(
    25, 120, n_samples
).round(2)

total_charges = (
    tenure * monthly_charges
    + np.random.normal(0, 100, n_samples)
).clip(0).round(2)

support_calls = np.random.poisson(
    1.5, n_samples
).clip(0, 8)

senior_citizen = np.random.choice(
    [0, 1], n_samples, p=[0.84, 0.16]
)

dependents = np.random.choice(
    ["Yes", "No"], n_samples, p=[0.35, 0.65]
)

paperless_billing = np.random.choice(
    ["Yes", "No"], n_samples, p=[0.65, 0.35]
)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    n_samples,
    p=[0.55, 0.25, 0.20]
)

internet_service = np.random.choice(
    ["Fiber optic", "DSL", "No"],
    n_samples,
    p=[0.50, 0.35, 0.15]
)

payment_method = np.random.choice(
    ["Electronic check", "Credit card",
     "Bank transfer", "Mailed check"],
    n_samples
)

log_odds = (
    -1.8
    - 0.035 * tenure
    + 0.018 * (monthly_charges - 70)
    + 0.45 * support_calls
    + 0.55 * (contract == "Month-to-month")
    - 0.65 * (contract == "Two year")
    + 0.45 * (internet_service == "Fiber optic")
    + 0.35 * (payment_method == "Electronic check")
    + 0.35 * (paperless_billing == "Yes")
    + 0.35 * senior_citizen
    - 0.30 * (dependents == "Yes")
)

churn_probability = 1 / (1 + np.exp(-log_odds))

churn = np.random.binomial(
    1, churn_probability
)

df = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "support_calls": support_calls,
    "senior_citizen": senior_citizen,
    "dependents": dependents,
    "paperless_billing": paperless_billing,
    "contract": contract,
    "internet_service": internet_service,
    "payment_method": payment_method,
    "Churn": churn
})

print("\nDataset created successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

df.to_csv(
    os.path.join(OUTPUT_DIR, "customer_churn_dataset.csv"),
    index=False
)

X = df.drop("Churn", axis=1)
y = df["Churn"]

numeric_features = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "senior_citizen"
]

categorical_features = [
    "dependents",
    "paperless_billing",
    "contract",
    "internet_service",
    "payment_method"
]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

scale_pos_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "XGBoost": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
}

print("\nScale Pos Weight:", round(scale_pos_weight, 3))

baseline_scores = {}

print("\nBASELINE MODEL ROC-AUC SCORES")
print("-" * 45)

for name, model in models.items():

    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    baseline_scores[name] = auc

    print(f"{name}: {auc:.4f}")

best_two = sorted(
    baseline_scores,
    key=baseline_scores.get,
    reverse=True
)[:2]

print("\nBest two models selected for tuning:")

for name in best_two:
    print(f"{name}: {baseline_scores[name]:.4f}")

parameter_distributions = {

    "Logistic Regression": {
        "classifier__C": uniform(0.01, 10),
        "classifier__solver": ["lbfgs"],
        "classifier__class_weight": ["balanced"]
    },

    "Random Forest": {
        "classifier__n_estimators": randint(100, 401),
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": randint(2, 15),
        "classifier__min_samples_leaf": randint(1, 8),
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": ["balanced", "balanced_subsample"]
    },

    "XGBoost": {
        "classifier__n_estimators": randint(100, 401),
        "classifier__max_depth": randint(3, 8),
        "classifier__learning_rate": uniform(0.01, 0.19),
        "classifier__subsample": uniform(0.7, 0.3),
        "classifier__colsample_bytree": uniform(0.7, 0.3),
        "classifier__min_child_weight": randint(1, 8),
        "classifier__scale_pos_weight": [
            max(1, round(scale_pos_weight * 0.8, 2)),
            round(scale_pos_weight, 2),
            round(scale_pos_weight * 1.2, 2)
        ]
    }
}

tuned_models = {}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

for name in best_two:

    print("\nTuning:", name)

    search = RandomizedSearchCV(
        estimator=models[name],
        param_distributions=parameter_distributions[name],
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    tuned_models[name] = search.best_estimator_

    print("Best CV ROC-AUC:", round(search.best_score_, 4))
    print("Best Parameters:")
    print(search.best_params_)

final_models = {}

for name in models:

    if name in tuned_models:
        final_models[name] = tuned_models[name]
    else:
        final_models[name] = models[name]

    final_models[name].fit(X_train, y_train)

def find_best_threshold(y_true, probabilities,
                        minimum_recall=0.75):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    # Last precision and recall values have no matching threshold
    valid_precision = precision[:-1]
    valid_recall = recall[:-1]

    eligible = np.where(
        valid_recall >= minimum_recall
    )[0]

    if len(eligible) == 0:
        return 0.0

    # Highest precision while maintaining required recall
    best_index = eligible[
        np.argmax(valid_precision[eligible])
    ]

    return thresholds[best_index]

comparison_results = []
test_probabilities = {}
selected_thresholds = {}

for name, model in final_models.items():

    val_prob = model.predict_proba(X_val)[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_prob,
        minimum_recall=0.75
    )

    test_prob = model.predict_proba(X_test)[:, 1]

    test_pred = (
        test_prob >= threshold
    ).astype(int)

    auc = roc_auc_score(y_test, test_prob)

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    precision_value = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    recall_value = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    comparison_results.append({
        "Model": name,
        "ROC-AUC": auc,
        "Threshold": threshold,
        "F1 Score": f1,
        "Precision": precision_value,
        "Recall": recall_value
    })

    test_probabilities[name] = test_prob
    selected_thresholds[name] = threshold


comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print("\nFINAL MODEL COMPARISON")
print("=" * 85)
print(comparison_df.round(4).to_string(index=False))

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

plt.figure(figsize=(9, 7))

for name, probabilities in test_probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=300
)

plt.show()

best_model_name = comparison_df.iloc[0]["Model"]

best_model = final_models[best_model_name]

best_threshold = selected_thresholds[best_model_name]

best_probabilities = test_probabilities[best_model_name]

best_predictions = (
    best_probabilities >= best_threshold
).astype(int)

print("\nBEST MODEL:", best_model_name)

print("Selected Business Threshold:",
      round(best_threshold, 4))

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

classifier = best_model.named_steps["classifier"]

feature_names = (
    best_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

if hasattr(classifier, "feature_importances_"):

    importances = classifier.feature_importances_

elif hasattr(classifier, "coef_"):

    importances = np.abs(
        classifier.coef_[0]
    )

else:
    raise ValueError(
        "Feature importance is not available for this model."
    )


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_5 = importance_df.head(5).copy()

top_5["Feature"] = (
    top_5["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("_", " ", regex=False)
)

print("\nTOP 5 IMPORTANT FEATURES")
print(top_5.to_string(index=False))

plt.figure(figsize=(10, 6))

plt.barh(
    top_5["Feature"],
    top_5["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title(f"Top 5 Feature Importances - {best_model_name}")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_importance.png"),
    dpi=300
)

plt.show()

business_explanations = {
    "tenure": (
        "Customer tenure: Newer customers may be more likely "
        "to leave because their relationship with the company "
        "is not yet established."
    ),

    "monthly charges": (
        "Monthly charges: Higher bills may make customers "
        "consider cheaper competitors."
    ),

    "total charges": (
        "Total charges: Reflects the customer's accumulated "
        "spending and relationship with the company."
    ),

    "support calls": (
        "Support calls: Frequent support requests may indicate "
        "unresolved problems or customer dissatisfaction."
    ),

    "senior citizen": (
        "Senior citizen: Helps identify customer segments "
        "that may need different service or support."
    ),

    "month-to-month": (
        "Month-to-month contract: Customers without long-term "
        "commitments can leave more easily."
    ),

    "one year": (
        "One-year contract: Contract commitment can influence "
        "customer retention."
    ),

    "two year": (
        "Two-year contract: Longer commitments may reduce "
        "the likelihood of immediate churn."
    ),

    "fiber optic": (
        "Fiber optic internet: Service quality, pricing and "
        "customer expectations may influence churn."
    ),

    "electronic check": (
        "Electronic check: Payment behavior may identify "
        "customer groups with different churn patterns."
    ),

    "paperless billing": (
        "Paperless billing: Digital billing preferences may "
        "help identify different customer segments."
    ),

    "dependents": (
        "Dependents: Household characteristics may influence "
        "service needs and customer retention."
    ),

    "credit card": (
        "Credit card: Payment method may reveal differences "
        "in customer behavior."
    ),

    "bank transfer": (
        "Bank transfer: Payment preferences can help identify "
        "customer segments."
    ),

    "mailed check": (
        "Mailed check: Payment preferences may be associated "
        "with different customer behaviors."
    )
}

print("\nBUSINESS INTERPRETATION - TOP 5 FEATURES")
print("=" * 65)

for _, row in top_5.iterrows():

    feature = row["Feature"].lower()

    explanation = next(
        (
            value
            for key, value in business_explanations.items()
            if key in feature
        ),
        "This feature contributes to the model's churn prediction."
    )

    print(f"\nFeature: {row['Feature']}")
    print(f"Importance: {row['Importance']:.4f}")
    print("Business Explanation:", explanation)

final_predictions_df = pd.DataFrame({
    "Actual_Churn": y_test.values,
    "Churn_Probability": best_probabilities,
    "Predicted_Churn": best_predictions
})

final_predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_predictions.csv"),
    index=False
)
print("\n" + "=" * 65)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nBest Model:", best_model_name)

print("ROC-AUC:",
      round(roc_auc_score(y_test, best_probabilities), 4))

print("Business Threshold:",
      round(best_threshold, 4))

print("F1 Score:",
      round(f1_score(y_test, best_predictions), 4))

print("Churn Recall:",
      round(recall_score(y_test, best_predictions), 4))

print("\nGenerated Files:")

for filename in os.listdir(OUTPUT_DIR):
    print("-", filename)

print("\nAll outputs saved inside:", OUTPUT_DIR)
