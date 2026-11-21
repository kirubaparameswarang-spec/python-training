
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


majority_class = np.bincount(y_train).argmax()
y_pred = np.full(len(y_test), majority_class)


accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(
    f"Majority-class model: accuracy {accuracy:.2f}, "
    f"recall {recall:.2f}, F1 {f1:.2f}"
)
