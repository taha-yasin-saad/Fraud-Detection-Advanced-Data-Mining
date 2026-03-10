# ==========================================
# fraud_detection.py - CLEAN EXECUTABLE VERSION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest

import warnings
warnings.filterwarnings("ignore")


# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("creditcard.csv")

print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nClass distribution:")
print(df["Class"].value_counts())


# ==========================================
# Class Distribution Plot
# ==========================================
plt.figure(figsize=(5, 4))
df["Class"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xticks([0, 1], ["Legitimate (0)", "Fraud (1)"])
plt.ylabel("Count")
plt.savefig("class_distribution.png")
plt.show()


# ==========================================
# Statistical Summary
# ==========================================
print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# Amount Distribution
# ==========================================
plt.figure(figsize=(6, 4))
df[df["Class"] == 0]["Amount"].plot(kind="hist", bins=50)
plt.title("Amount Distribution - Legitimate")
plt.xlabel("Amount")
plt.savefig("amount_legitimate.png")
plt.show()

plt.figure(figsize=(6, 4))
df[df["Class"] == 1]["Amount"].plot(kind="hist", bins=50)
plt.title("Amount Distribution - Fraud")
plt.xlabel("Amount")
plt.savefig("amount_fraud.png")
plt.show()


# ==========================================
# Fraud Over Time
# ==========================================
plt.figure(figsize=(6, 4))
df[df["Class"] == 1]["Time"].plot(kind="hist", bins=50)
plt.title("Fraud Transactions Over Time")
plt.xlabel("Time (seconds)")
plt.savefig("fraud_over_time.png")
plt.show()


# ==========================================
# Missing Values
# ==========================================
print("\nMissing values per column:")
print(df.isnull().sum())


# ==========================================
# Preprocessing (Scaling + Train-Test)
# ==========================================
data = df.copy()

scaler = StandardScaler()
data[["Time", "Amount"]] = scaler.fit_transform(data[["Time", "Amount"]])

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split Completed.")
print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))


# ==========================================
# Logistic Regression
# ==========================================
log_reg = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    n_jobs=-1
)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

print("\n========== Logistic Regression ==========")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_proba_lr)
pr_auc_lr = auc(recall_lr, precision_lr)

print(f"ROC-AUC: {roc_auc_lr:.4f}")
print(f"PR-AUC: {pr_auc_lr:.4f}")


# ==========================================
# Random Forest (Main Model)
# ==========================================
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n========== Random Forest ==========")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

print(f"ROC-AUC: {roc_auc_rf:.4f}")
print(f"PR-AUC: {pr_auc_rf:.4f}")


# ==========================================
# ROC CURVE PLOT
# ==========================================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(6, 4))
plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={roc_auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={roc_auc_rf:.3f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve Comparison")
plt.savefig("roc_curve.png")
plt.show()


# ==========================================
# Precision-Recall Curve
# ==========================================
plt.figure(figsize=(6, 4))
plt.plot(recall_rf, precision_rf, label=f"RF PR-AUC={pr_auc_rf:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Random Forest)")
plt.legend()
plt.savefig("precision_recall_curve.png")
plt.show()


# ==========================================
# Isolation Forest (Unsupervised)
# ==========================================
iso = IsolationForest(
    n_estimators=200,
    contamination=float(y.mean()),
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train[y_train == 0])  # Fit on normal only

iso_pred = iso.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)

print("\n========== Isolation Forest (Anomaly Detection) ==========")
print("Confusion Matrix:")
print(confusion_matrix(y_test, iso_pred))

print("\nClassification Report:")
print(classification_report(y_test, iso_pred))


print("\n\n=== ALL TASKS COMPLETED SUCCESSFULLY ===")
print("All graphs saved to the current directory.")
