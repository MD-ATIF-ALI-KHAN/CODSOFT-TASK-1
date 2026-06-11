#  Titanic Survival Prediction — Complete ML Pipeline
#  Works standalone or imported by a Streamlit frontend

# ── Imports ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

#  STEP 1 — Load Data
print("=" * 55)
print("  STEP 1 — Loading Data")
print("=" * 55)

df = pd.read_csv("train.csv")

print(f"Shape       : {df.shape}")
print(f"Columns     : {df.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nData types + null counts:\n{df.info()}")
print(f"\nStatistical summary:\n{df.describe()}")


#  STEP 2 — Exploratory Data Analysis (EDA)
print("\n" + "=" * 55)
print("  STEP 2 — Exploratory Data Analysis")
print("=" * 55)

# Survival counts and proportions
print("Survival counts:")
print(df["Survived"].value_counts())
print("\nSurvival proportions:")
print(df["Survived"].value_counts(normalize=True).round(2))

# Missing values
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nMissing %:")
print((df.isnull().sum() / len(df) * 100).round(1))

# ── EDA Plots ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Titanic — Exploratory Data Analysis", fontsize=14)

# Plot 1: Survival by Sex
sns.barplot(x="Sex", y="Survived", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Survival Rate by Gender")
axes[0, 0].set_ylabel("Survival Rate")

# Plot 2: Survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Survival Rate by Class")

# Plot 3: Age distribution by survival
df[df["Survived"] == 1]["Age"].hist(alpha=0.6, label="Survived",
                                     bins=25, ax=axes[1, 0])
df[df["Survived"] == 0]["Age"].hist(alpha=0.6, label="Died",
                                     bins=25, ax=axes[1, 0])
axes[1, 0].legend()
axes[1, 0].set_title("Age Distribution by Survival")
axes[1, 0].set_xlabel("Age")

# Plot 4: Missing values heatmap
sns.heatmap(df.isnull(), yticklabels=False, cbar=False,
            cmap="viridis", ax=axes[1, 1])
axes[1, 1].set_title("Missing Values (yellow = missing)")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA plots saved → eda_plots.png")


#  STEP 3 — Preprocessing
print("\n" + "=" * 55)
print("  STEP 3 — Preprocessing")
print("=" * 55)

# 1. Drop Cabin (77% missing)
df.drop(columns=["Cabin"], inplace=True)

# 2. Fill Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# 3. Fill Embarked with mode (only 2 missing)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# 4. Encode Sex: male=1, female=0
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

# 5. One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# 6. Fix boolean columns → integers
df["Embarked_Q"] = df["Embarked_Q"].astype(int)
df["Embarked_S"] = df["Embarked_S"].astype(int)

# 7. Drop columns with no predictive value
df.drop(columns=["PassengerId", "Ticket"], inplace=True)

print(f"Shape after preprocessing : {df.shape}")
print(f"Missing values            :\n{df.isnull().sum()}")
print(f"Columns remaining         : {df.columns.tolist()}")


#  STEP 4 — Feature Engineering
print("\n" + "=" * 55)
print("  STEP 4 — Feature Engineering")
print("=" * 55)

# 1. Extract Title from Name
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
print("Raw title counts:")
print(df["Title"].value_counts())

# 2. Group rare titles
df["Title"] = df["Title"].replace(
    ["Dr", "Rev", "Major", "Col", "Capt",
     "Sir", "Lady", "Countess", "Don", "Jonkheer"], "Rare"
)
df["Title"] = df["Title"].replace({
    "Mlle": "Miss",
    "Ms"  : "Miss",
    "Mme" : "Mrs"
})
print("\nCleaned title counts:")
print(df["Title"].value_counts())

# 3. Encode Title
df["Title"] = LabelEncoder().fit_transform(df["Title"])

# 4. Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 5. Is alone?
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# 6. Age × Pclass interaction
df["Age_Class"] = df["Age"] * df["Pclass"]

# 7. Drop Name (no longer needed)
df.drop(columns=["Name"], inplace=True)

print(f"\nShape after feature engineering : {df.shape}")
print(f"Final columns                   : {df.columns.tolist()}")
print(f"\nSample (5 rows):\n{df.head()}")


#  STEP 5 — Train / Test Split + Scale
print("\n" + "=" * 55)
print("  STEP 5 — Train/Test Split + Scaling")
print("=" * 55)

X = df.drop(columns=["Survived"])
y = df["Survived"]

# 80/20 split, stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")

# Scale — fit only on train, transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Scaling complete.")



#  STEP 6 — Train Models
print("\n" + "=" * 55)
print("  STEP 6 — Training Models")
print("=" * 55)

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
print("Logistic Regression trained.")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
print("Random Forest trained.")


#  STEP 7 — Evaluate
print("\n" + "=" * 55)
print("  STEP 7 — Evaluation")
print("=" * 55)

print("─── Accuracy ───────────────────────────────")
print(f"Logistic Regression : {accuracy_score(y_test, lr_preds):.4f}")
print(f"Random Forest       : {accuracy_score(y_test, rf_preds):.4f}")

print("\n─── Random Forest Classification Report ────")
print(classification_report(y_test, rf_preds,
      target_names=["Died", "Survived"]))

# ── Evaluation Plots ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Random Forest — Evaluation", fontsize=14)

# Confusion matrix
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"],
            yticklabels=["Died", "Survived"],
            ax=axes[0])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")

# Feature importance
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

importances.plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title("Feature Importances")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("evaluation_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Evaluation plots saved → evaluation_plots.png")

print("\n" + "=" * 55)
print("  PIPELINE COMPLETE")
print(f"  Best model  : Random Forest")
print(f"  Accuracy    : {accuracy_score(y_test, rf_preds):.4f}")
print(f"  Top feature : {importances.idxmax()}")
print("=" * 55)