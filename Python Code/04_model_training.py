import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("Output/cardio_cleaned.csv")

print("Dataset Shape:")
print(df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

# Calculate BMI
df["height_m"] = df["height"] / 100
df["bmi"] = df["weight"] / (df["height_m"] ** 2)

features = [
    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bmi"
]

X = df[features]
y = df["cardio"]


# ==========================================
# 3. DISPLAY FEATURES AND TARGET
# ==========================================

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("cardio")


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 6. TRAIN LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_scaled)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# ==========================================
# 9. DECISION TREE MODEL
# ==========================================

print("\n========== DECISION TREE ==========")

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=6
)

decision_tree.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nAccuracy:")
print(tree_accuracy)

print("\nAccuracy Percentage:")
print(f"{tree_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
# ==========================================
# 10. RANDOM FOREST MODEL
# ==========================================

print("\n========== RANDOM FOREST ==========")

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nAccuracy:")
print(rf_accuracy)

print("\nAccuracy Percentage:")
print(f"{rf_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
# ==========================================
# 11. MODEL COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========")

print(f"Logistic Regression : {accuracy * 100:.2f}%")
print(f"Decision Tree      : {tree_accuracy * 100:.2f}%")
print(f"Random Forest      : {rf_accuracy * 100:.2f}%")
# ==========================================
# 12. DECISION TREE TUNING
# ==========================================

print("\n========== DECISION TREE TUNING ==========")

depths = [3, 5, 7, 10, 15, 20]

for depth in depths:

    model_tuned = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model_tuned.fit(X_train, y_train)

    y_pred_tuned = model_tuned.predict(X_test)

    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

    print(
        f"Max Depth = {depth} --> "
        f"Accuracy = {tuned_accuracy * 100:.2f}%"
    )
# ============================================================
# 13. FINAL DECISION TREE MODEL
# ============================================================

print("\n========== FINAL MODEL ==========")

# Create the final Decision Tree model
final_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
final_model.fit(X_train, y_train)

# Make predictions
final_pred = final_model.predict(X_test)

# Calculate accuracy
final_accuracy = accuracy_score(y_test, final_pred)

print("Final Model: Decision Tree")
print("Max Depth: 5")
print(f"Final Accuracy: {final_accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, final_pred))

# Confusion Matrix
final_cm = confusion_matrix(y_test, final_pred)

print("\nConfusion Matrix:")
print(final_cm)


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

import pickle
import os

os.makedirs("Output", exist_ok=True)

with open("Output/final_decision_tree_model.pkl", "wb") as file:
    pickle.dump(final_model, file)

print("\nFinal model saved successfully!")


# ============================================================
# 15. FINAL CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title("Final Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "Output/final_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion matrix saved successfully!")