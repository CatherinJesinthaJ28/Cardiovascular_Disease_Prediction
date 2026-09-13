import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("Output/cardio_cleaned.csv")

# ==========================================
# 1. CARDIOVASCULAR DISEASE DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="cardio")

plt.title("Cardiovascular Disease Distribution")
plt.xlabel("Cardiovascular Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("Output/Plots/cardio_distribution.png", dpi=300)

plt.show()
# ==========================================
# 2. AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="age_years", bins=20, kde=True)

plt.title("Age Distribution of Patients")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("Output/Plots/age_distribution.png", dpi=300)

plt.show()
# ==========================================
# 3. AGE VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="age_years", hue="cardio")

plt.title("Cardiovascular Disease by Age")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Patients")
plt.legend(title="Cardio", labels=["No Disease", "Disease"])

plt.tight_layout()

plt.savefig("Output/Plots/age_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 4. GENDER VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="gender", hue="cardio")

plt.title("Cardiovascular Disease by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.legend(title="Cardio", labels=["No Disease", "Disease"])

plt.tight_layout()

plt.savefig("Output/Plots/gender_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 5. CHOLESTEROL VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="cholesterol", hue="cardio")

plt.title("Cardiovascular Disease by Cholesterol Level")
plt.xlabel("Cholesterol Level")
plt.ylabel("Number of Patients")
plt.legend(title="Cardio", labels=["No Disease", "Disease"])

plt.tight_layout()

plt.savefig("Output/Plots/cholesterol_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 6. GLUCOSE VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="gluc", hue="cardio")

plt.title("Cardiovascular Disease by Glucose Level")
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")
plt.legend(title="Cardio", labels=["No Disease", "Disease"])

plt.tight_layout()

plt.savefig("Output/Plots/glucose_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 7. SYSTOLIC BLOOD PRESSURE VS CARDIO
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="cardio", y="ap_hi")

plt.title("Systolic Blood Pressure by Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease (0 = No, 1 = Yes)")
plt.ylabel("Systolic Blood Pressure")

plt.tight_layout()

plt.savefig("Output/Plots/systolic_bp_vs_cardio.png", dpi=300)

plt.show()


# ==========================================
# 8. DIASTOLIC BLOOD PRESSURE VS CARDIO
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="cardio", y="ap_lo")

plt.title("Diastolic Blood Pressure by Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease (0 = No, 1 = Yes)")
plt.ylabel("Diastolic Blood Pressure")

plt.tight_layout()

plt.savefig("Output/Plots/diastolic_bp_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 9. WEIGHT VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="cardio", y="weight")

plt.title("Weight by Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease (0 = No, 1 = Yes)")
plt.ylabel("Weight (kg)")

plt.tight_layout()

plt.savefig("Output/Plots/weight_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 10. CALCULATE BMI
# ==========================================

df["height_m"] = df["height"] / 100

df["bmi"] = df["weight"] / (df["height_m"] ** 2)

print("\n========== BMI SUMMARY ==========")
print(df["bmi"].describe())
# ==========================================
# 11. BMI VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="cardio", y="bmi")

plt.title("BMI by Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease (0 = No, 1 = Yes)")
plt.ylabel("BMI")

plt.tight_layout()

plt.savefig("Output/Plots/bmi_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 12. PHYSICAL ACTIVITY VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="active", hue="cardio")

plt.title("Cardiovascular Disease by Physical Activity")
plt.xlabel("Physical Activity (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.legend(title="Cardio", labels=["No Disease", "Disease"])

plt.tight_layout()

plt.savefig("Output/Plots/active_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 13. SMOKING VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="smoke", hue="cardio")

plt.title("Cardiovascular Disease by Smoking Status")
plt.xlabel("Smoking (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.legend(title="Cardio", labels=["No Disease", "Disease"])

plt.tight_layout()

plt.savefig("Output/Plots/smoking_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 14. ALCOHOL VS CARDIOVASCULAR DISEASE
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="alco", hue="cardio")

plt.title("Cardiovascular Disease by Alcohol Consumption")
plt.xlabel("Alcohol Consumption (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.legend(title="Cardio", labels=["No Disease", "Disease"])

plt.tight_layout()

plt.savefig("Output/Plots/alcohol_vs_cardio.png", dpi=300)

plt.show()
# ==========================================
# 15. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Cardiovascular Disease Dataset")

plt.tight_layout()

plt.savefig("Output/Plots/correlation_heatmap.png", dpi=300)

plt.show()