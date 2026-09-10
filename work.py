import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
print("Probiotic Data Analysis Project")

data = {
    "Strain": [
        "Lactobacillus rhamnosus GG",
        "Lactobacillus acidophilus",
        "Bifidobacterium lactis",
        "Saccharomyces boulardii",
        "Lactobacillus plantarum"
    ],
    "Dose_CFU": [
        1e10,
        1e10,
        1e9,
        5e9,
        1e10
    ],
    "Duration_Weeks": [
        8,
        6,
        8,
        4,
        12
    ],
    "Outcome": [
        "Improved",
        "No significant change",
        "Improved",
        "Improved",
        "Improved"
    ]
}

df = pd.DataFrame(data)
# Numerical representation of outcome
df["Improvement_Rate"] = df["Outcome"].map({
    "Improved": 100,
    "No significant change": 0
})

print("\nOverall improvement statistics:")
print("Mean improvement rate:", df["Improvement_Rate"].mean())
print("Standard deviation:", df["Improvement_Rate"].std())
print(df)
print("\nData information:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())
plt.figure(figsize=(10, 5))

plt.bar(df["Strain"], df["Dose_CFU"])

plt.title("Probiotic Dose by Strain")
plt.xlabel("Probiotic Strain")
plt.ylabel("Dose (CFU)")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/dose_vs_improvement.png")
plt.show()
print("\nOutcome counts")
print(df["Outcome"].value_counts())
# Plot 2: Outcome Distribution

plt.figure(figsize=(7, 5))

df["Outcome"].value_counts().plot(kind="bar")

plt.title("Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Number of Studies")

plt.xticks(rotation=0)
plt.tight_layout()

plt.show()
# Plot 3: Dose vs Duration

plt.figure(figsize=(8, 5))

for outcome in df["Outcome"].unique():
    subset = df[df["Outcome"] == outcome]
    plt.scatter(
        subset["Duration_Weeks"],
        subset["Dose_CFU"],
        label=outcome,
        s=100
    )

plt.title("Probiotic Dose vs Study Duration")
plt.xlabel("Duration (Weeks)")
plt.ylabel("Dose (CFU)")

plt.legend()
plt.tight_layout()

plt.savefig("results/dose_vs_improvement.png")
plt.show()



print("\nAverage dose by outcome:")
print(df.groupby("Outcome")["Dose_CFU"].mean())
print("\nAverage duration by outcome:")
print(df.groupby("Outcome")["Duration_Weeks"].mean())
# Outcome by probiotic strain

outcome_by_strain = pd.crosstab(df["Strain"], df["Outcome"])

outcome_by_strain.plot(kind="bar", figsize=(10, 6))

plt.title("Outcome by Probiotic Strain")
plt.xlabel("Probiotic Strain")
plt.ylabel("Number of Studies")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Outcome")

plt.tight_layout()
plt.savefig("results/correlation_matrix.png")
plt.show()
print("\nDose and duration by outcome:")

analysis = df.groupby("Outcome")[["Dose_CFU", "Duration_Weeks"]].mean()

print(analysis)
print("\nOutcome by strain:")

print("\nOutcome by strain:")
print(pd.crosstab(df["Strain"], df["Outcome"]))
plt.figure(figsize=(8, 5))

dose_by_outcome = df.groupby("Outcome")["Dose_CFU"].mean()

dose_by_outcome.plot(kind="bar")

plt.title("Average Probiotic Dose by Outcome")
plt.xlabel("Outcome")
plt.ylabel("Average Dose (CFU)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("results/average_improvement.png")
plt.show()
print("\nImprovement rate by strain:")

improvement_rate = (
    df.groupby("Strain")["Outcome"]
    .apply(lambda x: (x == "Improved").mean() * 100)
)

print(improvement_rate)
improvement_rate.plot(kind="bar", figsize=(10, 6))

plt.title("Improvement Rate by Probiotic Strain")
plt.xlabel("Probiotic Strain")
plt.ylabel("Improvement Rate (%)")
plt.xticks(rotation=45)
plt.ylim(0, 110)

plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))

for outcome in df["Outcome"].unique():
    subset = df[df["Outcome"] == outcome]
    plt.scatter(
        subset["Dose_CFU"],
        subset["Duration_Weeks"],
        label=outcome,
        s=100
    )

plt.title("Probiotic Dose and Duration by Outcome")
plt.xlabel("Dose (CFU)")
plt.ylabel("Duration (Weeks)")
plt.legend()
plt.tight_layout()
plt.show()
print("\nCorrelation between dose and duration:")

correlation = df[["Dose_CFU", "Duration_Weeks"]].corr()

print(correlation)
from scipy.stats import chi2_contingency

print("\nChi-square test: Strain vs Outcome")

contingency_table = pd.crosstab(
    df["Strain"],
    df["Outcome"]
)

print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
# ==========================================
# Dose vs Improvement Rate
# ==========================================

print("\nImprovement rate by dose:")

dose_improvement = (
    df.groupby("Dose_CFU")["Outcome"]
    .apply(lambda x: (x == "Improved").mean() * 100)
)

print(dose_improvement)

# Plot
plt.figure(figsize=(8, 5))

plt.bar(
    dose_improvement.index.astype(str),
    dose_improvement.values
)

plt.xlabel("Probiotic Dose (CFU)")
plt.ylabel("Improvement Rate (%)")
plt.title("Improvement Rate by Probiotic Dose")

plt.ylim(0, 100)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# ==========================================
# Duration vs Improvement Rate
# ==========================================

print("\nImprovement rate by duration:")

duration_improvement = (
    df.groupby("Duration_Weeks")["Outcome"]
    .apply(lambda x: (x == "Improved").mean() * 100)
)

print(duration_improvement)

# Plot
plt.figure(figsize=(8, 5))

plt.bar(
    duration_improvement.index.astype(str),
    duration_improvement.values
)

plt.xlabel("Duration (Weeks)")
plt.ylabel("Improvement Rate (%)")
plt.title("Improvement Rate by Treatment Duration")

plt.ylim(0, 100)

plt.tight_layout()
plt.show()
# Statistical analysis: Chi-square test

from scipy.stats import chi2_contingency

print("\nStatistical analysis:")

# Strain vs Outcome
table_strain = pd.crosstab(df["Strain"], df["Outcome"])
chi2, p, dof, expected = chi2_contingency(table_strain)

print("\nStrain vs Outcome")
print("Chi-square:", chi2)
print("p-value:", p)

if p < 0.05:
    print("Significant association")
else:
    print("No significant association")


# Dose vs Outcome
table_dose = pd.crosstab(df["Dose_CFU"], df["Outcome"])
chi2, p, dof, expected = chi2_contingency(table_dose)

print("\nDose vs Outcome")
print("Chi-square:", chi2)
print("p-value:", p)

if p < 0.05:
    print("Significant association")
else:
    print("No significant association")


# Duration vs Outcome
table_duration = pd.crosstab(df["Duration_Weeks"], df["Outcome"])
chi2, p, dof, expected = chi2_contingency(table_duration)

print("\nDuration vs Outcome")
print("Chi-square:", chi2)
print("p-value:", p)

if p < 0.05:
    print("Significant association")
else:
    print("No significant association")
    # ==========================================
# NEW DATASET - Probiotic Study
# ==========================================

new_data = {
    "Study_ID": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    ],

    "Strain": [
        "Lactobacillus rhamnosus GG",
        "Lactobacillus acidophilus",
        "Bifidobacterium lactis",
        "Saccharomyces boulardii",
        "Lactobacillus plantarum",
        "Lactobacillus rhamnosus GG",
        "Bifidobacterium lactis",
        "Lactobacillus plantarum",
        "Lactobacillus acidophilus",
        "Saccharomyces boulardii",
        "Lactobacillus rhamnosus GG",
        "Bifidobacterium lactis",
        "Lactobacillus plantarum",
        "Lactobacillus acidophilus",
        "Saccharomyces boulardii"
    ],

    "Dose_CFU": [
        1e9, 5e9, 1e10, 5e9, 1e10,
        1e9, 1e10, 5e9, 1e10, 1e9,
        5e9, 1e10, 1e9, 5e9, 1e10
    ],

    "Duration_Weeks": [
        4, 8, 12, 6, 8,
        12, 4, 8, 6, 12,
        8, 6, 12, 4, 8
    ],

    "Baseline_Score": [
        70, 65, 80, 75, 60,
        72, 68, 77, 63, 74,
        69, 81, 66, 73, 62
    ],

    "Followup_Score": [
        82, 71, 91, 81, 75,
        88, 73, 86, 70, 84,
        80, 89, 78, 79, 76
    ]
}

new_df = pd.DataFrame(new_data)

print("\n==========================================")
print("NEW PROBIOTIC DATASET")
print("==========================================")

print(new_df)

print("\nDataset information:")
print(new_df.info())

print("\nBasic statistics:")
print(new_df.describe())
# ==========================================
# Calculate Improvement
# ==========================================

new_df["Improvement"] = (
    new_df["Followup_Score"] - new_df["Baseline_Score"]
)

new_df["Improvement_Percent"] = (
    new_df["Improvement"] / new_df["Baseline_Score"]
) * 100

print("\n==========================================")
print("IMPROVEMENT ANALYSIS")
print("==========================================")

print(
    new_df[
        [
            "Study_ID",
            "Strain",
            "Baseline_Score",
            "Followup_Score",
            "Improvement",
            "Improvement_Percent"
        ]
    ]
)

print("\nAverage improvement:")
print(new_df["Improvement"].mean())

print("\nAverage improvement percentage:")
print(new_df["Improvement_Percent"].mean())
print("REAL COLUMNS:")
print(df.columns.tolist())
print("REAL COLUMNS:")
print(df.columns.tolist())

print("\nAverage Improvement Rate:")
print(df["Improvement_Rate"].mean())

print("\nStrain Analysis:")

strain_analysis = df.groupby("Strain")["Improvement_Rate"].agg(
    ["mean", "min", "max", "count"]
)

print(strain_analysis)







print("Columns:", df.columns.tolist())
strain_analysis = df.groupby("Strain")["Improvement_Rate"].agg(
    ["mean", "count", "std", "min", "max"]
)

print("\nStrain Analysis:")
print(strain_analysis)




print(strain_analysis)

print("\nAverage improvement percentage by strain:")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
strain_percent = df.groupby("Strain")["Improvement_Rate"].mean()



strain_analysis["mean"].sort_values().plot(kind="bar")

plt.title("Average Improvement by Probiotic Strain")
plt.xlabel("Probiotic Strain")
plt.ylabel("Average Improvement")

df["Improvement_Rate"] = pd.to_numeric(df["Improvement_Rate"], errors="coerce")
print("\nAverage improvement by strain:")

strain_analysis = df.groupby("Strain")["Improvement_Rate"].agg(
    ["mean", "min", "max", "count"]
)

print(strain_analysis)
plt.figure(figsize=(10, 5))

strain_analysis["mean"].sort_values().plot(kind="bar")

plt.title("Average Improvement by Probiotic Strain")
plt.xlabel("Probiotic Strain")
plt.ylabel("Average Improvement")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()
print("\nCOLUMN NAMED:")
print(df.columns.tolist())
print("\nCOLUMN NAMES:")
print(df.columns.tolist())
print(df.columns.tolist())
print("\nFINAL DATA CHECK:")
print(df)

print("\nSUMMARY:")
print(df.describe(include="all"))
print("\nFULL DATASET:")
print(df.to_string(index=False))

print("\nCOLUMN NAMES:")
print(df.columns.tolist())
print("\nOUTCOME COUNTS:")
print(df["Outcome"].value_counts())

print("\nAVERAGE IMPROVEMENT:")
print(df["Improvement_Rate"].mean())

print("\nCORRELATION MATRIX:")
print(df[["Dose_CFU", "Duration_Weeks", "Improvement_Rate"]].corr())
# ==========================================
# Relationship between dose and improvement
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(df["Dose_CFU"], df["Improvement_Rate"])

plt.xlabel("Dose (CFU)")
plt.ylabel("Improvement Rate (%)")
plt.title("Dose vs Improvement Rate")

plt.tight_layout()
plt.show()


# ==========================================
# Relationship between duration and improvement
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(df["Duration_Weeks"], df["Improvement_Rate"])

plt.xlabel("Duration (Weeks)")
plt.ylabel("Improvement Rate (%)")
plt.title("Duration vs Improvement Rate")

plt.tight_layout()
plt.show()
print("\n" + "="*50)
print("MULTIVARIABLE ANALYSIS")
print("="*50)

# Convert categorical variables to numerical form
df["Outcome_Binary"] = df["Outcome"].map({
    "Improved": 1,
    "No significant change": 0
})

# Show the prepared data
print("\nPrepared data:")
print(df[[
    "Strain",
    "Dose_CFU",
    "Duration_Weeks",
    "Outcome_Binary",
    "Improvement_Rate"
]])

# Correlation with improvement
print("\nCorrelation with Improvement Rate:")

correlations = df[
    ["Dose_CFU", "Duration_Weeks", "Outcome_Binary", "Improvement_Rate"]
].corr()["Improvement_Rate"].sort_values(ascending=False)

print(correlations)
print("\n" + "="*50)
print("STATISTICAL SIGNIFICANCE")
print("="*50)

from scipy.stats import pearsonr, spearmanr

variables = [
    "Dose_CFU",
    "Duration_Weeks"
]

for var in variables:
    pearson_r, pearson_p = pearsonr(
        df[var],
        df["Improvement_Rate"]
    )

    spearman_r, spearman_p = spearmanr(
        df[var],
        df["Improvement_Rate"]
    )

    print(f"\n{var}")
    print(f"Pearson r = {pearson_r:.3f}")
    print(f"Pearson p-value = {pearson_p:.4f}")
    print(f"Spearman rho = {spearman_r:.3f}")
    print(f"Spearman p-value = {spearman_p:.4f}")
    print("\n" + "="*50)
print("STRAIN EFFECT ANALYSIS")
print("="*50)

strain_analysis = df.groupby("Strain")["Improvement_Rate"].agg(
    ["mean", "count"]
)

print(strain_analysis)
print("\n" + "="*50)
print("LINEAR REGRESSION MODEL")
print("="*50)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

X = df[[
    "Dose_CFU",
    "Duration_Weeks"
]]

y = df["Improvement_Rate"]

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

print("R2 Score:", r2_score(y, predictions))

print("MAE:", mean_absolute_error(y, predictions))

print("\nModel coefficients:")

print("Dose coefficient:", model.coef_[0])
print("Duration coefficient:", model.coef_[1])

print("Intercept:", model.intercept_)
import os

os.makedirs("results", exist_ok=True)

