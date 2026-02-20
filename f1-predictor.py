import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

np.random.seed(42)

df = pd.read_csv("f1_data.csv")

FEATURES = [
    "grid_position",
    "quali_position",
    "driver_points",
    "avg_track_temp",
    "rainfall",
    "avg_lap_gap",
    "n_pitstops",
]

TARGET = "won"

df = df.dropna(subset=FEATURES)

X = df[FEATURES]
y = df[TARGET]

print("Feature preview:")
print(X.describe().round(2))

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model initialisation

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


print("Training complete")

# model evaluation

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("Model performance:")
print(f"ROC-AUC Score: {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["No Win", "Win"]))

# predicting 2026 Australian GP

australian_gp_2026 = pd.read_csv("australian_gp_2026.csv")

X_race = australian_gp_2026[FEATURES]
australian_gp_2026["win_probability"] = model.predict_proba(X_race)[:, 1]

total = australian_gp_2026["win_probability"].sum()
australian_gp_2026["win_probability_pct"] = (
    australian_gp_2026["win_probability"] / total * 100
).round(1)

print("2026 Australian GP Prediction:")

result = australian_gp_2026[
    ["driver", "team", "grid_position", "win_probability_pct"]
].sort_values("win_probability_pct", ascending=False)

for _, row in result.head(5).iterrows():
    bar = "|" * int(row["win_probability_pct"] / 2)
    print(
        f" P{int(row['grid_position']):<2} {row['driver']} ({row['team']:<12}) {bar} {row['win_probability_pct']:.1f}%"
    )
