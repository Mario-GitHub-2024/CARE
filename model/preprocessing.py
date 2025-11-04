import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TRAIN_CSV = "data/train.csv"
TARGET_COL = "churn"

df = pd.read_csv(TRAIN_CSV)

# Keep labels as strings (normalized)
y = df[TARGET_COL].astype(str).str.strip().str.lower()
X = df.drop(columns=[TARGET_COL])

# --- NEW: decide types BEFORE the split ---
numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
# set category dtype on the full X so train/test share the same categories
if categorical_cols:
    X[categorical_cols] = X[categorical_cols].astype("category")

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Scale only numeric columns (use the *names* chosen above)
sc = StandardScaler()
if numeric_cols:
    x_train_num = pd.DataFrame(
        sc.fit_transform(x_train[numeric_cols]),
        columns=numeric_cols, index=x_train.index
    )
    x_test_num = pd.DataFrame(
        sc.transform(x_test[numeric_cols]),
        columns=numeric_cols, index=x_test.index
    )
else:
    x_train_num = pd.DataFrame(index=x_train.index)
    x_test_num = pd.DataFrame(index=x_test.index)

# Rebuild X with scaled numerics + unscaled categoricals (still dtype=category)
x_train = x_train_num
x_test = x_test_num
for col in categorical_cols:
    x_train[col] = X.loc[x_train.index, col]   # retains category dtype + categories
    x_test[col]  = X.loc[x_test.index,  col]

# Optional uppercase aliases
X_train, X_test = x_train, x_test
Y_train, Y_test = y_train, y_test