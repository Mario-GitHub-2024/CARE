import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score, accuracy_score

from model.preprocessing import x_test, y_test
from model.train_model import xgb_best, rf_best, log_reg, catboost_model, LABEL_ENCODER as label_enc

# Identify cat_features by dtype (must match training)
cat_features = [c for c in x_test.columns if str(x_test[c].dtype) == "category"]

# same helper as in train_model (keep duplicated here for simplicity)
def to_numeric_for_sklearn(df):
    df_num = df.copy()
    for col in df_num.columns:
        s = df_num[col]
        if hasattr(s, "dtype") and str(s.dtype) == "category":
            codes = s.cat.codes
            df_num[col] = codes.replace({-1: codes.max() + 1}).astype("int64")
    return df_num

X_test_num = to_numeric_for_sklearn(x_test)

y_test_enc = label_enc.transform(y_test)

def add_roc_plot(model, name):
    if model is None:
        return
    # Pick the correct X and prediction method per model
    if name in ("Random Forest", "Logistic Regression"):
        test_X = X_test_num  # numeric-coded test we already build for sklearn
        probs = model.predict_proba(test_X)[:, 1]
    elif name == "CatBoost":
        if cat_features:
            from catboost import Pool
            test_pool = Pool(x_test, cat_features=cat_features)
            probs = model.predict_proba(test_pool)[:, 1]
        else:
            probs = model.predict_proba(x_test)[:, 1]
    else:  # XGBoost
        probs = model.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test_enc, probs)
    plt.plot(fpr, tpr, label=name)

def evaluate_model(model, name):
    if model is None:
        return None, None

    if name in ("Random Forest", "Logistic Regression"):
        test_X = X_test_num
        probs = model.predict_proba(test_X)[:, 1]
    elif name == "CatBoost":
        if cat_features:
            from catboost import Pool
            test_pool = Pool(x_test, cat_features=cat_features)
            probs = model.predict_proba(test_pool)[:, 1]
        else:
            probs = model.predict_proba(x_test)[:, 1]
    else:  # XGBoost
        probs = model.predict_proba(x_test)[:, 1]

    preds01 = (probs > 0.5).astype(int)
    pred_labels = label_enc.inverse_transform(preds01)

    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test_enc, probs), 3),
        "recall":  round(recall_score(y_test_enc, preds01), 3),
        "precision": round(precision_score(y_test_enc, preds01), 3),
        "accuracy": round(accuracy_score(y_test_enc, preds01), 3),
    }

    preds_df = pd.DataFrame(
        {"churn_probability": probs, "churn_prediction": pred_labels},
        index=getattr(x_test, "index", None),
    )
    return metrics, preds_df

models = []
if xgb_best is not None:       models.append(("Extreme Gradient Boosting", xgb_best))
if catboost_model is not None: models.append(("CatBoost", catboost_model))
if rf_best is not None:        models.append(("Random Forest", rf_best))
if log_reg is not None:        models.append(("Logistic Regression", log_reg))

plt.figure(figsize=(18, 8))
for name, mdl in models:
    add_roc_plot(mdl, name)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.tight_layout(); plt.show()

all_metrics = []
first_preds_df = None
for name, mdl in models:
    m, preds_df = evaluate_model(mdl, name)
    if m is not None:
        all_metrics.append(m)
        if first_preds_df is None:
            first_preds_df = preds_df.copy()

metrics_df = pd.DataFrame(all_metrics).set_index("model")
print(metrics_df)

if first_preds_df is not None:
    first_preds_df.to_csv("data/predictions.csv", index=True)
    print("Saved predictions to data/predictions.csv")