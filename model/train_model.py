import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Data from preprocessing
from model.preprocessing import x_train, y_train, x_test, y_test

from sklearn.preprocessing import LabelEncoder

# labels: keep 'yes'/'no' in the dataset, use numeric for models/metrics
label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_test_enc  = label_enc.transform(y_test)

def to_numeric_for_sklearn(df):
    """Return a copy where categorical columns are replaced by .cat.codes (int)."""
    df_num = df.copy()
    for col in df_num.columns:
        s = df_num[col]
        # if it's a pandas Categorical (or object but you forced category in preprocessing),
        # use integer codes; unseen values will not occur because categories were set on full X
        if hasattr(s, "dtype") and str(s.dtype) == "category":
            codes = s.cat.codes  # -1 means NaN; fill with a distinct code
            df_num[col] = codes.replace({-1: codes.max() + 1}).astype("int64")
    return df_num

X_train_num = to_numeric_for_sklearn(x_train)
X_test_num  = to_numeric_for_sklearn(x_test)

# Encode labels once for all models that require 0/1
label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_test_enc  = label_enc.transform(y_test)

# ===== XGBoost with Hyperopt (same structure, numeric labels) =====
dtrain = xgb.DMatrix(data=x_train, label=y_train_enc, enable_categorical=True)
N_FOLDS = 10

def xgb_objective(params, n_folds=N_FOLDS):
    params = dict(params)
    params["objective"] = "binary:logistic"
    cv = xgb.cv(
        dtrain=dtrain,
        params=params,
        nfold=n_folds,
        num_boost_round=10_000,
        early_stopping_rounds=100,
        metrics="auc",
        as_pandas=True,
        seed=42,
    )
    loss = 1.0 - cv["test-auc-mean"].iloc[-1]
    n_estimators = int(cv["test-auc-mean"].idxmax() + 1)
    return {"loss": loss, "params": params, "n_estimators": n_estimators, "status": STATUS_OK}

hyperparameter_space = {
    "n_jobs": -1,
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 0.8),
    "subsample": hp.uniform("subsample", 0.6, 0.8),
    "min_child_weight": hp.uniform("min_child_weight", 1, 7),
    "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
    "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
    "max_depth": hp.randint("max_depth", 1, 16),
    "gamma": hp.uniform("gamma", 0.0, 0.4),
    "max_delta_step": hp.randint("max_delta_step", 0, 10),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "auc",
    "random_state": 42,
}

trials = Trials()
MAX_EVALS = 15
best = fmin(fn=xgb_objective, space=hyperparameter_space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
best_n_estimators = int(trials.best_trial["result"]["n_estimators"])

xgb_best = xgb.XGBClassifier(
    **best,
    n_estimators=best_n_estimators,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=42,
)
xgb_best.fit(x_train, y_train_enc)

# ===== Random Forest =====
rf_best = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt",
    random_state=42, n_jobs=-1
)
rf_best.fit(X_train_num, y_train_enc)

# ===== CatBoost (use numeric labels for consistency) =====
# ===== CatBoost (with cat_features) =====
try:
    from catboost import CatBoostClassifier, Pool

    # All columns with pandas 'category' dtype are categorical
    cat_features = [c for c in x_train.columns if str(x_train[c].dtype) == "category"]

    catboost_model = CatBoostClassifier(
        iterations=100,
        depth=3,
        learning_rate=1.0,
        loss_function="Logloss",
        verbose=False,
        random_state=42,
    )

    if cat_features:
        train_pool = Pool(x_train, y_train_enc, cat_features=cat_features)
        catboost_model.fit(train_pool)
    else:
        # No categorical columns: train directly
        catboost_model.fit(x_train, y_train_enc)

except Exception as e:
    print("Skipping CatBoost:", e)
    catboost_model = None

# ===== Logistic Regression =====
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_num, y_train_enc)

# Expose label encoder for downstream usage
LABEL_ENCODER = label_enc
