"""Fase 8-extra: baselines XGBoost y MLP con el MISMO protocolo del notebook 07.

- Mismo split estratificado 80/20 (random_state=42), X_val intacto.
- XGBoost: encoding de arboles (OrdinalEncoder NaN->-2, unknown->-1),
  params comparables al LightGBM sin tunear (lr=0.05, 48 hojas, subsample/colsample 0.8,
  lambda 1.0, min_child_weight 50, spw=1.0), early stopping en sub-split 90/10.
- MLP: preprocesamiento lineal del notebook 07 (mediana+scaler, OneHot,
  TargetEncoder cross-fitted), arquitectura (64,32) ReLU + Adam, alpha=1e-2,
  seleccion del mejor epoch en el sub-split de early stopping.

Resultados obtenidos (val intacto):
  XGBoost: AUC train 0.8655 | val 0.7858 | best_iter 338
  MLP:     AUC train 0.8024 | val 0.7701 | best_epoch 6

Ejecutar desde la raiz del repo: python scripts/08_extra_baselines.py
"""
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, TargetEncoder)

PROC = 'data/processed'

# ---------- carga y split (identico al notebook 07) ----------
master = pd.read_parquet(f'{PROC}/master_train.parquet')
y = master['TARGET'].astype('int8')
X = master.drop(columns=['TARGET', 'SK_ID_CURR'])
del master; gc.collect()

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
del X; gc.collect()

cat_cols = X_tr.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in X_tr.columns if c not in cat_cols]

# ---------- XGBoost (encoding de arboles) ----------
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                     encoded_missing_value=-2)
enc.fit(X_tr[cat_cols])

def prep_tree(df):
    out = pd.DataFrame(enc.transform(df[cat_cols]), columns=cat_cols,
                       index=df.index).astype('float32')
    return pd.concat([out, df[num_cols].astype('float32')], axis=1)

A_tr, A_val = prep_tree(X_tr), prep_tree(X_val)
X_fit, X_es, y_fit, y_es = train_test_split(
    A_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=42)

xgb_clf = xgb.XGBClassifier(
    n_estimators=3000, learning_rate=0.05,
    max_leaves=48, grow_policy='lossguide', max_depth=0,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, min_child_weight=50,
    tree_method='hist', eval_metric='auc',
    early_stopping_rounds=100, random_state=42, n_jobs=-1)
xgb_clf.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=100)

auc_tr = roc_auc_score(y_tr, xgb_clf.predict_proba(A_tr)[:, 1])
auc_val = roc_auc_score(y_val, xgb_clf.predict_proba(A_val)[:, 1])
print(f"XGBoost: best_iter={xgb_clf.best_iteration} | "
      f"AUC train {auc_tr:.4f} | val {auc_val:.4f} | gap {auc_tr-auc_val:+.4f}")
del A_tr, A_val, X_fit, X_es; gc.collect()

# ---------- MLP (preprocesamiento lineal del notebook 07) ----------
cat_high = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']
cat_low = [c for c in cat_cols if c not in cat_high]
num_lin = [c for c in num_cols if c != 'REGION_RATING_CLIENT_W_CITY']

prep_lin = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), num_lin),
    ('cat_low', Pipeline([('imp', SimpleImputer(strategy='constant',
                                                fill_value='Missing')),
                          ('oh', OneHotEncoder(drop='if_binary',
                                               handle_unknown='ignore',
                                               sparse_output=False))]), cat_low),
    ('cat_high', Pipeline([('te', TargetEncoder(target_type='binary',
                                                random_state=42)),
                           ('sc', StandardScaler())]), cat_high),
], remainder='drop')

M_tr = prep_lin.fit_transform(X_tr, y_tr).astype('float32')
M_val = prep_lin.transform(X_val).astype('float32')
del X_tr, X_val; gc.collect()

idx = np.arange(len(M_tr))
fit_idx, es_idx = train_test_split(idx, test_size=0.10,
                                   stratify=y_tr, random_state=42)
ytr_np = y_tr.to_numpy()
Mfit, yfit = M_tr[fit_idx], ytr_np[fit_idx]
Mes, yes_ = M_tr[es_idx], ytr_np[es_idx]

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                    solver='adam', alpha=1e-2, batch_size=1024,
                    learning_rate_init=1e-3, random_state=42, max_iter=1)
rng = np.random.RandomState(42)
best_auc, best_ep, patience, coefs_best = -1, 0, 4, None
for ep in range(1, 26):
    p = rng.permutation(len(Mfit))
    mlp.partial_fit(Mfit[p], yfit[p], classes=[0, 1])
    a_es = roc_auc_score(yes_, mlp.predict_proba(Mes)[:, 1])
    if a_es > best_auc:
        best_auc, best_ep = a_es, ep
        coefs_best = ([c.copy() for c in mlp.coefs_],
                      [b.copy() for b in mlp.intercepts_])
    if ep - best_ep >= patience:
        break

mlp.coefs_, mlp.intercepts_ = coefs_best
a_tr = roc_auc_score(ytr_np, mlp.predict_proba(M_tr)[:, 1])
a_val = roc_auc_score(y_val, mlp.predict_proba(M_val)[:, 1])
print(f"MLP: best_epoch={best_ep} | AUC train {a_tr:.4f} | "
      f"val {a_val:.4f} | gap {a_tr-a_val:+.4f}")
