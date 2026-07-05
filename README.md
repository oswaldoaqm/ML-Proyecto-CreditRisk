# Home Credit Default Risk — Predicción de Riesgo de Impago Crediticio

Proyecto final del curso **Machine Learning (CS 3061)** — UTEC.
Autores: Oswaldo Alejandro Quispe Monzón, Maricielo Patricia Valverde Quispe.

Predicción de impago crediticio (clasificación binaria, 8.07% de positivos) sobre el
ecosistema relacional de 7 tablas del dataset [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(más de 58 millones de registros en total).

## Resultado principal

**LightGBM ajustado con Optuna: AUC 0.7882** sobre un conjunto de validación intacto (20%).
Validación externa en Kaggle (late submission): **0.78309 público / 0.77958 privado** — coherente con la estimación interna (sin sobreajuste ni fuga).

| Modelo | AUC train | AUC val | Gap |
|---|---|---|---|
| Regresión logística (L2) | 0.7763 | 0.7749 | +0.0014 |
| Árbol de decisión (prof. 5) | 0.7166 | 0.7131 | +0.0035 |
| Árbol sin podar | 0.9943 | 0.5387 | +0.4556 |
| Random Forest | 0.9836 | 0.7675 | +0.2161 |
| MLP (64–32) | 0.8024 | 0.7701 | +0.0323 |
| XGBoost | 0.8655 | 0.7858 | +0.0798 |
| LightGBM (sin ajustar) | 0.8840 | 0.7849 | +0.0991 |
| **LightGBM ajustado** | 0.8514 | **0.7882** | +0.0632 |

Además: umbral de decisión elegido con OOF (0.172 → precisión 0.288, recall 0.424, lift 3.6×),
interpretabilidad SHAP y auditoría de equidad algorítmica por género y edad con mitigación
de igualdad de oportunidad.

## Estructura

```
├── notebooks/
│   ├── 01_audit.ipynb                  # Auditoría, downcasting, CSV → parquet
│   ├── 02a_application_clean.ipynb     # Centinelas (365243, XNA/XAP), target
│   ├── 02b_application_univariate.ipynb# Univariado por familias, drops
│   ├── 02c_application_bivariate.ipynb # Bivariado vs TARGET, ranking AUC/MI
│   ├── 03a_bureau_eda.ipynb            # EDA + agregación bureau (24 feats)
│   ├── 03b_bureau_balance.ipynb        # Agregación en 2 niveles (25 feats)
│   ├── 04_previous_application.ipynb   # Solicitudes previas (37 feats)
│   ├── 05a_pos_cash.ipynb              # POS (25 feats)
│   ├── 05b_credit_card.ipynb           # Tarjeta: utilización, ATM (31 feats)
│   ├── 05c_installments.ipynb          # Puntualidad de cuotas (23 feats)
│   ├── 06_aggregation_final_merge.ipynb# Merge maestro 1:1 → 253 columnas
│   ├── 07_modeling.ipynb               # Modelos, Optuna, umbral OOF, SHAP, fairness
│   ├── models/lgbm_tuned.txt           # Booster final
│   └── reports/                        # Figuras y rankings exportados
├── scripts/
│   └── 08_extra_baselines.py           # Baselines XGBoost y MLP (mismo protocolo)
├── paper/
│   └── main.tex                        # Informe final (IEEE Transactions)
├── data/                               # (no versionada — ver abajo)
└── PROJECT_STATE.md                    # Bitácora completa de decisiones y hallazgos
```

## Reproducir

1. Descargar los CSV de [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) en `data/raw/`.
2. Entorno: Python 3.11 con `pandas numpy pyarrow scikit-learn lightgbm xgboost optuna shap matplotlib seaborn missingno`.
3. Ejecutar los notebooks en orden (01 → 07). Cada fase lee/escribe parquet en `data/processed/` y libera memoria al cerrar (pipeline diseñado para 16 GB de RAM).
4. `scripts/08_extra_baselines.py` reproduce los baselines XGBoost y MLP.

## Decisiones metodológicas clave

- **AUC-ROC como métrica** (robusta al desbalance); accuracy descartada (~92% trivial).
- **Validación honesta**: held-out 20% intacto; hiperparámetros por CV 5-fold estratificada; umbral por predicciones out-of-fold.
- **Desbalance tratado en el umbral, no en el entrenamiento**: `scale_pos_weight=11.4` degrada el AUC del boosting.
- **Ausencia informativa**: flags `HAS_*` por tabla (thin file) en lugar de indicadores por columna.
- **Fairness**: modelo calibrado por grupo pero con TPR/FPR desiguales a umbral fijo (teorema de imposibilidad); quitar `CODE_GENDER` solo reduce ~32% del gap (proxies).
