# 🏦 HOME CREDIT DEFAULT RISK — PROJECT_STATE.md
## Documento maestro de estado del proyecto

---

## 1. CONTEXTO
- **Objetivo**: predecir default crediticio (clasificación binaria). Dataset Home Credit Default Risk (Kaggle).
- **Unidad de predicción**: una solicitud de préstamo (`SK_ID_CURR`).
- **Variable objetivo**: `TARGET` en application_train (1 = default).
- **Ecosistema relacional de 7 tablas**, no una sola tabla.
- **Sílabo del curso** cubre: regresión, regularización (L0/L1/L2), bias-variance, cross-validation, desbalance de clases, feature engineering, métodos probabilísticos (MLE/MAP/Naive Bayes), clasificación (log reg, SVM, árboles ID3, ensambles boosting/bagging), redes neuronales, reducción de dimensionalidad (PCA/LDA/tSNE/UMAP), clustering, learning theory (algorithmic bias, interpretabilidad).

## 2. SETUP TÉCNICO
- **Hardware**: Lenovo Legion i5, 16 GB RAM.
- **Entorno**: conda env `homecredit`, Python 3.11, Jupyter en Anaconda; scikit-learn 1.6.1, lightgbm, optuna, shap, pandas/numpy, matplotlib/seaborn, missingno, pyarrow.
- **Estructura**: `data/{raw,processed}/`, `notebooks/{reports}/`, `models/`, `PROJECT_STATE.md`.
- **Pipeline**: los 8 CSV principales se convierten a parquet con downcasting de dtypes (~50% menos memoria); cada fase lee parquet, ejecuta su análisis y exporta sus salidas.

## 3. METODOLOGÍA DE TRABAJO
- Trabajo por fases liberando memoria entre tablas (`del` + `gc.collect()`).
- Poder predictivo evaluado con **AUC univariada** corrigiendo dirección: `max(auc, 1-auc)`.
- **MI no es de fiar** en columnas con muchos nulos: la imputación con centinela introduce un artefacto (la nulidad se vuelve informativa, no el valor).
- Cada fase cierra con un `.parquet` en disco y un mini-reporte en `notebooks/reports/`.

## 4. DECISIONES GLOBALES
- **Métrica principal: AUC-ROC**. Secundarias: F1 / precision / recall sobre clase 1. NUNCA accuracy (≈92% trivial por desbalance).
- **Desbalance**: 8.07% positivos, ratio 11.39:1 (moderado). Stratified split obligatorio. El reweighting va en el **umbral**, no necesariamente en el entrenamiento (ver Fase 7).
- **Centinelas**: `365243` → NaN en columnas `DAYS_*` (con indicador `_ANOM`); `XNA`/`XAP` → NaN en categóricas.
- **Redundancias eliminadas antes de modelar**: `YEARS_*` derivadas de `DAYS_*` (corr ≈ -1.00); `FLAG_EMP_PHONE` (anti-corr perfecta con `DAYS_EMPLOYED_ANOM`); housing `_MODE`/`_MEDI` (corr interna media 0.983). `REGION_RATING_CLIENT_W_CITY` ≈ `REGION_RATING_CLIENT` (0.95): drop solo para modelos lineales.
- **Outliers**: winsorización al p99 en feature engineering (AMT_INCOME_TOTAL llega a 117M; bureau AMT_CREDIT_SUM a 585M, etc.).
- **Ausencia informativa**: flags de presencia `HAS_*` por tabla agregada en lugar de flags por columna.

## 5. HALLAZGOS TRANSVERSALES (clave para el informe)
1. **Los scores externos (`EXT_SOURCE_*`) son el predictor más fuerte del proyecto** (AUC univariada 0.66–0.68) y, además, mutuamente poco correlacionados (0.11–0.21) → tres señales independientes.
2. **Dentro de cada tabla interna, la RECENCIA / LONGITUD del historial domina** sobre montos y morosidad puntual ("credit thinness"). Mejor feature por tabla: bureau → `BUR_DAYS_CREDIT_MEAN` (0.6030); bureau_balance → `BB_MONTHS_OBSERVED_MEAN` (0.5952); POS_CASH → `POS_MONTHS_BALANCE_MIN` (0.5584); previous_app → recencia `PREV_DAYS_DECISION_MEAN` (0.5581).
3. **La utilización de crédito es el 2º mejor predictor individual** (`CC_UTILIZATION_MEAN` AUC 0.6292): valida el principio clásico de credit scoring.
4. **Avances en cajero (ATM) predicen default** (`CC_HAS_DRAWING_ATM_MEAN` 0.6120): señal de estrés financiero.
5. **Upselling es anti-patrón**: `PREV_AMT_RATIO_MEAN` (0.5729, ↑ default) — clientes a quienes aprobaron MÁS de lo pedido defaultean más.
6. **Pensionistas = bajo riesgo** (5.40% vs 8.66% del resto): los 55,374 "anómalos" de `DAYS_EMPLOYED` (18.01%) son casi todos pensionistas.
7. **Los DPD agregados son débiles en TODAS las tablas mensuales** (~0.50–0.55): los eventos de mora son raros y al promediar se diluyen.
8. **El comportamiento de pago directo (installments) supera a los DPD agregados** pero con techo moderado (~0.56) por la misma rareza de eventos.
9. **Gender gap**: hombres 10.14% vs mujeres 7.00% (relevante para algorithmic bias).
10. **Distribution shift train→test**: los clientes de test tienen historial más rico en Home Credit (HAS_PREV 98.06% vs 94.65%, HAS_CC 34.16% vs 28.26%, HAS_INST 98.36% vs 94.84%); las flags `HAS_*` lo capturan.
11. **Bias-variance ilustrado en el modelado**: árbol sin podar overfittea (train 0.9943 / val 0.5387), el bagging (RF) mata la varianza (0.7675), el boosting (LightGBM) mata el sesgo (0.7882). El logístico, sorprendentemente fuerte (0.7749), confirma que gran parte de la señal es ~lineal (EXT_SOURCE monótonas).

---

# BITÁCORA POR FASES

## Fase 1 — Auditoría inicial y conversión a parquet
**Notebook: 01_audit.ipynb**

### Decisiones
- Función `downcast()` (entera/flotante) que reduce ~50% de memoria sin pérdida.
- Pipeline de carga: CSV → downcasting → parquet en `data/processed/` (solo las 8 tablas de datos; el diccionario y el sample_submission se quedan en CSV).

### Hallazgos del inventario (10 archivos, 346 columnas, 0 duplicados en todos)
| Archivo | Filas | Cols | Mem (MB) | % nulos |
|---|---|---|---|---|
| application_train | 307,511 | 122 | 505.0 | 24.40 |
| application_test | 48,744 | 121 | 79.7 | 23.81 |
| bureau | 1,716,428 | 17 | 472.8 | 13.50 |
| bureau_balance | 27,299,925 | 3 | 1,718.3 | 0.00 |
| previous_application | 1,670,214 | 37 | 1,703.0 | 17.98 |
| POS_CASH_balance | 10,001,358 | 8 | 1,060.9 | 0.07 |
| credit_card_balance | 3,840,312 | 23 | 846.4 | 6.65 |
| installments_payments | 13,605,401 | 8 | 830.4 | 0.01 |
| HomeCredit_columns_description | 219 | 5 | 0.1 | 12.15 |
| sample_submission | 48,744 | 2 | 0.7 | 0.00 |

- `bureau_balance` es la tabla más voluminosa (27.3M filas). application_train con 24.40% de celdas nulas, concentradas (no aleatorias).

### Archivos generados
- `data/processed/{application_train, application_test, bureau, bureau_balance, previous_application, POS_CASH_balance, credit_card_balance, installments_payments}.parquet`
- `reports/reports_audit_summary.csv`

### Próximo paso
- Fase 2A: limpieza base y análisis del target.

---

## Fase 2A — Limpieza base de application
**Notebook: 02a_application_clean.ipynb**

### Decisiones
- `365243` → NaN en `DAYS_*` conservando indicador `<col>_ANOM` (solo aplica a `DAYS_EMPLOYED`).
- `XNA`/`XAP` → NaN en categóricas.
- Versiones positivas en años para EDA: `AGE_YEARS`, `EMPLOYMENT_YEARS`, `YEARS_SINCE_REGISTRATION`, `YEARS_SINCE_ID_PUBLISH`, `YEARS_SINCE_PHONE_CHANGE` (creadas con `-días/365.25`).

### Hallazgos
- **TARGET**: 0 = 282,686 (91.93%), 1 = 24,825 (8.07%); ratio 11.39:1.
- `DAYS_EMPLOYED == 365243`: 55,374 filas (18.01%) → todas coinciden con `ORGANIZATION_TYPE = XNA`.
- Centinelas categóricos: `ORGANIZATION_TYPE` XNA 55,374 (18.01%); `CODE_GENDER` XNA 4; `NAME_FAMILY_STATUS` Unknown 2.
- Nulidad por bandas (de 128 columnas): 0% → 61, 0–10% → 7, 10–30% → 10, 30–50% → 9, 50–70% → 41, 70–100% → 0. **72 columnas con al menos un nulo.**
- Top nulidad: bloque housing `COMMONAREA_*` 69.87%, `NONLIVINGAPARTMENTS_*` 69.43%, `OWN_CAR_AGE` 65.99%, `EXT_SOURCE_1` 56.38%.

### Archivos generados
- `application_train_clean.parquet` (307,511 × 128, 348.5 MB), `application_test_clean.parquet` (48,744 × 127, 55.2 MB)
- `reports/target_distribution.png`, `reports/missingness_matrix.png`, `reports/missingness_ranking_train.csv`

### Próximo paso
- Fase 2B: univariado por familias de variables.

---

## Fase 2B — Univariado por familias de variables
**Notebook: 02b_application_univariate.ipynb**

### Decisiones (drops confirmados para 2C)
- Drop housing `_MODE` (15) y `_MEDI` (14) por correlación interna 0.98+.
- Drop 9 `FLAG_DOCUMENT_*` con <0.1% de activación.
- Drop `FLAG_MOBIL` (constante: 307,510 unos y 1 cero).
- Feature creada: `DOCUMENTS_PROVIDED_COUNT` = suma de `FLAG_DOCUMENT_*`.
- `EXT_SOURCE_*` se conservan las 3 (señales independientes); `AMT_INCOME_TOTAL` → log + winsorización en FE.

### Hallazgos
- Los 55,374 "anómalos" se descomponen en **55,352 pensionistas** + 22 desempleados. Default del grupo anómalo 5.40% vs 8.66% del resto → `DAYS_EMPLOYED_ANOM` es predictor fuerte.
- `AMT_INCOME_TOTAL`: skewness 391.56, kurtosis 191,786, max 117,000,000 (un cliente).
- **EXT_SOURCE** (AUC univariada corregida / % nulos / correlación interna):
  - `EXT_SOURCE_3` = 0.6794 (19.83% nulos)
  - `EXT_SOURCE_1` = 0.6657 (56.38% nulos)
  - `EXT_SOURCE_2` = 0.6561 (0.21% nulos)
  - correlaciones entre sí 0.11–0.21; todas inversas (mayor score → menor riesgo).
- Housing: correlación interna media 0.983 (máx `YEARS_BUILD` 0.992).
- `FLAG_DOCUMENT_3` activa en 71.00% (ID estándar); 270,056 clientes (87.8%) proveen exactamente 1 documento.
- Cardinalidades: `ORGANIZATION_TYPE` 57, `OCCUPATION_TYPE` 18 (31.3% nulos). Casi-constantes: `HOUSETYPE_MODE` (98.23%), `EMERGENCYSTATE_MODE` (98.56%). Columnas de varianza cero: ninguna.

### Archivos generados
- `application_train_clean.parquet` actualizado (con `DOCUMENTS_PROVIDED_COUNT`)
- `reports/univar_loan_financials.png`, `reports/univar_ext_sources.png`

### Próximo paso
- Fase 2C: bivariado contra TARGET + ranking unificado.

---

## Fase 2C — Bivariado y ranking unificado
**Notebook: 02c_application_bivariate.ipynb**

### Decisiones
- Aplicado el drop de **39 columnas** (29 housing `_MODE`/`_MEDI` + 9 docs raros + 1 constante): shape (307,511 × 90) train / (48,744 × 89) test.
- Para modelar: mantener `DAYS_*` y dropear las `YEARS_*` derivadas; drop `FLAG_EMP_PHONE`; `REGION_RATING_CLIENT_W_CITY` solo para lineales.
- Ranking de referencia: **AUC > MI** (MI corrompido por imputación en columnas con muchos nulos).

### Hallazgos (ranking unificado)
- Top numéricas por AUC univariada: `EXT_SOURCE_3` 0.6794, `EXT_SOURCE_1` 0.6657, `EXT_SOURCE_2` 0.6561, `DAYS_BIRTH`/`AGE_YEARS` 0.5830, `DAYS_EMPLOYED`/`EMPLOYMENT_YEARS` 0.5824, `OWN_CAR_AGE` 0.5589, `DAYS_LAST_PHONE_CHANGE` 0.5569, `DAYS_ID_PUBLISH` 0.5557, `REGION_RATING_CLIENT_W_CITY` 0.5492.
- Default por banda de edad (monótono): 20–25 = 12.29% → 25–30 = 11.13% → 30–40 = 9.59% → 40–50 = 7.64% → 50–60 = 6.12% → 60–70 = 4.92%.
- Default por decil de `EXT_SOURCE_3`: peor decil 20.00% → mejor decil 3.23% (ratio ≈ 6.2×).
- Gender: M 10.14% vs F 7.00%.
- Educación: Lower secondary 10.93% → Academic degree 1.83% (≈6× de spread).
- `OCCUPATION_TYPE` top riesgo: Low-skill Laborers 17.15%, Drivers 11.33%, Waiters/barmen 11.28%, Security 10.74%, Laborers 10.58%; bajo riesgo: Accountants 4.83%, High skill tech 6.16%, Managers 6.21%.
- `NAME_CONTRACT_TYPE`: Cash loans 8.35% vs Revolving loans 5.48%.
- Cramér's V máximo 0.082 (`OCCUPATION_TYPE`); todas las categóricas significativas pero con efecto individual pequeño.
- **Caveat MI**: el bloque housing domina el MI (`FLOORSMIN_AVG` 0.0468, etc.) porque refleja informatividad de la NULIDAD, no del valor. Decisión: usar AUC para selección; en FE, flags `IS_NULL_<col>` si se reincorporan.

### Archivos generados
- `application_train_reduced.parquet` (307,511 × 90), `application_test_reduced.parquet` (48,744 × 89)
- `reports/auc_rank_numeric.csv`, `chi2_rank_categorical.csv`, `mi_rank.csv`, `ranking_combined_top30.csv`, `bivar_age_ext3.png`, `bivar_corr_top20.png`

### Próximo paso
- Fase 3A: EDA + agregación de bureau.

---

## Fase 3A — EDA y agregación de bureau
**Notebook: 03a_bureau_eda.ipynb**

### Decisiones
- Flag `HAS_BUREAU_HISTORY` en FE (14.31% de clientes sin historial).
- Drop `CREDIT_CURRENCY` (99.92% en una sola categoría).
- 24 features agregadas (prefix `BUR_`), incluyendo bloques de créditos activos/cerrados y `BUR_ACTIVE_RATIO`.

### Hallazgos
- 85.69% de clientes train con historial (263,491 de 307,511); sin historial 44,020.
- Créditos por cliente: media 5.6, mediana 4, máx 116.
- `CREDIT_ACTIVE`: Closed 62.88%, Active 36.74%, Sold 0.38%, Bad debt 21 casos.
- `CREDIT_TYPE`: Consumer credit 72.92%, Credit card 23.43%, Car loan 1.61%, Mortgage 1.07%.
- Top features (todas temporales): `BUR_DAYS_CREDIT_MEAN` 0.6030, `BUR_DAYS_CREDIT_UPDATE_MEAN` 0.5893, `BUR_ACTIVE_RATIO` 0.5840, `BUR_DAYS_CREDIT_MAX` 0.5803, `BUR_DAYS_CREDIT_MIN` 0.5783.
- Overdue débiles: `BUR_AMT_CREDIT_MAX_OVERDUE_MAX` 0.5345; `BUR_CREDIT_DAY_OVERDUE_*` ≈ 0.507. `BUR_N_CREDITS` solo 0.5049 → la cantidad de créditos sola no predice; importa la composición.
- Calidad detectada (a tratar en FE): `DAYS_CREDIT_ENDDATE` rango [-42060, 31199] (602,603 valores positivos = fin a futuro de créditos activos); `DAYS_ENDDATE_FACT`/`DAYS_CREDIT_UPDATE` con extremos ≈ -42k; `AMT_CREDIT_SUM` máx 585M, `AMT_ANNUITY` máx 118M, `AMT_CREDIT_MAX_OVERDUE` máx 116M; `AMT_CREDIT_SUM_DEBT` negativo hasta -4.7M (sobrepagos). Nulos altos: `AMT_ANNUITY` 71.47%, `AMT_CREDIT_MAX_OVERDUE` 65.51%.

### Archivos generados
- `bureau_aggregated.parquet` (305,811 × 25)
- `reports/bureau_credits_per_client.png`, `auc_rank_bureau_agg.csv`

### Próximo paso
- Fase 3B: bureau_balance (agregación en dos niveles).

---

## Fase 3B — EDA y agregación de bureau_balance
**Notebook: 03b_bureau_balance.ipynb**

### Decisiones
- Agregación en dos niveles: `SK_ID_BUREAU` → `SK_ID_CURR`.
- Mapeo `STATUS → DPD_LEVEL` (0–5) con C como 0 y X como NaN; ambos conservados como conteos/proporciones.
- `SK_ID_BUREAU` sin mapeo a cliente (43,041 = 5.27%) → descartados.
- `bureau_master` = `bureau_aggregated` + `bb_at_client` (49 columnas).

### Hallazgos
- 27.3M filas, 817,395 créditos únicos, hasta 96 meses de historial.
- `STATUS`: C 49.99%, 0 27.47%, X 21.28%, 1 0.888%, 5 0.229%, 2 0.086%, 3 0.033%, 4 0.021%.
- Anomalía: `STATUS=5` (write-off, 62,406) supera a 2+3+4 juntos (38,190) → muchos casos saltan directo a write-off en vez de progresar 1→2→3→4→5.
- Filas con cualquier DPD: 342,943 (1.26%).
- Cobertura desigual: bureau tiene 305,811 clientes pero bureau_balance solo mapea a 134,542 (44%); 171,269 (56%) de clientes con bureau no tienen seguimiento mensual.
- Top features (longitud/madurez del historial): `BB_MONTHS_OBSERVED_MEAN` 0.5952, `BB_STATUS_C_CNT_MEAN` 0.5751, `BB_STATUS_C_PCT_MEAN` 0.5600. DPD débiles (≈0.54) por rareza.

### Archivos generados
- `bureau_balance_aggregated.parquet` (134,542 × 25)
- `bureau_master.parquet` (305,811 × 49) ← el que entra al merge final
- `reports/auc_rank_bureau_balance_agg.csv`

### Próximo paso
- Fase 4: previous_application.

---

## Fase 4 — EDA y agregación de previous_application
**Notebook: 04_previous_application.ipynb**

### Decisiones
- Limpieza de `365243` en 6 columnas `DAYS_*` y de `XNA`/`XAP` en categóricas.
- Drop tentativo en FE: `RATE_INTEREST_PRIMARY` y `RATE_INTEREST_PRIVILEGED` (99.64% nulos).
- Features derivadas: `PREV_AMT_DIFF` = AMT_APPLICATION − AMT_CREDIT; `PREV_AMT_RATIO` = AMT_CREDIT / AMT_APPLICATION.
- 37 features agregadas (prefix `PREV_`), con bloques de aprobadas/rechazadas/canceladas/unused y tasas (`PREV_APPROVAL_RATE`, `PREV_REFUSAL_RATE`, `PREV_UNUSED_RATE`).

### Hallazgos
- Cobertura 94.65% (291,057 de 307,511). Solicitudes por cliente: media 4.9, mediana 4, máx 77.
- `NAME_CONTRACT_STATUS`: Approved 62.07%, Canceled 18.94%, Refused 17.40%, Unused offer 1.58%.
- `NAME_CLIENT_TYPE`: Repeater 73.72%, New 18.04%, Refreshed 8.12%.
- `CODE_REJECT_REASON`: NaN 81.33%, HC 10.49%, LIMIT 3.33%, SCO 2.24% (señal latente sin explotar a fondo).
- Centinelas relevantes: `DAYS_FIRST_DRAWING` 934,444 (55.95%) = créditos aprobados pero no desembolsados (96.25% nulo tras limpieza).
- Solicitudes donde aprobado < pedido: 357,691 (21.42%); `PREV_AMT_RATIO` media 1.03, máx 20.0.
- **Insight central**: `PREV_PREV_AMT_RATIO_MEAN` es el top predictor (AUC 0.5729, ↑ más default) → clientes upselled defaultean más. Le siguen `PREV_REFUSAL_RATE` 0.5602, `PREV_APPROVAL_RATE` 0.5589 (↓ default), `PREV_DAYS_DECISION_MEAN` 0.5581 (recencia), `PREV_PREV_AMT_DIFF_MAX` 0.5572.

### Archivos generados
- `previous_app_aggregated.parquet` (338,857 × 38)
- `reports/prev_per_client.png`, `auc_rank_previous_app.csv`

### Próximo paso
- Fase 5A: POS_CASH_balance.

---

## Fase 5A — EDA y agregación de POS_CASH_balance
**Notebook: 05a_pos_cash.ipynb**

### Decisiones
- Agregación directa a `SK_ID_CURR`. Indicadores `POS_HAS_DPD`, `POS_HAS_DPD_DEF`, `POS_IS_COMPLETED`, `POS_IS_ACTIVE`.
- 25 features agregadas (prefix `POS_`; los nombres con doble prefijo `POS_POS_*` son inofensivos).

### Hallazgos
- Cobertura 94.12% (289,444 de 307,511). 91.50% de filas Active, 7.45% Completed.
- DPD raros: `SK_DPD>0` en 2.952%, `SK_DPD_DEF>0` en 1.140%.
- Es la tabla mensual **más débil**: mejor feature `POS_MONTHS_BALANCE_MIN` 0.5584 (↑ default); el resto de DPD 0.52–0.53.

### Archivos generados
- `pos_cash_aggregated.parquet` (337,252 × 26)
- `reports/auc_rank_pos_cash.csv`

### Próximo paso
- Fase 5B: credit_card_balance.

---

## Fase 5B — EDA y agregación de credit_card_balance
**Notebook: 05b_credit_card.ipynb**

### Decisiones
- Feature clave `CC_UTILIZATION` = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL (límite 0 → NaN; sobregiro con límite 0 marcado por `CC_IS_OVERDRAFT` y utilización fijada a 1.0).
- Indicadores `CC_HAS_DPD`, `CC_HAS_DRAWING_ATM`. 31 features agregadas (prefix `CC_`).
- Cobertura baja: NaN en `CC_*` significará "sin tarjeta Home Credit".

### Hallazgos
- Cobertura SOLO 28.26% (86,905 de 307,511).
- `NAME_CONTRACT_STATUS`: Active 96.31%, Completed 3.36%.
- 7,992 filas con límite 0 y saldo deudor (overdraft); avances ATM en 11.06% de filas; `SK_DPD>0` en 3.993%.
- **Top hallazgo**: `CC_UTILIZATION_MEAN` AUC 0.6292 = 2º mejor predictor del proyecto (tras EXT_SOURCE). Familia de gasto/disposición muy fuerte: `CC_CNT_DRAWINGS_CURRENT_MEAN` 0.6235, `CC_HAS_DRAWING_ATM_MEAN` 0.6120 (ATM = estrés), `CC_AMT_DRAWINGS_CURRENT_MEAN` 0.6071, `CC_AMT_BALANCE_MEAN` 0.6051.
- DPD débiles otra vez (0.50–0.51). credit_card es mucho más rica que POS_CASH (6 features >0.60 vs 0).

### Archivos generados
- `credit_card_aggregated.parquet` (103,558 × 32)
- `reports/auc_rank_credit_card.csv`

### Próximo paso
- Fase 5C: installments_payments.

---

## Fase 5C — EDA y agregación de installments_payments
**Notebook: 05c_installments.ipynb**

### Decisiones
- Features derivadas: `INST_DAYS_LATE` = DAYS_ENTRY_PAYMENT − DAYS_INSTALMENT (positivo = tarde); `INST_PAYMENT_DIFF` = AMT_INSTALMENT − AMT_PAYMENT (positivo = pagó de menos); `INST_PAYMENT_RATIO` = AMT_PAYMENT / AMT_INSTALMENT (<1 = pagó de menos).
- Cuotas sin pago registrado tratadas como impago real (`INST_IS_UNPAID`), con el peor escenario continuo asignado (ratio 0, diff = cuota completa).
- Indicadores `INST_IS_LATE`, `INST_IS_UNDERPAID`. 23 features agregadas (prefix `INST_`).

### Hallazgos
- Cobertura 94.84% (291,643 de 307,511). Pagos por cliente: media 40.1, mediana 25, máx 372.
- Pagos tardíos (incluye impagos) 8.45%; incompletos 9.54% (eventos relativamente raros).
- El comportamiento de pago directo SÍ supera a los DPD agregados, pero con techo ~0.56. Top: `INST_IS_LATE_MEAN` 0.5645, `INST_PAYMENT_RATIO_MEAN` 0.5577, `INST_PAYMENT_DIFF_MEAN` 0.5554, `INST_IS_UNDERPAID_MEAN` 0.5547.

### Archivos generados
- `installments_aggregated.parquet` (339,587 × 24)
- `reports/auc_rank_installments.csv`

### Próximo paso
- Fase 6: merge maestro.

---

## Fase 6 — Merge maestro
**Notebook: 06_aggregation_final_merge.ipynb**

### Decisiones
- Left join secuencial sobre `application_*_reduced` (base 90/89 cols) con las 5 tablas agregadas, todas on `SK_ID_CURR`, `validate='1:1'` (sin colisiones `_x`/`_y`).
- Flags de presencia creadas ANTES del join (ausencia informativa): `HAS_BUREAU`, `HAS_PREV`, `HAS_POS`, `HAS_CC`, `HAS_INST`.
- Drops de redundancia (6, info-preserving): `AGE_YEARS`, `EMPLOYMENT_YEARS`, `YEARS_SINCE_REGISTRATION`, `YEARS_SINCE_ID_PUBLISH`, `YEARS_SINCE_PHONE_CHANGE` (corr ≈ -1.00 con sus `DAYS_*`) + `FLAG_EMP_PHONE`.
- Conservadas: housing `YEARS_BEGINEXPLUATATION_AVG`, `YEARS_BUILD_AVG` (el drop por prefijo `YEARS_` se hace de forma selectiva para no borrarlas) y ambos `REGION_RATING` (su drop es por-modelo, se decide en Fase 7).
- Downcast + guardado en parquet con orden de columnas test == train sin TARGET.

### Hallazgos
- Aportes al merge: bureau_master +48, previous_app +37, pos_cash +25, credit_card +31, installments +23.
- **`master_train`: (307,511 × 253) | `master_test`: (48,744 × 252)** — paridad = solo `TARGET`.
- Cobertura train: HAS_BUREAU 85.69%, HAS_PREV 94.65%, HAS_POS 94.12%, HAS_CC 28.26%, HAS_INST 94.84%.
- **Distribution shift train→test**: coberturas de test sistemáticamente más altas (HAS_PREV 98.06%, HAS_POS 98.08%, HAS_CC 34.16%, HAS_INST 98.36%, HAS_BUREAU 86.82%). Retomado en SHAP/fairness.
- Memoria tras downcast: train 553.1 MB / test 87.6 MB. Roundtrip de parquet verificado; `TARGET` int8, 8.07% positivos.

### Archivos generados
- `data/processed/master_train.parquet` (307,511 × 253)
- `data/processed/master_test.parquet` (48,744 × 252)

### Próximo paso
- Fase 7: split estratificado + encoding/imputación + modelado.

---

## Fase 7 — Modelado, tuning, interpretabilidad y fairness
**Notebook: 07_modeling.ipynb**

### Decisiones
- Split estratificado 80/20 (random_state=42): `X_tr` (246,008 × 251) / `X_val` (61,503 × 251); positivos 8.073% en ambos. `X_val` se mantiene como held-out honesto.
- **Dos preprocesamientos** según familia:
  - *Lineal*: imputación mediana + StandardScaler (num); OneHot(drop='if_binary', NaN→'Missing') para cat de baja cardinalidad; `TargetEncoder` cross-fitted para `ORGANIZATION_TYPE` y `OCCUPATION_TYPE`. Se dropea `REGION_RATING_CLIENT_W_CITY` (colinealidad 0.95). num=234, cat_low=14, cat_high=2 → 302 features tras OneHot.
  - *Árboles*: `OrdinalEncoder` (NaN→-2, unknown→-1) + numéricas crudas. LightGBM con categóricas dtype `category` (splits nominales nativos); se conservan ambos `REGION_RATING`.
- No se crean flags `IS_NULL` por columna: los nulos de las agregadas son estructurales y los capturan las flags `HAS_*` (197 de 235 numéricas tienen nulos; las `CC_*` ~71.7% por la cobertura del 28%).
- Métrica de selección: AUC-ROC vía CV estratificada 5-fold.
- **Lección clave**: `scale_pos_weight=11.4` **destruye** el boosting basado en AUC (best_iter 1–3). El AUC es robusto al desbalance; el reweighting va en el **umbral**, no en el entrenamiento.
- Tuning: Optuna, 180.2 min / **114 trials**, cada trial = `lgb.cv` 5-fold completo.

### Hallazgos — comparativa (held-out X_val)
| Modelo | AUC train | AUC val | gap |
|---|---|---|---|
| LogReg L2 (balanced) | 0.7763 | 0.7749 | +0.0014 |
| Árbol decisión depth=5 | 0.7166 | 0.7131 | +0.0035 |
| Árbol sin podar | 0.9943 | 0.5387 | +0.4556 (depth 103, 22,634 hojas) |
| Random Forest | 0.9836 | 0.7675 | +0.2161 |
| LightGBM (sin tunear, spw=1.0) | 0.8840 | 0.7849 | +0.0991 |
| LightGBM (spw=11.4) | 0.7286 | 0.7195 | +0.0091 |
| **LightGBM TUNEADO** | 0.8514 | **0.7882** | +0.0632 |

- **GANADOR: LightGBM tuneado.** CV óptima 0.7861 (vs 0.7820 sin tunear) | held-out 0.7882. +0.013 sobre el baseline logístico.
- Params finales: `num_leaves=16`, `min_child_samples=104`, `feature_fraction=0.5557`, `bagging_fraction=0.9053`, `bagging_freq=1`, `lambda_l1=9.0122`, `lambda_l2=9.9375`, `min_split_gain=0.3132`, `learning_rate=0.05`, `num_boost_round=983` (best_round CV × 1.1). Modelo fuertemente regularizado: la señal está en combinar muchos predictores débiles.

### Umbral y decisión de negocio
- Umbral elegido FUERA de X_val con predicciones OOF de train: **F1-óptimo = 0.172** (no 0.5; las probas se centran en el base rate 8%).
- En ese punto sobre X_val: marca 7,288 / 61,503 solicitudes (**11.8%**), **precision 0.288** (lift ≈ 3.6× sobre el base rate), **recall 0.424**. Matriz de confusión [[51353, 5185],[2862, 2103]].
- Trade-off PR empinado sobre X_val: recall≥0.50 → prec 0.260; recall≥0.70 → prec 0.188; recall≥0.80 → prec 0.157. Es consecuencia del AUC 0.79 con 8% de positivos, no un defecto: el umbral es palanca de política (FN ≫ FP en costo → favorecer recall).

### Interpretabilidad y fairness (sección 5.10)
- **SHAP** (|SHAP| medio, top): `EXT_SOURCE_2` 0.298, `EXT_SOURCE_3` 0.255, `EXT_SOURCE_1` 0.144, `AMT_GOODS_PRICE` 0.118, `AMT_ANNUITY` 0.116, **`CODE_GENDER` 0.113 (#6 de 251)**, `AMT_CREDIT` 0.110, `ORGANIZATION_TYPE` 0.096, `INST_IS_LATE_MEAN` 0.095. Dependence de `EXT_SOURCE_3` monótona decreciente con nulos como riesgo elevado; `PREV_AMT_RATIO_MEAN` confirma el anti-patrón de upselling.
- El modelo se apoya FUERTE en el atributo protegido directo `CODE_GENDER` (disparate treatment, problemático legalmente).
- **Fairness por género (umbral 0.172)**: AUC F 0.7830 ≈ M 0.7854 y calibración casi perfecta (pred_prob_mean ≈ default_real en ambos), PERO a umbral fijo divergen recall (F 0.3695 vs M 0.4955) y FPR (F 0.0722 vs M 0.1309) → **teorema de imposibilidad** (con base rates distintos no coexisten calibración + igual TPR + igual FPR). flag_rate F 0.093 vs M 0.168.
- **Fairness por edad**: calibrada; AUC peor en extremos (<25: 0.7470, 60+: 0.7446) por thin-file y homogeneidad; flag_rate cae ≈6× con la edad (0.2337 → 0.0380).
- **Mitigación equal-opportunity**: umbrales por grupo (F 0.139, M 0.184) para TPR ~0.45 → desempeño global precision 0.270 / recall 0.458. Igualar TPR tiene costo directo en precisión: es decisión de política, no solo técnica.
- **Experimento "unawareness"**: quitar `CODE_GENDER` cuesta solo 0.0018 de AUC (0.7882 → 0.7864), pero recorta el gap predicho M−F apenas de 0.0300 a 0.0204 (real 0.0318) → los proxies reconstruyen ~68% del gap. Necesario pero insuficiente.

### Archivos generados
- `models/lgbm_tuned.txt` (booster final)
- `reports/pr_roc_lgbm_tuned.png`, `shap_summary.png`, `shap_dep_ext3.png`

### Próximo paso
- Fase 8: predicciones sobre master_test + informe final.

---

## Fase 8 — Submission + informe final
**Notebook: 07_modeling.ipynb (celda final)**

- Predicciones sobre `master_test` con las mismas columnas/orden/categorías que vio el modelo: `reports/submission_lgbm_tuned.csv` (48,744 × 2), prob media 0.0716 (base rate train 0.0807).
- Informe final del proyecto: pendiente de redacción.
- **PROYECTO: EDA + agregación + modelado COMPLETOS.**

---

## Fase 8-extra — Baselines XGBoost y MLP (para el informe final)
**Script: scripts/08_extra_baselines.py**

- Motivación: el informe parcial prometía XGBoost; el sílabo cubre redes neuronales. Mismo protocolo del notebook 07 (split 80/20 rs=42, sub-split 90/10 para early stopping, X_val intacto).
- **XGBoost** (hist, lossguide 48 hojas, lr=0.05, spw=1.0): AUC train 0.8655 | **val 0.7858** | best_iter 338. Queda entre LightGBM sin tunear (0.7849) y el tuneado (0.7882) → LightGBM tuneado sigue siendo el ganador.
- **MLP** (64–32, ReLU, Adam, alpha=1e-2, mejor epoch=6 por early stopping): AUC train 0.8024 | **val 0.7701**. Por debajo incluso de la logística (0.7749) → replica a Gunnarsson et al. (EJOR 2021): deep learning no supera al boosting en tabular de crédito.

## Fase 8-bis — Validación externa Kaggle (late submission)
- `submission_lgbm_tuned.csv` → **AUC público 0.78309 | privado 0.77958** (ganador de la competencia: 0.80570 privado). Coherente con la validación interna 0.7882 → protocolo honesto confirmado. Incluido en el paper (sección "Validación externa en Kaggle").

## Fase 9 — Entrega final
- `paper/main.tex`: informe IEEE Transactions completo (abstract ≤150 palabras, 15 referencias, metodología, resultados con 5 tablas + 5 figuras, subsección de fairness, conclusiones, trabajo futuro).
- `README.md` + `.gitignore` listos para publicar el repo en GitHub (data excluida).
- Pendiente (usuario): crear repo GitHub y reemplazar el placeholder del link en `paper/main.tex` (sección "Código implementado"); compilar en Overleaf; enviar PDF a vmartinez@utec.edu.pe antes del 11 de julio 11:59 p.m.