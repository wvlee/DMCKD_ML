# 🚀 DMCKD_ML
Sample codes for model fitting and the final pickle-formatted model for Lee _et al_ (_Diabetes Res Clin Pract_, 2024).

## 🏸 Model training and hyperparameter optimization
Hyperparameters for the eXtreme Gradient Boosting (XGBoost) algorithm were tuned using a Bayesian optimization framework with 5-fold cross-validation. The optimized booster hyperparameters were as follows: tree structure (maximum depth and number of estimators), sampling strategy (subsample and column sampling by tree), regularization terms (*α* and *λ*), split policy (*γ*), shrinkage factor (*η*), and weight considerations (minimum sum of weight in a child). To account for class imbalance, the weight given to the positive class was fixed to the ratio of the number of individuals experiencing non-rapid decline in kidney function to the number of individuals experiencing rapid decline in kidney function. The objective function to maximize was the mean AUPRC score, and the expected improvement (EI) acquisition function was utilized to determine the next test sample point. The exploitation-exploration trade-off parameter (*ξ*) of EI was set to 0.01.6 Initial exploration was performed in 3 randomly selected points on the hyperparameter space, and hyperparameters were optimized after 50 iterations.

## 🏸 Feature names and corresponding clinical variables
| feature names          | predictor                                 |
| ---------------------- | ----------------------------------------- |
| `num_GS_311702`        | Urine albumin-to-creatinine ratio [mg/mg] |
| `num_GS_3311`          | Serum albumin [g/dL]                      |
| `num_GS_3012`          | Alkaline phosphatase [IU/L]               |
| `num_GS_3011`          | Total bilirubin [mg/dL]                   |
| `num_GS_3010`          | Serum albumin [g/dL]                      |
| `num_GS_2004`          | Hematocrit [%]                            |
| `num_CLN_systbp`       | Systolic blood pressure [mmHg]            |
| `num_CHAR_NUM_age_ckd` | Age at baseline [year]                    |

