##################################################################################
#######################                                  #########################
####################### XGBoost optimization sample code #########################
#######################                                  #########################
##################################################################################

# This is a sample code for Bayesian optimization followed by fitting an XGBoost classifier

#################################################
############# Set working directory #############
#################################################

import os
wkdir = '' # Set working directory
os.chdir(wkdir)

#################################################
############### Import libraries ################
#################################################

# Data analytics
import pandas as pd
import numpy as np

# Stratified k-fold cross-validation
from sklearn.model_selection import StratifiedKFold

# Set model
from xgboost import XGBClassifier

# Optimization
from bayes_opt import BayesianOptimization, UtilityFunction

# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score, \
confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, class_likelihood_ratios

# Calibration
from sklearn.calibration import CalibratedClassifierCV

# Save model
import pickle
import joblib

###############################################
############## Define functions ###############
###############################################

# Acquire model performance after fitting model
def fitted_model_performance(fitted_model,X_test_,y_test_,cutoff):
    auto_df = pd.DataFrame()
    
    y_pred_ = fitted_model.predict(X_test_)
    y_prob_ = fitted_model.predict_proba(X_test_)[:,1]

    auroc_auto = roc_auc_score(y_test_,y_prob_)
    auprc_auto = average_precision_score(y_test_,y_prob_)

    # Default cutoff
    cm_auto     = confusion_matrix(y_test_,y_pred_)
    tn_auto     = cm_auto[0,0]
    fp_auto     = cm_auto[0,1]
    fn_auto     = cm_auto[1,0]
    tp_auto     = cm_auto[1,1]
    
    sensi_auto  = tp_auto / (tp_auto + fn_auto)
    speci_auto  = tn_auto / (tn_auto + fp_auto)
    prsco_auto  = precision_score(y_test_,y_pred_)
    rcsco_auto  = recall_score(y_test_,y_pred_)
    
    acsco_auto  = accuracy_score(y_test_,y_pred_)
    f1sco_auto  = f1_score(y_test_,y_pred_)
    mcsco_auto  = matthews_corrcoef(y_test_,y_pred_)
    pos_LR_auto, neg_LR_auto  = class_likelihood_ratios(y_test_,y_pred_)
    dor_auto    = pos_LR_auto / neg_LR_auto
    
    # Update cutoff
    y_pred_upd_ = (y_prob_ > cutoff).astype(int)
    cm_auto_upd     = confusion_matrix(y_test_,y_pred_upd_)
    tn_auto_upd     = cm_auto_upd[0,0]
    fp_auto_upd     = cm_auto_upd[0,1]
    fn_auto_upd     = cm_auto_upd[1,0]
    tp_auto_upd     = cm_auto_upd[1,1]
    
    sensi_auto_upd  = tp_auto_upd / (tp_auto_upd + fn_auto_upd)
    speci_auto_upd  = tn_auto_upd / (tn_auto_upd + fp_auto_upd)
    prsco_auto_upd  = precision_score(y_test_,y_pred_upd_)
    rcsco_auto_upd  = recall_score(y_test_,y_pred_upd_)
    
    acsco_auto_upd  = accuracy_score(y_test_,y_pred_upd_)
    f1sco_auto_upd  = f1_score(y_test_,y_pred_upd_)
    mcsco_auto_upd  = matthews_corrcoef(y_test_,y_pred_upd_)
    pos_LR_auto_upd, neg_LR_auto_upd  = class_likelihood_ratios(y_test_,y_pred_upd_)
    dor_auto_upd    = pos_LR_auto_upd / neg_LR_auto_upd
    
    auto_df.loc[0,'featnum']            = fitted_model.n_features_in_
    auto_df.loc[0,'sfm_feature_list']   = ', '.join(X_test_.columns)
    auto_df.loc[0,'model_auroc']        = auroc_auto
    auto_df.loc[0,'model_auprc']        = auprc_auto
    auto_df.loc[0,'initial_TN']         = tn_auto
    auto_df.loc[0,'initial_FP']         = fp_auto
    auto_df.loc[0,'initial_FN']         = fn_auto
    auto_df.loc[0,'initial_TP']         = tp_auto
    auto_df.loc[0,'initial_sensitivity'] = sensi_auto
    auto_df.loc[0,'initial_specificity'] = speci_auto
    auto_df.loc[0,'initial_precision']  = prsco_auto
    auto_df.loc[0,'initial_recall']     = rcsco_auto
    auto_df.loc[0,'initial_accuracy']   = acsco_auto
    auto_df.loc[0,'initial_f1score']    = f1sco_auto
    auto_df.loc[0,'initial_mcc']        = mcsco_auto
    auto_df.loc[0,'initial_LR+']        = pos_LR_auto
    auto_df.loc[0,'initial_LR−']        = neg_LR_auto
    auto_df.loc[0,'initial_DOR']        = dor_auto
    auto_df.loc[0,'best_cutoff']        = cutoff
    auto_df.loc[0,'updated_TN']         = tn_auto_upd
    auto_df.loc[0,'updated_FP']         = fp_auto_upd
    auto_df.loc[0,'updated_FN']         = fn_auto_upd
    auto_df.loc[0,'updated_TP']         = tp_auto_upd
    auto_df.loc[0,'updated_sensitivity'] = sensi_auto_upd
    auto_df.loc[0,'updated_specificity'] = speci_auto_upd
    auto_df.loc[0,'updated_precision']  = prsco_auto_upd
    auto_df.loc[0,'updated_recall']     = rcsco_auto_upd
    auto_df.loc[0,'updated_accuracy']   = acsco_auto_upd
    auto_df.loc[0,'updated_f1score']    = f1sco_auto_upd
    auto_df.loc[0,'updated_mcc']        = mcsco_auto_upd
    auto_df.loc[0,'updated_LR+']        = pos_LR_auto_upd 
    auto_df.loc[0,'updated_LR−']        = neg_LR_auto_upd
    auto_df.loc[0,'updated_DOR']        = dor_auto_upd
    
    performance = auto_df.T
    return performance

#############################################
################# Set BOpt ##################
#############################################
# Hyperparameter space
xgb_param_space = {
    'max_depth':(3,10),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'min_child_weight':(0,5),
    'gamma':(0,10),
    'reg_alpha':(0,50),
    'reg_lambda':(0,1),
    'eta':(0.01,0.2),
    'n_estimators':(50,1000)
}

# Fit function
def xgb_fit(
    # Dataset
    X_t,y_t,X_v,
    # Parameters
    max_depth,subsample,colsample_bytree,min_child_weight,gamma,reg_alpha,reg_lambda,eta,n_estimators
):
    xgb_params = {
        'max_depth':int(round(max_depth)),
        'subsample':round(subsample,1),
        'colsample_bytree':round(colsample_bytree,1),
        'min_child_weight':int(round(min_child_weight)),
        'gamma':int(round(gamma)),
        'reg_alpha':int(round(reg_alpha,-1)),
        'reg_lambda':round(reg_lambda,1),
        'eta':round(eta,2),
        'n_estimators':int(round(n_estimators,-1))
    }
    
    # Class weight
    pos_weight = y_t.value_counts()[0] / y_t.value_counts()[1]
    
    xgb_model = XGBClassifier(**xgb_params,
                              eval_metric='aucpr',
                              scale_pos_weight = pos_weight,
                              random_state=156).fit(X_t,y_t)
    
    y_v_prob = xgb_model.predict_proba(X_v)[:,1]
    return y_v_prob

# Objective function
def xgb_objf_auprc(
    # Parameters
    max_depth,subsample,colsample_bytree,min_child_weight,gamma,reg_alpha,reg_lambda,eta,n_estimators
):
    # Fold CV
    skfold = StratifiedKFold(5,shuffle=True,random_state=156)
    # Initialize
    score = 0
    lst = []
    # CV
    for trn_idx,val_idx in skfold.split(X_train,y_train):
        X_trn_cv,X_val_cv = X_train.iloc[trn_idx],X_train.iloc[val_idx]
        y_trn_cv,y_val_cv = y_train.iloc[trn_idx],y_train.iloc[val_idx]
        y_prob_cv = xgb_fit(X_trn_cv,y_trn_cv,X_val_cv,
                            max_depth,subsample,colsample_bytree,min_child_weight,gamma,reg_alpha,reg_lambda,eta,n_estimators)
        lst += [average_precision_score(y_val_cv,y_prob_cv)]
    
    lst_df = pd.Series(lst)
    score_mean = lst_df.mean()
    return score_mean

# Bayesian optimization followed by performance measurement
def xgb_bopt_res(
    # Fit dataset
    X_t,y_t,X_v,y_v,
    # Optimizer spec
    param_space,init_,iter_,objf,verb_): 
    
    # Set optimizer
    boptimizer = BayesianOptimization(f=objf,pbounds=param_space,verbose=verb_,random_state=156)
    
    # Set acquisition function
    acq = UtilityFunction(kind='ei',xi=0.01)
    
    # Run optimizer
    boptimizer.maximize(init_points=init_,n_iter=iter_,
                        acquisition_function=acq)
    
    # Get best parameters
    best_score = boptimizer.max['target']
    best_param = boptimizer.max['params']
    best_param = {
        'max_depth':       int(round(best_param['max_depth'])),
        'subsample':           round(best_param['subsample'],1),
        'colsample_bytree':    round(best_param['colsample_bytree'],1),
        'min_child_weight':int(round(best_param['min_child_weight'])),
        'gamma':           int(round(best_param['gamma'])),
        'reg_alpha':       int(round(best_param['reg_alpha'],-1)),
        'reg_lambda':          round(best_param['reg_lambda'],1),
        'eta':                 round(best_param['eta'],2),
        'n_estimators':    int(round(best_param['n_estimators'],-1))
    }
    
    # Weight
    pos_weight = y_t.value_counts()[0] / y_t.value_counts()[1]
    
    # Set best model
    tuned_model = XGBClassifier(**best_param,
                                eval_metric='aucpr',
                                scale_pos_weight = pos_weight,
                                random_state=156)

    # Fit
    tuned_model.fit(X_t,y_t)
    tuned_prob = tuned_model.predict_proba(X_t)[:,1]

    # Determine best classification cutoff
    tuned_cutoffs = np.linspace(0.001, 0.999, 999)
    tuned_f1_scores = []
    tuned_abs_delta = []
    for tuned_cutoff in tuned_cutoffs:
        tuned_pred = (tuned_prob > tuned_cutoff).astype(int)
        f1 = f1_score(y_t,tuned_pred)
        tuned_f1_scores.append(f1)
        tuned_abs_delta.append(np.abs(tuned_cutoff-f1/2))
    tuned_best_cutoff = tuned_cutoffs[np.argmin(tuned_abs_delta)]
    
    perf_df_tuned = fitted_model_performance(tuned_model,X_v,y_v,tuned_best_cutoff)
    
    # Calibration
    skfold = StratifiedKFold(5,shuffle=True,random_state=156)
    calib_model = CalibratedClassifierCV(tuned_model, method='sigmoid', cv=skfold, ensemble=False, n_jobs=-1)
    calib_model.fit(X_t,y_t)
    calib_prob = calib_model.predict_proba(X_t)[:,1]
    
    calib_cutoffs = np.linspace(0.001, 0.999, 999)
    calib_f1_scores = []
    calib_abs_delta = []
    for calib_cutoff in calib_cutoffs:
        calib_pred = (calib_prob > calib_cutoff).astype(int)
        f1 = f1_score(y_t,calib_pred)
        calib_f1_scores.append(f1)
        calib_abs_delta.append(np.abs(calib_cutoff-f1/2))
    calib_best_cutoff = calib_cutoffs[np.argmin(calib_abs_delta)]
    
    perf_df_calib = fitted_model_performance(calib_model,X_v,y_v,calib_best_cutoff)
    
    return best_score,best_param,tuned_model,perf_df_tuned,calib_model,perf_df_calib

##############################################
###############  Import data #################
##############################################

# Set directory
X_train_dir = '' # X_train directory
y_train_dir = '' # y_train directory
X_test_dir = '' # X_test directory
y_test_dir = '' # y_test directory
sigfig_dir  = '' # Significant figure directory
meta_dict_dir = '' # Metadata directory
elim_rferf_dir = '' # Elimination order directory

# Import data
X_train_original = pd.read_csv(X_train_dir)
y_train = pd.read_csv(y_train_dir)
X_test_original = pd.read_csv(X_train_dir)
y_test = pd.read_csv(y_train_dir)
feat_ord = pd.read_excel(elim_rferf_dir)

##############################################
###### Get vector with selected features #####
##############################################
featnum = # Set the number of features
colname = '' # Set the column name indicating the feature column name in the feat_ord dataset

# Get feature list
featlst = feat_ord.loc[0:featnum-1,colname].tolist()

# Select columns
X_train = X_train_original[featlst]
X_test = X_test_original[featlst]

###############################################
############  Execute model fitting ###########
###############################################

sco,par,tun,tun_perf,cal,cal_perf = xgb_bopt_res(
    X_train,y_train,X_test,y_test,
    xgb_param_space,3,50,xgb_objf_auprc,2)

##############################################
###############  Save results ################
##############################################

# Set directory
savedir = '' # Set directory
# Save optimized hyperparameters
par_df = pd.DataFrame([par]).T
par_df.to_excel(f'{savedir}/hyperparameter.xlsx')
# Save model
joblib.dump(tun,f'{savedir}/fitted_model.pkl')
# Save model performance
tun_perf.to_excel(f'{savedir}/performance.xlsx')
































