#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:28:02 2019

Ing.,Mgr. (MSc.) Jan Cimbálník
Biomedical engineering
International Clinical Research Center
St. Anne's University Hospital in Brno
Czech Republic
&
Mayo systems electrophysiology lab
Mayo Clinic
200 1st St SW
Rochester, MN
United States
"""

# Standard library imports

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize


# Local imports

# %% Load data

path_to_data = 'data.pkl'
svm_df = pd.read_pickle(path_to_data)

# %% Prepare dataframes for descriptive statistcs

roc_res_i = 0
roc_results_df = pd.DataFrame(columns=['target', 'institution',
                                       'AUC_good_outcome','p_good_outcome','N_good_outcome',
                                       'AUC_bad_outcome', 'p_bad_outcome','N_bad_outcome'])
best_performing_params = pd.DataFrame(columns=['taret', 'institution',
                                               'kernel', 'C', 'gamma'])


# %% Presets

use_pca = False

# Threshold to determine the good and bad outcome ptients
bad_o_thresh = 1  # exclusive for bad outcome

# Maximum number of iterations
max_iter = 500000
    
# %% Define results dataframe - will be used to create tables

features = list(svm_df.keys())
features = [x for x in svm_df if any(c.isdigit() for c in x)] + ['pac', 'fac', 'pse',]
features.sort()

hfo_stumps = ['CS', 'LL']
hfo_features = [x for x in features if any(x.startswith(s) for s in hfo_stumps)]
conn_stumps = ['xcorr', 'ren', 'pli', 'phase_sync', 'phase_const', 'lin_corr']
connectivity_features = [x for x in features if any(x.startswith(s) for s in conn_stumps)]
uni_stumps = ['power', 'fac', 'pac', 'pse']
univariate_features = [x for x in features if any(x.startswith(s) for s in uni_stumps)]

feature_groups = {'hfo': hfo_features,
                  'conn': connectivity_features,
                  'univar': univariate_features}
        

feature_results_df = pd.DataFrame(index=features)
    
# %% Loop over targets and institutions
    
# Options are:
    # 'onset_channel' - localization of SOZ
    # 'resected' - localization of resected channels
    # 'SOZ_resected' - localization of rescted channels overlapped with SOZ

target = 'SOZ_resected'

# Options are:
    # 'all' - both datasets
    # 'fnusa' - St.Anne's dataset
    # 'mayo' - Mayo Clinic dataset
inst_subset = 'mayo'
    
# %% Data wrangle

# Take care of institution
if inst_subset != 'all':
    svm_df = svm_df[svm_df.dataset == inst_subset]
    
# Discretize the structure field
sustruct_df = pd.DataFrame(columns=['structure'],data=svm_df.structure.unique())
sustruct_df.reset_index(inplace=True)
sustruct_df.rename(columns={'index':'coded_structure'},inplace=True)

svm_df = pd.merge(svm_df,sustruct_df,on='structure',how='left')


# %% Binarize targets
    
svm_df.loc[:,'pathology'] = label_binarize(svm_df.pathology, ['normal', 'pathologic'])[:, 0]
svm_df.loc[:,'onset_channel'] = label_binarize(svm_df.onset_channel, ['SOZ', 'IZ', 'NON_SOZ'])[:, 0]   
svm_df['SOZ_resected'] = svm_df['resected'] * svm_df['onset_channel']

# This is not needed for FNUSA patients since they are guaranteed to be at least one year from resection

# Set outcomes shoreter than one year to 0
if target != 'onset_channel':
    #svm_df.loc[:,'dat_diff'] = (svm_df.outcome_dt - svm_df.resection_dt)
    svm_df.loc[:,'dat_diff'] = ((svm_df.outcome_dt - svm_df.resection_dt).dt.days) / 365 # Years since surgery
    svm_df.loc[svm_df.dat_diff<1,'outcome'] = np.NAN
    svm_df = svm_df.loc[~svm_df.dat_diff.isnull()] # This removes patients with missing date info and shorter than one year from surgery
    svm_df = svm_df[~svm_df.outcome.isnull()] # Removes patients without outcome

# %% Remove patients without targets

no_target_pats = []
for pat in svm_df.pat.unique():
    has_nans = sum(svm_df.loc[svm_df.pat == pat,target].isnull()) != 0
    has_all_nans = len(svm_df.loc[svm_df.pat == pat]) == sum(svm_df.loc[svm_df.pat == pat,target].isnull())
    has_no_target = sum(svm_df.loc[svm_df.pat == pat,target]) == 0
    if (has_nans and has_all_nans) or has_no_target:
        print('Patient '+str(pat)+' does not have target.')
        svm_df = svm_df[~(svm_df.pat == pat)]
svm_df.reset_index(inplace=True,drop=True)

# %% Binarize positive X negative cases

svm_df.loc[svm_df.outcome > 1, 'outcome'] = 0

# %% Get some numbers for descriptive statistics

no_res_cnt = 0
res_cnt = 0

for pat in svm_df.pat.unique():
    if len(svm_df.loc[(~svm_df.resected.isnull()) & (svm_df.pat==pat)]):
        res_cnt+=1
    else:
        no_res_cnt+=1
        
# %% Bring in machine learning module

from machine_learn_module import iEEG_machine_learn 
from sklearn.svm import SVC

iml = iEEG_machine_learn(svm_df, 'pat', 'outcome', target, feature_groups)

iml._normalize_features()
iml.create_outcome_sets()
iml.select_features()
iml.create_leave_x_out()

# Create sets for parameter sweep
params = []
kernels = ['linear', 'rbf']
for kernel in kernels:
            
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]

    if kernel == 'linear':
        gammas = ['auto']
    else:
        gammas = [0.001, 0.01, 0.1, 1, 10, 100]
        
    for C in Cs:
        for gamma in gammas:
        
            params.append({'kernel': kernel,
                           'C': C,
                           'gamma': gamma,
                           'class_weight': 'balanced',
                           'probability': True,
                           'max_iter': 500000})

best_params = iml.perform_param_sweep(params=params)

f, ax_arr = plt.subplots(1, 2, figsize=[18, 8])
mean_auc, p, pat_cnt = iml.run_lo_crossval(plot=True, axes=ax_arr[0])

xclf = SVC(**best_params.to_dict())
xclf.fit(iml.POS_X[:, iml.sel_features_map], iml.POS_y)

iml.run_lo_crossval(trained_clf=xclf, cases='negative',
                    plot=True, axes=ax_arr[1])