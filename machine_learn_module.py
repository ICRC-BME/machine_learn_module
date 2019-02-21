#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:08:40 2019

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors

from scipy.stats import norm, ranksums

# Local imports

# %% Class definition

class iEEG_machine_learn:
    """
    Class for for training and classification of iEEG feature data.
    
    Attributes: 
        df (DataFrame): Pandas DataFrame with feature data
        pat_id_col (str): Column with patient ID
        pos_result_col (str): Column marking patients with positive result.
        target (str): Column with target for classification
        feature_cols (list or dict): List or group dict of EEG feature columns
        use_pca (bool): Whether to use PCA, default=False
    """
    def __init__(self, df, pat_id_col, pos_result_col, target, feature_cols, 
                 use_pca=False):
        
        self.df = df
        self.pat_id_col = pat_id_col
        
        self.feature_cols = feature_cols
        self.use_pca = use_pca
        
        self._target = target
        
        self.pos_result_col = pos_result_col
        
        self.sel_features_map = None
        self.POS_X = None
        self.POS_y = None
        self.NEG_X = None
        self.NEG_y = None
        
        self.pos_pat_splits = None
        self.neg_pat_splits = None
        self.all_pat_splits = None
                
    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, target):
        self._target = target
        
    @property
    def pos_result_col(self):
        return self._pos_result_col
    
    @pos_result_col.setter
    def pos_result_col(self, pos_result_col):
        self._pos_result_col = pos_result_col
        
        self._pos_patients_df = self.df.loc[self.df[pos_result_col] == 1].copy()
        self._pos_patients_df.reset_index(inplace=True, drop=True)
        self._check_target_presence(self._pos_patients_df)
        
        self._neg_patients_df = self.df.loc[self.df[pos_result_col] != 1].copy()
        self._neg_patients_df.reset_index(inplace=True, drop=True)
        self._check_target_presence(self._neg_patients_df)
        
    # ----- Main funcions -----
        
    def create_outcome_sets(self):
        """
        Function to create sets for good and bad outcomes.
        """
        if isinstance(self.feature_cols, dict):
            features_to_array = []
            for key, value in self.feature_cols.items():
                features_to_array += value
        elif isinstance(self.feature_cols, list):
            features_to_array = self.feature_cols
            
        # TODO: call feature normalization
            
        features_to_array.sort()
            
        feature_mapper = DataFrameMapper([(features_to_array,None)])
        target_mapper = DataFrameMapper([(self.target,None)])

        # Positive result set
        self.POS_X = feature_mapper.fit_transform(self._pos_patients_df)
        y = target_mapper.fit_transform(self._pos_patients_df)
        y = y.astype('int8')
        self.POS_y = y.ravel()
        
        # Negative results set
        self.NEG_X = feature_mapper.fit_transform(self._neg_patients_df)
        y = target_mapper.fit_transform(self._neg_patients_df)
        y = y.astype('int8')
        self.NEG_y = y.ravel()
        
        # All result set
        self.ALL_X = feature_mapper.fit_transform(self.df)
        y = target_mapper.fit_transform(self.df)
        y = y.astype('int8')
        self.ALL_y = y.ravel()
        
    def select_features(self, threshold=3, method='modz'):
        """
        Parameters:
        -----------
            threshold - threshold to determine the best features
            method - method to use for statistic calculation, options:
                'z' - zscore calculation and thresholding
                'modz' - modified z score (Iglewicz & Hoaglin 1993)
        
        """
        if isinstance(self.feature_cols, dict):
            features_to_array = []
            for key, value in self.feature_cols.items():
                features_to_array += value
        elif isinstance(self.feature_cols, list):
            features_to_array = self.feature_cols
        
        features_to_array.sort()
        
        stat_vals = pd.DataFrame(columns=('feature', 'rank', 'rank_p_val',
                                          'F_score', 'F_p_val','fisher_score'))
        
        X, y = self.POS_X, self.POS_y
        
        F, fp = f_classif(X, y)
        
        for i in range(X.shape[1]):
            s, p = ranksums(X[y, i],
                            X[~y, i])
            
            stat_vals.loc[i] = [features_to_array[i], s, p, F[i], fp[i], None]
        
        stat_vals.loc[stat_vals['F_score'].isnull(), 'F_score'] = 0
        
        
        # Determine the best features - whole feature set
        if isinstance(self.feature_cols, list):
            sel_feat_bool = (stat_vals['F_score'] - stat_vals['F_score'].mean()) / stat_vals['F_score'].std() > threshold
            selected_features = list(stat_vals.loc[sel_feat_bool, 'feature'].values)
        
        # Determine the best features - feature sets
        elif isinstance(self.feature_cols, dict):
            selected_features = []
            for i in self.feature_cols.items():
                floc = stat_vals.feature.isin(i[1])
                out_arr = self._is_outlier(stat_vals.loc[floc, 'F_score'], thresh=threshold)
                sub = stat_vals.loc[floc]
                selected_features += list(stat_vals.loc[sub.index[out_arr], 'feature'].values)
                
                # Cross-compare the selected features with the rest - as per suggestion by Ben - sohuld eliminate the cases when all features are either useful or not useful
                _,p = ranksums(sub.loc[out_arr, 'F_score'],
                               sub.loc[~out_arr, 'F_score'])
                
                if p>0.05:
                    raise RuntimeError('Selected features are too similar to the nonselected features!!!')
                            
        self.sel_features_map = stat_vals['feature'].isin(selected_features)
        
        print('Selected features are', selected_features)
        
    def create_leave_x_out(self, n_lo=1):
        """
        Function to create leave x out splits
        
        Parameters:
        -----------
        n_lo - number of leave out, default=1
        """
        
        pats = list(self._pos_patients_df.pat.unique())
        self.pos_pat_splits = []
        for i in range(len(pats) // n_lo):
            curr_i = i*n_lo
            lo_pats = pats[curr_i: curr_i+n_lo]
            test_split = np.array(self._pos_patients_df.loc[self._pos_patients_df.pat.isin(lo_pats)].index)
            train_split = np.array(self._pos_patients_df.loc[~self._pos_patients_df.pat.isin(lo_pats)].index)
            
            self.pos_pat_splits.append([train_split,test_split])
        
        pats = list(self._neg_patients_df.pat.unique())
        self.neg_pat_splits = []
        for i in range(len(pats) // n_lo):
            curr_i = i*n_lo
            lo_pats = pats[curr_i: curr_i+n_lo]
            test_split = np.array(self._neg_patients_df.loc[self._neg_patients_df.pat.isin(lo_pats)].index)
            train_split = np.array(self._neg_patients_df.loc[~self._neg_patients_df.pat.isin(lo_pats)].index)
            
            self.neg_pat_splits.append([train_split,test_split])
            
        pats = list(self.df.pat.unique())
        self.all_pat_splits = []
        for i in range(len(pats) // n_lo):
            curr_i = i*n_lo
            lo_pats = pats[curr_i: curr_i+n_lo]
            test_split = np.array(self.df.loc[self.df.pat.isin(lo_pats)].index)
            train_split = np.array(self.df.loc[~self.df.pat.isin(lo_pats)].index)
            
            self.all_pat_splits.append([train_split,test_split])
            
    def perform_param_sweep(self, classif=SVC, cases='positive',
                            params=[{'kernel': 'linear', 'C': 0.1}]):
        """
        Function to do a parameter sweep to find the best settings.
        
        Parameters:
        -----------
        classif - ScikitLearn classifier to use
        cases - sweep group - all / positive / negative, def: positive
        params - list of parameter dictionaries for each run
        """
        
        param_list = [i for s in [list(x.keys()) for x in params] for i in s]
        param_list = tuple(set(param_list))
        sweep_df = pd.DataFrame(columns=(param_list
                                         +('AUC','accuracy','f1_score')))
        param_i = 0
        
        # Create sets
        if cases == 'all':
            X = self.ALL_X[:, self.sel_features_map]
            y = self.ALL_y
            pats_df = self.df
            splits = self.all_pat_splits
        elif cases == 'positive':
            X = self.POS_X[:, self.sel_features_map]
            y = self.POS_y
        elif cases == 'negative':
            X = self.NEG_X[:, self.sel_features_map]
            y = self.NEG_y
            pats_df = self._neg_patients_df
            splits = self.neg_pat_splits
        else:
            raise ValueError('Cases can be "all" / "positive" / "negative"')
        
        for param in params:
            clf = classif(**param)
            
            mean_tpr = np.zeros(100)
            mean_fpr = np.linspace(0, 1, 100)
                
            TP = FP = TN = FN = 0
            pat_cnt = 0
            auc_df = pd.DataFrame(columns=('pat','AUC','accuracy','f1_score'))
            i = 0
            
            print(param)
            
            for pat,split in zip(pats_df.pat.unique(), splits):
                
                pat_cnt += 1
                
                train_idx,test_idx = split
                
                # Do leave one out split
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                if self.use_pca:
                    n_comp = 2
                    if n_comp > len(self.sel_features_map):
                        n_comp = len(self.sel_features_map)
                    
                    pca = PCA(n_components=n_comp,
                              whiten=True,
                              svd_solver='full')
                    X_train = pca.fit_transform(X_train[:,self.sel_features_map])
                
                # Fit the model
                clf.fit(X_train, y_train)
                        
                if self.use_pca:
                    X_test = pca.fit_transform(X_test[:,self.sel_features_map])
                
                # Classify the one left out
                pred = clf.predict(X_test)
                
                # Calcluate probabilities
                probas = clf.predict_proba(X_test)
                
                # Overall ROC ROC
                TP += np.sum((y_test == 1) & (pred ==1))
                FP += np.sum((y_test == 0) & (pred ==1))
                TN += np.sum((y_test == 0) & (pred ==0))
                FN += np.sum((y_test == 1) & (pred ==0))
                
                # Calculate ROC
                fpr, tpr, thresholds = roc_curve(y_test, probas[:,1])
                
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                if np.isnan(mean_tpr[1]):
                    print('Nan now!! msel '+str(pat))
                    break
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                
                # Calculate accuracy
                accu = accuracy_score(y_test, pred)
                
                # Calculate f1
                f1 = f1_score(y_test, pred)
                
                auc_df.loc[i] = [pat, roc_auc, accu, f1]
                i += 1
                
                fpr = np.concatenate([[0.],fpr,[1.]]) # Ensure the end points
                tpr = np.concatenate([[0.],tpr,[1.]]) # Ensure the end points
                        
            # Calculate mean ROC    
            mean_tpr /= len(self._pos_patients_df.pat.unique())
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            
            param_keys= list(param.keys())
            sweep_df.loc[param_i, param_keys+['AUC',
                                              'accuracy',
                                              'f1_score']] = [param[x] for x in param_keys]+[mean_auc, auc_df.accuracy.mean(), auc_df.f1_score.mean()]
    
            param_i += 1
    
        # Get the best performing parameters
        self.best_perfrom_params = sweep_df.loc[sweep_df.AUC.astype('float').idxmax(),
                                                param_keys]
        print('Best performance parameters')
        print('----------')
        print(self.best_perfrom_params)
        
        return self.best_perfrom_params
    
    def run_lo_crossval(self, trained_clf=None, cases='positive',
                        plot=False, axes=None):
        """
        Function to run leave out crossvalidation.
        
        Parameters:
        -----------
        clf - trained classifier, default None - classifier is retrained in
        each crossvalidation step
        cases - crossvalidate group - all / positive / negative, def: positive
        plot - whether to plot the results, default False
        axes - axes can be provided if plot=True
        
        Returns:
        --------
        mean_auc - mean area under the curve
        p - Hanley & McNeil test
        pat_cnt - number of patients ran in crossvalidation
        """
        
        
        if plot:
            if axes:
                plt.sca(axes)
            else:
                f, ax_arr = plt.subplots(1, 1, figsize=[9, 8])
        
        if trained_clf is None:
            clf = SVC(**self.best_perfrom_params.to_dict())
        else:
            clf = trained_clf
        mean_tpr = np.zeros(100)
        mean_fpr = np.linspace(0, 1, 100)
        
        TP = FP = TN = FN = 0
        pat_cnt = 0
        auc_df = pd.DataFrame(columns=('pat','AUC','accuracy'))
        i = 0
        
        
        # Create sets
        if cases == 'all':
            X = self.ALL_X[:, self.sel_features_map]
            y = self.ALL_y
            pats_df = self.df
            splits = self.all_pat_splits
        elif cases == 'positive':
            X = self.POS_X[:, self.sel_features_map]
            y = self.POS_y
            pats_df = self._pos_patients_df
            splits = self.pos_pat_splits
        elif cases == 'negative':
            X = self.NEG_X[:, self.sel_features_map]
            y = self.NEG_y
            pats_df = self._neg_patients_df
            splits = self.neg_pat_splits
        else:
            raise ValueError('Cases can be "all" / "positive" / "negative"')
    
        for pat,split in zip(pats_df.pat.unique(), splits):
            
            pat_cnt += 1
            print('Patient',pat,'number',pat_cnt)

            train_idx,test_idx = split
            
            # Do leave one out split
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if self.use_pca:
                n_comp = 2
                if n_comp > len(self.sel_features_map):
                    n_comp = len(self.sel_features_map)
                
                pca = PCA(n_components=n_comp,
                          whiten=True,
                          svd_solver='full')
                X_train = pca.fit_transform(X_train[:,self.sel_features_map])
            
            # Fit the model
            if trained_clf is None:
                clf.fit(X_train, y_train)
                    
            if self.use_pca:
                X_test = pca.fit_transform(X_test[:,self.sel_features_map])
                
            # Classify the one left out
            loo_pred = clf.predict(X_test)
            
            # Calcluate probabilities
            probas = clf.predict_proba(X_test)
            
            # Assign back to the dataframe
            pats_df.loc[test_idx,'probability'] = probas[:, 1]
            pats_df.loc[test_idx,'pred'] = loo_pred
            
            # Kneighbors based on radius
            pac_mni_df = pats_df.loc[test_idx,['channel_name','MNI_x', 'MNI_y', 'MNI_z','probability','pred','resected','onset_channel','SOZ_resected', 'structure']]
            class_df = pac_mni_df.loc[(pac_mni_df['pred']==1)]
            dist_df = self._calculate_distances(pac_mni_df.loc[:,['channel_name','MNI_x', 'MNI_y', 'MNI_z']].rename(columns={'MNI_x':'X','MNI_y':'Y','MNI_z':'Z'}))
            labels = self._create_clusters(class_df, dist_df.distance.min()*2)
            class_df.loc[:,'clusters'] = labels
            pats_df.loc[(pats_df['pat']==pat) & (pats_df['pred']==1), 'clu_labels'] = labels
            
            cluster_probas = {}
            if np.all(labels == -1):
                print('No clusters found, assigning highest probability')
                if len(class_df):
                    new_clu_idx = class_df.probability.idxmax()
                else:
                    new_clu_idx = pac_mni_df.probability.idxmax()
            else:
                # Get cluster with highest probability
                curr_cluster = -1
                curr_cluster_proba = 0
                for cluster in np.unique(labels):
                    idcs = class_df.loc[labels == cluster].index.values
                    cluster_proba = class_df.loc[idcs, 'probability'].mean()
                    cluster_probas[cluster] = cluster_proba
                    if cluster_proba > curr_cluster_proba:
                        curr_cluster_proba = cluster_proba
                        curr_cluster = cluster
                
                # get only clusters larger than 2 contacts
                new_clu_idx = class_df.loc[labels == curr_cluster].index.values
        #        new_clu_idx = class_df.loc[db.labels_>=0].index.values
            
            pats_df.loc[test_idx, 'clu_pred'] = 0
            pats_df.loc[new_clu_idx,'clu_pred'] = 1
            
            # Try only the channels with highest probability
            pats_df.loc[test_idx, 'probab_class'] = 0
            pats_df.loc[pats_df.loc[test_idx,'probability'].idxmax(), 'probab_class'] = 1
        
            
            # Patient specific statistics
            pacTP = np.sum((y_test == 1)
                            & (pats_df.loc[test_idx, 'pred'] == 1).values)
            pacFP = np.sum((y_test == 0)
                            & (pats_df.loc[test_idx, 'pred'] == 1).values)
            pacTN = np.sum((y_test == 0)
                            & (pats_df.loc[test_idx, 'pred'] == 0).values)
            pacFN = np.sum((y_test == 1)
                            & (pats_df.loc[test_idx, 'pred'] == 0).values)
            
            pac_sensitivity = pacTP / (pacTP + pacFN)
            pac_specificity = pacTN / (pacTN + pacFP)
            pac_pos_val = pacTP / (pacTP + pacFP)
            pac_neg_val = pacTN / (pacTN + pacFN)
            
            print('----------')
            print('Patient statisticts')
            print('TP =',pacTP)
            print('FP =',pacFP)
            print('TN =',pacTN)
            print('FN =',pacFN)
            print('Patient sensitivity =',pac_sensitivity)
            print('Patient specificity =',pac_specificity)
            print('Patient negative pval =',pac_neg_val)
            print('Patient positive pval =',pac_pos_val)
            print('----------')
            # Overall ROC ROC
            TP += pacTP
            FP += pacFP
            TN += pacTN
            FN += pacFN
            
            fpr, tpr, thresholds = roc_curve(y_test, probas[:,1])
            
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            if np.isnan(mean_tpr[1]):
                print('Nan now!! patient '+str(pat))
                break
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            
            # Calculate accuracy
            accu = accuracy_score(y_test, loo_pred)
            
            auc_df.loc[i] = [pat, roc_auc, accu]
            i += 1
            
            fpr = np.concatenate([[0.],fpr,[1.]]) # Ensure the end points
            tpr = np.concatenate([[0.],tpr,[1.]]) # Ensure the end points
            
            if plot:
                plt.plot(fpr, tpr, lw=1, color='k',alpha=0.3)
                
            
        # Calculate mean ROC    
        mean_tpr /= len(pats_df.pat.unique())
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        # Do the Hanley & McNeil test to test if we are significantly above chance
        N_positives = np.sum(y == 1)
        N_negatives = np.sum(y == 0)
        p = self._hanley_mcneil_test(N_positives,N_negatives,mean_auc,
                                     N_positives,N_negatives,0.5)
        
        print("One tail p value compared to AUC=0.5 is "+str(p))
        print('Mean accuracy is '+str(auc_df['accuracy'].mean()))
        
        # Plot the break point
        ROC_coord = np.column_stack([mean_fpr,mean_tpr])
        
        dist_min = 1
        thresh_i = 0
        for i,row in enumerate(ROC_coord):
            dist = np.linalg.norm(np.array([0,1])-row)
            if dist < dist_min:
                dist_min = dist
                thresh_i = i
                
        
        # Plot overall value
        overall_sensitivity = TP / (TP + FN)
        overall_specificity = TN / (TN + FP)
        overall_pos_val = TP / (TP + FP)
        overall_neg_val = TN / (TN + FN)
        
        print('Overall sensitivity and specificity: ',overall_sensitivity, overall_specificity)
        print('Overall positive predictive value:',overall_pos_val)
        print('Overall negative predictive value:',overall_neg_val)
        
        if plot:
        
            plt.plot(mean_fpr[thresh_i], mean_tpr[thresh_i],'r*',
                     label='Break point; Sensitivity=%0.3f; Specificity=%0.3f' % (mean_tpr[thresh_i], 1-mean_fpr[thresh_i]))
            
            # Plot the average ROC
            plt.plot(mean_fpr, mean_tpr, color='b',
                     label='Mean ROC (area = %0.3f)' % mean_auc, lw=2)
            
            # Plot chance
            plt.plot([0,1],[0,1],'r--',lw=2)
            
            # Plot descriptors
            plt.ylabel('TPR (Sensitivty)')
            plt.xlabel('FPR (1 - Specificity)')
            
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            plt.legend(loc=4)
            plt.title('Good outcome patients (N = '+str(pat_cnt)+')')
                
        return mean_auc, p, pat_cnt
    
    # ----- Helper functions -----
        
    def _discretize_column(self, column, new_column_name=None):
        """
        Function to discretize one column of the dataframe.
        """
        # Discretize the structure field
        temp_df = pd.DataFrame(columns=[column],
                               data=self.df[column].unique())
        temp_df.reset_index(inplace=True)
        temp_df.rename(columns={'index':'coded_structure'},inplace=True)
        
        self.df = pd.merge(self.df, temp_df, on=column, how='left')
            
    def _binarize_column(self, column, classes, **kwargs):
        """
        Function to binarize one column of the dataframe.
        """
        self.df.loc[:, column] = label_binarize(self.df[column], classes
                                                **kwargs)[:, 0]
    
    def _check_target_presence(self, df):
        """
        Function to check if the target is pesent in each patient.
        """
        for pat in df[self.pat_id_col].unique():
            if sum(df.loc[df[self.pat_id_col] == pat, self.target]) == 0:
                RuntimeWarning('Patient '+str(pat)+' - missing target')
                df = df[~(df[self.pat_id_col] == pat)]
        df.reset_index(inplace=True, drop=True)
    
    def _normalize_features(self):
        """
        Function to normalize computed features by standard scaling.
        """
        scaler = StandardScaler()
        
        if isinstance(self.feature_cols, dict):
            features_to_norm = []
            for key, value in self.feature_cols.items():
                features_to_norm += value
        elif isinstance(self.feature_cols, list):
            features_to_norm = self.feature_cols
        
        features_to_norm.sort()
        
        # Fit and scale th good outcome data
        for pat in self._pos_patients_df[self.pat_id_col].unique():
            self._pos_patients_df.loc[self._pos_patients_df[self.pat_id_col]==pat, features_to_norm] = \
                scaler.fit_transform(self._pos_patients_df.loc[self._pos_patients_df[self.pat_id_col]==pat, features_to_norm]) 
        
        # Scale the bad outcome data based on good outcomes
        for pat in self._neg_patients_df[self.pat_id_col].unique():
            self._neg_patients_df.loc[self._neg_patients_df[self.pat_id_col]==pat, features_to_norm] = \
                scaler.transform(self._neg_patients_df.loc[self._neg_patients_df[self.pat_id_col]==pat, features_to_norm])

    def _point_distance(self, x,y):
        """
        Calculates eucleidian distance between two points.
        
        Parameters:
        ----------
        x - 3 point list or numpy array (x,y,z)\n
        y - 3 point list or numpy array (x,y,z)\n
        
        Returns:
        --------
        dist - calculated distance\n
        """
        
        x = np.array(x)
        y = np.array(y)
        
        dist = np.linalg.norm(x-y)
        
        return dist
        
        # TODO: MNI2Talairach, Talairach2MNI - we would need the affine transform matrices
    
    def _calculate_distances(self, MNI_df):
        """
        Function that takes pandas dataframe columns=(channel_name,X,Y,Z) and \n
        and returns the dataframe with distances between contacs.
        
        Parameters:
        ----------
        MNI_df - dataframe columns=('channel_name','X','Y','Z')
        
        Returns:
        -------
        out_df - dataframe columns=('ch1','ch2','distance')
        """
        df_loc = 0
        out_df = pd.DataFrame(columns=('ch1','ch2','distance'))
        for ch1 in MNI_df.iterrows():
            ch1_name = ch1[1].channel_name
            ch1_coordinates = np.array([ch1[1].X, ch1[1].Y, ch1[1].Z])
            for ch2 in MNI_df.iterrows():
                ch2_name = ch2[1].channel_name
                ch2_coordinates = np.array([ch2[1].X, ch2[1].Y, ch2[1].Z])
                
                # Calculate eucleidian distance
                dist = self._point_distance(ch1_coordinates, ch2_coordinates)
                out_df.loc[df_loc] = [ch1_name, ch2_name, dist]
                df_loc += 1
            
        # Get rid of zero distances (same contacts)
        out_df = out_df[out_df.distance != 0]
        out_df.reset_index(inplace=True, drop=True)
        
        return out_df

    def _create_clusters(self, class_df, dist_thresh):
        """
        Function to create clusters from spatial information.
        
        Parameters:
        -----------
            class_df - classification dataframe
            dist_thresh - distance threshold for creating clusters
            
        Returns:
        --------
            daatframe with clusters
        """    
        if not len(class_df):
            return np.array([])
        
        only_mni_df = class_df.loc[:,['channel_name','MNI_x','MNI_y','MNI_z']]
        only_mni_df.reset_index(inplace=True, drop=True)
        only_mni_df.rename(columns={'MNI_x':'X','MNI_y':'Y','MNI_z':'Z'},inplace=True)
            
        nbrs = NearestNeighbors(None, radius=dist_thresh, algorithm='brute')
        nbrs.fit(only_mni_df.loc[:,['X', 'Y', 'Z']].values)
        _, indices = nbrs.radius_neighbors(only_mni_df.loc[:,['X', 'Y', 'Z']].values)
        
        only_mni_df['cluster'] = -1
        clu_max = 0
        for i, idx in enumerate(indices):
            if only_mni_df.loc[idx, 'cluster'].max() > -1:
                only_mni_df.loc[idx, 'cluster'] = only_mni_df.loc[idx, 'cluster'].max()
            else:
                only_mni_df.loc[idx, 'cluster'] = clu_max
                clu_max += 1
                
        return only_mni_df.loc[:, 'cluster'].values
        

    # ----- Hanley & McNeil test -----
    
    def _auc_se(self, N_positives,N_negatives,auc,alpha=0.05):
        """
        Helper function fo Hanley-McNeil tets
        """
        
        Q1 = auc / (2-auc)
        Q2 = (2*auc**2) / (1+auc)
        
        auc_var = auc*(1-auc) + (N_positives-1)*(Q1-auc**2) + (N_negatives-1)*(Q2-auc**2)
        auc_var = auc_var / (N_positives*N_negatives)       
        auc_se = np.sqrt(auc_var)
        
        # Confidence interval
        z = norm.ppf(1-alpha/2)
        auc_ci = np.array([auc-z*auc_se,auc+z*auc_se])
        
        return auc_se, auc_ci
    
    
    def _hanley_mcneil_test(self, np_a,nn_a,auc_a,
                           np_b,nn_b,auc_b,
                           one_sided=True):
        """
        Hanley-McNeil test to test significance of ROC differences
        
        Parameters:
        -----------
            np_a - number of positives for ROC "a"
            nn_a - number of negatives for ROC "a"
            auc_a - the value of AUC for ROC "a"
            np_b - number of positives for ROC "b"
            nn_b - number of negatives for ROC "b"
            auc_b - the value of AUC for ROC "b"
            one_sided - whether to use onesided test
            
        Returns:
        --------
            p value
        """
    
        auc_se_a = self._auc_se(nn_a,nn_a,auc_a)[0]
        auc_se_b = self._auc_se(np_b,nn_b,auc_b)[0]
        
        area_diff = auc_a - auc_b
        
        se_diff = np.sqrt((auc_se_a**2) + (auc_se_b**2))
        
        z = area_diff / se_diff
        
        
        abs_z = np.abs(z)
        p2 = norm.pdf(abs_z)
        p1 = p2/2
        
        if one_sided:
            return p1
        else:
            return p2
    
    # ----- Otlier functions -----
    
    def _is_outlier(self, points, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.
    
        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.
    
        Returns:
        --------
            mask : A numobservations-length boolean array.
    
        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
    
        modified_z_score = 0.6745 * diff / med_abs_deviation
    
        return modified_z_score > thresh