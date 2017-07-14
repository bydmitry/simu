#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:05:33 2017

@author: bychkov
"""

import os
from datetime import datetime
from collections import OrderedDict

import plotly
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


#===================================================================================
# Survival data generator class.
#===================================================================================

class SurvMaker(object):
    # ------------------------------------------------------------------------------
    def __init__(self):
        pass
    # ------------------------------------------------------------------------------
    def linear_H(self, X, num_smpl, num_covs, num_fact):
        """
        Computes a linear scalar risk of over covariates in X.
        
        """
        # Vector of coefficients:
        betas = np.zeros((num_covs,), dtype = 'float32')
        betas[0:num_fact] = range(1,num_fact + 1)
        
        # Linear Combinations of Coefficients and Covariates
        risk = np.dot(X, betas).astype('float32')
        return risk
    # ------------------------------------------------------------------------------
    def gaussian_H(self, X, num_smpl, num_covs, num_fact,
                      c=0.0, rad=0.5, max_hr=2.0):
        """
        Computes Gaussian function.
        
        """
        z = np.square((X-c), dtype = 'float32')
        z = np.sum(z[:,0:num_fact], axis = -1)
    
        risk = max_hr * (np.exp( -(z) / (2 * rad ** 2) ))   
        risk = risk - (max_hr/2.0) 
        risk = risk.astype('float32')
        return risk
    # ------------------------------------------------------------------------------
    def generate_samples(self, num_smpl=1000, num_covs=5, num_fact=2, censor_rate = 0.3,
                         mean_surv_time = 60, max_observ_period = 180, method='linear'):
        
        # Baseline data:
        data = np.random.uniform(low = -1, high = 1, 
                                 size = (num_smpl, num_covs)).astype('float32')
        
        # Model risk scores:
        if method == 'linear':
            risk = self.linear_H(data, num_smpl, num_covs, num_fact)

        elif method == 'gaussian':
            risk = self.gaussian_H(data, num_smpl, num_covs, num_fact)
            
        # Normalize risk scores:
        risk = risk - np.mean(risk)
        
        # Generate time durations from Exponential distribution; See:
        # Peter C Austin. Generating survival times to simulate cox proportional
        # hazards models with time-varying covariates. Statistics in medicine,
        # 31(29):3946-3958, 2012.
        
        #  1 / lambda = mean time [ only for exponential! ]
        lmb = 1.0 / mean_surv_time
        
        failure_times = np.zeros((num_smpl), dtype='float32')
        censoring = np.zeros((num_smpl), dtype='int8')
        U = np.random.uniform(low=0, high=1, size=num_smpl)
        for i in range( num_smpl ):
            # TODO: Sort out how this compares to simple Exponential distr?!
            # failure_times[i] = np.random.exponential(lmb / (np.exp(risk[i])) ) 
            failure_times[i]  = -( (np.log(U[i])) / (lmb * np.exp(risk[i])) ) 
            censoring[i]      = np.random.binomial(1, (1-censor_rate), 1)
        
        # Clean-up:
        failure_times = np.round(failure_times,2)
        del U
                
        # Apply non-informative right-cencoring:
        # censoring [ 0: censored 1: uncensored ]
        observed_times  = np.copy(failure_times)        
        observed_times[ failure_times > max_observ_period ] = max_observ_period
        censoring[ failure_times > max_observ_period ] = 0
        
        ids = np.arange(1, num_smpl+1, dtype='int32')
        
        # Collect the data into dictionary:
        samples = {
            'id' : ids,
            'x'  : data,
            't'  : observed_times,
            'f'  : failure_times,
            'e'  : censoring,
            'h'  : risk,
            'method'         : method,
            'censor_rate'    : censor_rate,
            'mean_surv_time' : mean_surv_time,
            'observ_period'  : max_observ_period
        }
        return samples
    # ------------------------------------------------------------------------------
    def export_to_csv(self, data_dict, file_name=None):
        if file_name is None:
            file_name = 'survival_data' + datetime.now().strftime("-%d%m%y_%H.%M.%S")
        fname = file_name  + '.csv'
        save_to = os.path.join(os.getcwd(),'simulated_data',fname)
        
        d = OrderedDict()
        d['id'] = data_dict['id']
        # Prepare Xs:
        for c in range(data_dict['x'].shape[1]):
            d['x.%s' % (c + 1)] = data_dict['x'][:,c]
        
        d['t'] = data_dict['t']
        d['f'] = data_dict['f']
        d['e'] = data_dict['e']
        d['h'] = data_dict['h']
        
        # Create a data-frame:
        df = pd.DataFrame( data=d )
        
        df.to_csv(save_to, index = False)
        print 'Saved to: %s' % (save_to)
        return None
    # ------------------------------------------------------------------------------
    def export_to_pkl(self, data_dict, file_name=None):
        if file_name is None:
            file_name = 'survival_data' + datetime.now().strftime("-%d%m%y_%H.%M.%S")
        fname = file_name  + '.pkl'
        save_to = os.path.join(os.getcwd(),'simulated_data',fname)
        pickle.dump(data_dict, open(save_to,'wb'))
        print 'Saved to: %s' % (save_to)
        return None

    # ------------------------------------------------------------------------------
    def unpickle_data(self, file_name):
        fname = os.path.join('simulated_data',file_name)
        data  = pickle.load( open( fname, "rb" ) )
        return data    
    
#===================================================================================
# Done.
#===================================================================================