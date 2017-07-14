#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:06:37 2017

@author: bychkov
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import plotly
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py 
py.init_notebook_mode()
import plotly.graph_objs as go


#----------------------------------------------------------------------
# Settings:
#----------------------------------------------------------------------

# Number of samples:
NS = 1000
# Total number of covariates:
NC = 8
# Number of factors (relevant features):
NF = 2


#----------------------------------------------------------------------
# Simulate:
#----------------------------------------------------------------------

def linear_risk(X):
    """
    Computes a linear scalar risk over covariates in X.
    """
    # Vector of coefficients:
    betas = np.zeros((NC,))
    betas[0:NF] = range(1,NF + 1)
    
    # Linear Combinations of Coefficients and Covariates
    risk = np.dot(X, betas)
    return risk

def gaussian_risk(X, c=0.0, rad=0.5, max_hr=2.0):
    """
    Computes Gaussian function.
    """
    z = np.square((X-c))
    z = np.sum(z[:,0:NF], axis = -1)

    risk = max_hr * (np.exp( -(z) / (2 * rad ** 2) ))    
    return risk


def generate_data():
    pass


# Baseline data:
data = np.random.uniform(low = -1, high = 1, size = (NS,NC))

# 


# Center the risk:
risk = risk - np.mean(risk)

# Generate time of death:
# From exponential:
death_time = np.zeros((NS,1))
T = np.zeros((NS,1))
lmb = 0.5
for i in range(NS):
    death_time[i] = np.random.exponential(1 / (lmb*np.exp(risk[i])) ) 
    T[i] = -( (np.log(np.random.uniform(low = 0, high = 1)))/(lmb * np.exp(risk[i])) )        
    


plt.hist(T,55)
plt.hist(death_time,55)

print np.mean(T)
print np.mean(death_time)

#----------------------------------------------------------------------
# SeepSurv:
#----------------------------------------------------------------------
import lasagne
import deepsurv as DeepSurv

simulator = DeepSurv.datasets.SimulatedData(hr_ratio=2)

train_set = simulator.generate_data(N = 3000, method='linear')
valid_set = simulator.generate_data(N = 1000, method='linear')
test_set  = simulator.generate_data(N = 1000, method='linear')


model = DeepSurv.DeepSurv(n_in = 10,
                  learning_rate = 0.1,
                  hidden_layers_sizes = list((3,3)))

log = model.train(train_set, valid_set, n_epochs=30)

model.get_concordance_index(**test_set)
DeepSurv.plot_log(log)

model.plot_risk_surface(test_set['x'])

#==============================================================================
# Done.
#==============================================================================

file_name = 'Loss.png'

t  = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2*np.pi*t)
y2 = np.sin(4*np.pi*t)
y3 = np.sin(10*np.pi*t)

# Make a plot:
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
lns1 = ax1.plot(t, y1, label='y1', color='#348ABD')
lns2 = ax2.plot(t, y2, label='y2', color='#8EBA42')
lns3 = ax3.plot(t, y3, label='y3', color='#BA4252')
ax1.set_ylabel('y1'); ax2.set_ylabel('y2'); ax3.set_ylabel('y3');
            
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper center', ncol=3)
plotly_fig = tools.mpl_to_plotly(fig)
py.plot(plotly_fig)

plt.savefig(os.path.join(self.builder.wrk_dir, file_name))
plt.close()

#==============================================================================
# Done.
#==============================================================================

import plotly as py
import plotly.graph_objs as go
from plotly import tools
import numpy as np

left_trace = go.Scatter(x = np.random.randn(1000), y = np.random.randn(1000), yaxis = "y1", mode = "markers")
right_traces = []
right_traces.append(go.Scatter(x = np.random.randn(1000), y = np.random.randn(1000), yaxis = "y2", mode = "markers"))
right_traces.append(go.Scatter(x = np.random.randn(1000) * 10, y = np.random.randn(1000) * 10, yaxis = "y3", mode = "markers"))

fig = tools.make_subplots(rows = 1, cols = 2)
fig.append_trace(left_trace, 1, 1)
for trace in right_traces:
  yaxis = trace["yaxis"] # Store the yaxis
  fig.append_trace(trace, 1, 2)
  fig["data"][-1].update(yaxis = yaxis) # Update the appended trace with the yaxis

fig["layout"]["yaxis1"].update(range = [0, 3], anchor = "x1", side = "left")
fig["layout"]["yaxis2"].update(range = [0, 3], anchor = "x2", side = "left")
fig["layout"]["yaxis3"].update(range = [0, 30], anchor = "x2", side = "right", overlaying = "y2")

div = py.offline.plot(fig, include_plotlyjs=True, output_type='div')



fig['data']
fig['layout']

#==============================================================================
# Done.
#==============================================================================

#
#    def sanity_check(self, data_dict, nbinsx = 35):
#        # General settings:
#        xlim = float( data_dict['observ_period'] * 1.15 )
#        opct = 0.7
#        h_norm = ''
#        
#        fig = tools.make_subplots(
#                rows=1, cols=1, 
#                subplot_titles=('Failure-time distribution'))
#        
#        # Data layers: 
#        fig.append_trace(go.Histogram(
#            x         = data_dict['t'][data_dict['e'] == 1],
#            name      = 'Uncenored Observations',
#            histnorm  = h_norm,
#            autobinx  = False,
#            xbins     = dict(
#                            start = 0.0,
#                            end   = xlim,
#                            size  = xlim/nbinsx ),
#            opacity   = opct
#        ),1,1)
#        fig.append_trace(go.Histogram(
#            x         = data_dict['t'][data_dict['e'] == 0],
#            name      = 'Censored Observations',
#            autobinx  = False,
#            histnorm  = h_norm,
#            xbins     = dict(
#                            start = 0.0,
#                            end   = xlim,
#                            size  = xlim/nbinsx ),
#            opacity   = opct 
#        ),1,1)
#        fig.append_trace(go.Histogram(
#            x         = data_dict['f'],
#            name      = 'Failure Times',
#            autobinx  = False,
#            histnorm  = h_norm,
#            xbins     = dict(
#                            start = 0.0,
#                            end   = xlim,
#                            size  = xlim/nbinsx ),
#            opacity   = opct
#        ),1,1)
#        
#        # Layout settings:
##        layout = go.Layout(
##                title    = 'Failure-time distribution',
##                xaxis    = dict(title='Follow-up duration') 
##        )
#                
#        
#        
#        #fig.append_trace(trace_list, 1, 1)
#        #fig.append_trace(trace_list, 1, 2)
#        
#        fig['layout'].update(
#                title  = 'Simulated Data Statistics', 
#                legend = dict(orientation="h"),
#                height = 400, width=785)
#
#        #fig = go.Figure( data=trace_list, layout=layout )
#        py.iplot(fig, filename='failure_time_distributions')

            fname = os.path.join(self.builder.wrk_dir, 'loss.html')
            
            # Colors:
            cl1 = '#348ABD'; cl2 = '#8EBA42'; cl3='#BA4252';
            pmode = 'markers'
            
            XS = self.history['epoch']
            train_loss = go.Scatter( name='Train loss',
                x = XS, y = self.history['train_loss'], yaxis='y1'
                line = dict(color = cl1), mode = pmode )
            valid_loss = go.Scatter(name='Valid loss', yaxis='y2',
                x = XS, y = self.history['valid_loss'],
                line = dict(color = cl2), mode = pmode )
            test_loss  = go.Scatter(name='Test loss', yaxis='y3',
                x = XS, y = self.history['test_loss'],
                line = dict(color = cl3), mode = pmode )
            fake_trace = go.Scatter(name='Test loss', yaxis='y4',
                x = XS, y = self.history['test_loss'],
                line = dict(color = cl3), mode = pmode )
            
            loss_traces = [train_loss, valid_loss, test_loss]
            
            fig = tools.make_subplots(rows=2, cols=1,
                          shared_xaxes=True, shared_yaxes=False  )
            
            layout = go.Layout(
                title  = 'Training dynamics',
                yaxis  = dict( title='Train loss',
                    titlefont = dict(color=cl1),
                    tickfont  = dict(color=cl1)),
                yaxis2 = dict( title='Valid loss',
                    titlefont = dict(color=cl2),
                    tickfont  = dict(color=cl2),
                    overlaying='y', side='right', position=0.85, anchor ='free'   ),
                yaxis3 = dict( title='Test loss',
                    titlefont = dict(color=cl3),
                    tickfont  = dict(color=cl3),
                    overlaying='y', side='right', position=0.95, anchor ='free'   )
            )
            #fig = go.Figure(data=data, layout=layout)
            

            fig.append_trace(train_loss, 2, 1)
            fig.append_trace(valid_loss, 2, 1)
            fig.append_trace(test_loss, 1, 1)
            fig['layout'].update(layout)

            py.plot(fig, filename=fname, auto_open=False)
