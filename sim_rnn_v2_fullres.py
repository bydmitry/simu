import os
import theano
import shutil
import pickle
import platform

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
if platform.node() == 'lm4-fim-mg3qd.pc.helsinki.fi':
    plt.ioff()
plt.style.use('ggplot')

#import skimage
import pylab as pl
import numpy as np
#import pandas as pd

import sklearn
from sklearn.metrics import roc_auc_score

# CoxPH
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

#from matplotlib import offsetbox
#from sklearn.feature_extraction import image

import keras
from keras import backend as K

os.environ['KERAS_BACKEND'] = 'theano'
if platform.node() == 'lm4-fim-mg3qd.pc.helsinki.fi':
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'

np.random.seed(1337) # for reproducibility
#np.set_printoptions(precision=5, suppress=True)

#==============================================================================
# Custom model initializer Class:
#==============================================================================
class Builder(object):
    def __init__(self, opts, train_data, test_data, train_ids, test_ids):
        self.opts  = opts
        self.test_data  = test_data
        self.train_data = train_data
        self.train_ids  = train_ids
        self.test_ids   = test_ids

        # Initialize model architecture:
        self.model = opts['zoo_model']

        # Compile model:
        self.model.name = opts['exp_name']
        self.model.compile(loss = opts['loss'], optimizer = opts['optimizer'],metrics=['accuracy'])

        # Prepare working directory:
        self.wrk_dir    = os.path.join(opts['exp_path'], opts['exp_name'])
        self.state_dir  = os.path.join(self.wrk_dir, 'state')

        if self.opts['continue'] and os.path.exists(self.state_dir):
            # Continue training existing model:
            self.routines = self.EpochRoutine(builder=self, history=True)

            curr_epoch = self.routines.history['epoch'][-1]
            curr_state = os.path.join(self.state_dir, 'epoch_%s.h5' % str(curr_epoch))
            self.model.load_weights( curr_state )
            print('\nRestoring the model from: %s' % self.wrk_dir)
            print('continue from epoch: %s ' % str(curr_epoch))
        else:
            # Initialize a new model:
            print('\ninitializing a new model...')
            if os.path.exists(self.wrk_dir):
                shutil.rmtree(self.wrk_dir)

            # Create a working directory:
            os.makedirs(self.wrk_dir)
            os.makedirs(self.state_dir)

            # Save model structure:
            with open(os.path.join(self.wrk_dir, 'model.yml'), "w") as text_file:
                text_file.write( self.model.to_yaml() )

            self.routines = self.EpochRoutine(builder=self, history=False)

        print('Model initialized.')

    # ------------------------------------------------------------------------------
    def train(self, epoches, batch, verbose=1):
        history = self.model.fit( x = self.train_data[0],
                         y = self.train_data[1],
                         batch_size       = batch,
                         nb_epoch         = epoches,
                         verbose          = verbose,
                         validation_split = self.opts['val_split'],
                         #validation_data  = (self.test_data[0], self.test_data[1]),
                         callbacks        = [ self.routines ]  )

        return history

    # ------------------------------------------------------------------------------
    # EpochRoutine Class:
    # ------------------------------------------------------------------------------
    class EpochRoutine(keras.callbacks.Callback):
        def __init__(self, builder, history=False, *args, **kwargs):
            super(builder.EpochRoutine, self).__init__(*args, **kwargs)

            self.opts = builder.opts
            self.builder = builder

            if not history:
                self.history = dict()
                self.history['epoch']       = []
                self.history['train_loss']  = []
                self.history['valid_loss']  = []
                self.history['test_loss']   = []
                self.history['train_acc']   = []
                self.history['valid_acc']   = []
                self.history['test_acc']    = []

                self.history['train_AUC']   = []
                self.history['test_AUC']    = []
                self.history['train_HR']    = []
                self.history['test_HR']     = []

                self.epoch_count = 0
            else:
                history_path = os.path.join(builder.wrk_dir, 'history.pkl')
                self.history = pickle.load(open(history_path, 'rb'))
                self.epoch_count = self.history['epoch'][-1]

        # ------------------------------------------------------------------------------
        def on_epoch_end(self, epoch, logs={}):
            epoch += 1
            self.epoch_count += 1
            # Available entities:
            # epoch, logs, self.model, self.params, self.opts, self.history

            # Get statistics on test set:
            test_eval = self.model.model.evaluate(
                self.builder.test_data[0],
                self.builder.test_data[1],
                batch_size=20
            )

            # Get AUCs & HRs:
            auc_hr = self.get_auc_hr()

            # Save training performance (loss):
            self.history['epoch'].append( self.epoch_count )
            self.history['train_loss'].append( logs.get('loss') )
            self.history['valid_loss'].append( logs.get('val_loss') )
            self.history['test_loss'].append( test_eval[0] )
            self.history['train_acc'].append( logs.get('acc') )
            self.history['valid_acc'].append( logs.get('val_acc') )
            self.history['test_acc'].append( test_eval[1] )

            self.history['train_AUC'].append( auc_hr['train_AUC'] )
            self.history['test_AUC'].append( auc_hr['test_AUC'] )
            self.history['train_HR'].append( auc_hr['train_HR'] )
            self.history['test_HR'].append( auc_hr['test_HR'] )

            # Visualize loss at each apoch:
            self.plot_loss(y1='train_loss', y2='valid_loss', y3='test_loss')
            self.plot_accuracy(y1='train_acc', y2='valid_acc', y3='test_acc')

            # Viuslize AUCs & HRs:
            self.plot_auc_hr()

            # Dump model weights & update training history:
            if ( self.epoch_count % self.opts['dump_freq'] == 0 or self.epoch_count == 1):
                self.dump_state()
                self.dump_history()
        # ------------------------------------------------------------------------------

        def dump_state(self):
            self.model.save_weights(
                os.path.join(self.builder.state_dir,'epoch_%d.h5' % self.epoch_count) )

        def dump_history(self):
            path = os.path.join(self.builder.wrk_dir, 'history.pkl')
            pickle.dump( self.history, open(path, 'wb') )

        def plot_accuracy(self, y1, y2, y3):
            file_name = 'Accuracy.png'
            # Make a plot:
            pl.figure()
            pl.title( 'Accuracy (1-error)' )
            pl.plot(self.history['epoch'], self.history[y1], label=y1, color='#348ABD')
            pl.plot(self.history['epoch'], self.history[y2], label=y2, color='#8EBA42')
            pl.plot(self.history['epoch'], self.history[y3], label=y3, color='#BA4252')
            pl.ylim(0.0, 1.0)
            pl.legend(loc='upper center', ncol=3)
            pl.savefig(os.path.join(self.builder.wrk_dir, file_name))

        def plot_auc_hr(self):
            file_name = 'auc_hr.png'
            # Make a plot:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            #lns1 = ax1.plot(self.history['epoch'], self.history['train_AUC'], label='train_AUC', linestyle='--', color='#7D288A')
            lns2 = ax1.plot(self.history['epoch'], self.history['test_AUC'], label='test_AUC', color='#1DA0CC')
            #lns3 = ax2.plot(self.history['epoch'], self.history['train_HR'], label='train_HR', linestyle='--', color='#BD6734')
            lns4 = ax2.plot(self.history['epoch'], self.history['test_HR'], label='test_HR', color='#CC491D')
            #ax1.set_ylim(0.45,1)
            ax1.set_ylabel('AUC'); ax2.set_ylabel('CoxPH HR')
            ax1.tick_params(axis='y', colors='#1DA0CC')
            ax2.tick_params(axis='y', colors='#CC491D')
            ax1.yaxis.label.set_color('#1DA0CC')
            ax2.yaxis.label.set_color('#CC491D')
            # legend:
            lns = lns2+lns4
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper center', ncol=2)
            plt.savefig(os.path.join(self.builder.wrk_dir, file_name))

        def plot_loss(self, y1, y2, y3):
            file_name = 'Loss.png'
            # Make a plot:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            lns1 = ax1.plot(self.history['epoch'], self.history[y1], label=y1, color='#348ABD')
            lns2 = ax2.plot(self.history['epoch'], self.history[y2], label=y2, color='#8EBA42')
            lns3 = ax2.plot(self.history['epoch'], self.history[y3], label=y3, color='#BA4252')
            ax1.set_ylabel(y1); ax2.set_ylabel('valid/test losses')
            # legend:
            lns = lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper center', ncol=3)
            plt.savefig(os.path.join(self.builder.wrk_dir, file_name))

        def get_activations(self, x, model, layer, learn_phase = 0):
            func = K.function([model.layers[0].input, K.learning_phase()], [ model.layers[layer].output ])
            return func([x, learn_phase])

        def total_epoch_count(self):
            return self.epoch_count

        def get_auc_hr(self):
            res = dict()
            # Prepare annotations
            anno_loc   = '/homeappl/home/bychkov/deep_learning/spie/data/patient_tbl.csv'
            anno_data = pd.read_csv( anno_loc )
            del anno_loc

            # Prepare predictions:
            preds_train = np.squeeze(self.model.predict_proba(
                self.builder.train_data[0],
                batch_size=20
            ))
            preds_test = np.squeeze(self.model.predict_proba(
                self.builder.test_data[0],
                batch_size=20
            ))

            # Merge into one table:
            data = np.array([ np.concatenate([self.builder.train_ids,self.builder.test_ids]),
                              np.concatenate([preds_train,preds_test]),
                              np.concatenate([np.zeros_like(preds_train),np.ones_like(preds_test)]) ])
            res_df = pd.DataFrame(data.transpose(), columns=['code','preds','split'])

            res_df = pd.merge(res_df, anno_data, on='code')

            # Prepare columns:
            res_df['fu_int'] = np.round(res_df['fu_days2']).astype('int64')
            res_df['dss']    = res_df['censor_status2'] + 1
            res_df_tr        = pd.DataFrame( res_df.loc[ res_df['split'] == 0 ] )
            res_df_ts        = pd.DataFrame( res_df.loc[ res_df['split'] == 1 ] )

            res_df_tr['cat'] = res_df_tr['preds'] > res_df_tr['preds'].median()
            res_df_ts['cat'] = res_df_ts['preds'] > res_df_ts['preds'].median()


            # Fit Cox model:
            cf_tr = CoxPHFitter(normalize=False)
            cf_tr.fit(res_df_tr[['fu_int','dss','cat']], 'fu_int', event_col='dss')
            cf_ts = CoxPHFitter(normalize=False)
            cf_ts.fit(res_df_ts[['fu_int','dss','cat']], 'fu_int', event_col='dss')

            # Save results:
            res['train_HR']  = np.exp(cf_tr.hazards_.as_matrix()[0][0])
            res['test_HR']   = np.exp(cf_ts.hazards_.as_matrix()[0][0])

            res['train_AUC'] = roc_auc_score(res_df_tr['dss'].as_matrix(), res_df_tr['preds'].as_matrix())
            res['test_AUC']  = roc_auc_score(res_df_ts['dss'].as_matrix(), res_df_ts['preds'].as_matrix())
            return res

#==============================================================================
# Model
#==============================================================================

import keras.models as models
from keras.layers import InputLayer
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2

input_dims = 4096
sequence_length = 256

model = models.Sequential()
model.add( InputLayer(input_shape=(sequence_length, input_dims,), name='InputLayer') )
model.add( Dropout(0.05) )
model.add( LSTM(128, activation='tanh', return_sequences=True, W_regularizer=l1l2(l1=0.0005, l2=0.0005 )) ) # W_regularizer=l1l2(l1=0.0007, l2=0.0007 )
model.add( LSTM(64, activation='tanh', return_sequences=True, W_regularizer=l1l2(l1=0.0005, l2=0.0005 )) ) # W_regularizer=l1l2(l1=0.0007, l2=0.0007 )
model.add( LSTM(32, activation='tanh', W_regularizer=l1l2(l1=0.0005, l2=0.0005 )) ) # W_regularizer=l1l2(l1=0.0007, l2=0.0007 )
model.add( Dropout(0.05) )
model.add( Dense(1, activation='sigmoid') )

#==============================================================================
# Main code:
#==============================================================================
# Dataset npz-file:
feature_dir = '/wrk/bychkov/data/tiles/sim_rnn_large.npz'

# Which fold to process:
fold = 3

#----------------------------------------------------------------------
# Prepare data:
#----------------------------------------------------------------------
dataset = np.load( feature_dir )
del feature_dir

# Fold masks:
test_mask   = dataset['fold'] == fold
train_mask  = np.logical_not(test_mask)

# Final arrays:
train_set_x = dataset['features'][train_mask]
test_set_x  = dataset['features'][test_mask]

# Unroll tiles into sequence:
shp = train_set_x.shape
train_set_x = np.reshape(train_set_x, (shp[0],shp[1]*shp[2],shp[3]))
shp = test_set_x.shape
test_set_x  = np.reshape(test_set_x, (shp[0],shp[1]*shp[2],shp[3]))
del shp

# Now scale the features:
# !IMPORTANT: take into consideration that features are sparse
#from sklearn.preprocessing maxabs_scale

# labels:
train_set_y = dataset['dc5y'][train_mask]
test_set_y  = dataset['dc5y'][test_mask]

train_set_id = dataset['pt_ids'][train_mask]
test_set_id  = dataset['pt_ids'][test_mask]
#----------------------------------------------------------------------
# Set model & training parameters:
#----------------------------------------------------------------------
opts                    = dict()
opts['dump_freq']       = 1
opts['zoo_model']       = model
opts['val_split']       = 0.214285 # 60 out of 280
opts['loss']            = 'binary_crossentropy'
opts['optimizer']       = 'adadelta'
opts['exp_path']        = '/wrk/bychkov/spie/on_tiles'
opts['exp_name']        = 'sim_rnn_v2_fullres/fold_%s' % str(fold)
opts['continue']        = True

#----------------------------------------------------------------------
# Define the model:
#----------------------------------------------------------------------
network = Builder( opts = opts,
                    train_data = ( train_set_x, train_set_y ),
                    test_data  = ( test_set_x, test_set_y ),
                    train_ids  = train_set_id,
                    test_ids   = test_set_id  )

history = network.train(epoches=60, batch=20, verbose=1)
#----------------------------------------------------------------------
# Prepare for Shiny:
#----------------------------------------------------------------------
#opts = dict()
#opts['loss']            = 'binary_crossentropy'
#opts['optimizer']       = 'adadelta'
#
#train_set_ids, test_set_ids = (dataset['area_ids'][train_mask],  dataset['area_ids'][test_mask])
#
#model.compile(loss = opts['loss'], optimizer = opts['optimizer'],metrics=['accuracy'])
#model.load_weights( '../results/krs_ext_vgg_mlp_f3/state/epoch_9.h5' )
#
#preds_train = model.predict(train_set_x, batch_size=10, verbose=1)
#preds_test  = model.predict(test_set_x, batch_size=10, verbose=1)
#
#data = np.array([ np.concatenate([train_set_ids,test_set_ids]),
#                  np.concatenate([np.squeeze(preds_train),np.squeeze(preds_test)]) ])
#res_df = pd.DataFrame(data)
#res_df = res_df.transpose()
#
#res_df.to_csv('/Users/bychkov/Projects/rstudio/spie_clinical/results_csv/ext_cv3-vggMlp_fold.csv', index=False)


#----------------------------------------------------------------------
# misc
#----------------------------------------------------------------------

#hist = spie_cnn.model.fit( x = train_set_x,
#                    y = train_set_y,
#                    batch_size       = 20,
#                    nb_epoch         = 1,
#                    verbose          = 1,
#                    validation_data  = (test_set_x, test_set_y) )
#==============================================================================
# Done.
#==============================================================================