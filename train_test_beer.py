#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:38:33 2026

@author: krishna
"""
#####################################################################################################################################################
# We need to establish the training and test sets for the data we wish to analyze So run the code below for the beer data to generate the training and test sets. 
# The splitting of the data is exactly the same as that of the sequential model --- the "randomization' it replicated
#####################################################################################################################################################
def dataset_cnstrn():    
  
         # The training and test datasets are constructed 
         import pandas as pd 
         import numpy as np
         np.set_printoptions(precision= 3, suppress = True)
         from sklearn.model_selection import train_test_split
         shelves_relevant = pd.read_csv('~/Documents/Beer/Beer_data/shelves_relevant.csv')
         shelves_relevant.dropna(axis = 0, inplace = True)
         x = ['upc_description', 'storenum', 'wks']   # categorical variables used to group the data 
         y = shelves_relevant[x + ['logistic_target']].groupby(x) # 'logistic_target' is the response variable in the NN model.
         y = y.mean()
         y.reset_index(inplace = True)

         # Focus on the explanatory variables

         X = shelves_relevant.drop(['logistic_target', 'target_out_stock','year', 'upc', 'new_year_mth_wk', 'profit_unit', 'retail_price_unit'], axis = 1, inplace = False)
         X.dropna(axis  = 0, inplace = True) # There are no NAs
         X_grouped = X.groupby(x) # groupedby object - grouping on 'upc_description', 'storenum', 'wks'
         mean_amts = X_grouped.mean()
         mean_amts.reset_index(inplace = True)
         mean_amts.dropna(axis = 0, how = 'any', inplace = True) # Aren't any NAs

         store = [f's{x}' for x in mean_amts['storenum']] # This step and the two steps below are unnecessary 
         mean_amts.pop('storenum')
         mean_amts['storenum'] = store
         # reorder columns
         
         mean_amts = mean_amts[['upc_description', 'wks', 'storenum', 'gross_margin_dollars',    'scanned_retail_dollars', 'scanned_movement',
               'scanned_retail_dollars:scanned_movement']]
         #mean_amts.index
         #print(mean_amts.head(), mean_amts.shape)
          
         X_train,  X_test, y_train, y_test = train_test_split(mean_amts,y['logistic_target'],  test_size = .1, random_state = 100, stratify = None)
         
         return(X_train, y_train, X_test, y_test)
     

################################ Try Plots #################################3
# The code below plots the mean squared error ( mse) of the training data on both the sequential and symbolic model . Similary the predictions on the test set 
# are plotted. This gives a visual comparison between the sequential and symbolic models. 

def plot_model():
    X_train, y_train, X_test, y_test = dataset_cnstrn() 
    import Documents.Beer.class_beer_sq as seq_beer_model
    import Documents.Beer.class_beer_nn_functional as func_beer_model
    seq = seq_beer_model.beer_nn()
    func = func_beer_model.func_beer_model(X_train=X_train, y_train = y_train)
    
    import matplotlib.pyplot as plt
    import numpy as np
    mse_sq, sq_test_pred = seq.epoch_plot()
    mse_fn, fn_test_pred = func.epoch_plot(X_test, y_test)
    
    fig, axs = plt.subplots(2, 1, layout='constrained')
    
    ax = axs[0]
    ax.plot(range(4), mse_fn, color = 'red',  label = 'Functional Model Epochs' )
    ax.plot(range(4), mse_sq, color = 'blue', label = 'Sequential Model Epochs')
    ax.set_title('Mean Square Error (MSE) by Epoch for the Training Data')
    ax.set_ylabel('MSE')
    ax.legend(loc = 'best', fontsize = 'small')
    
    ax = axs[1]
    ax.plot(np.arange(len(y_test)), y_test, color = 'silver', label = 'Observed')
    ax.plot(np.arange(len(fn_test_pred)), fn_test_pred, color = 'red', label = 'Functional Model Prediction')
    ax.plot(np.arange(len(sq_test_pred)), sq_test_pred, color = 'blue', label = 'Sequential Model Prediction')
    
    ax.set_title('Model Test Predictions')
    ax.set_ylabel('Test Predictions vs. Observations')
    ax.tick_params(axis='x', rotation=55)
    ax.legend(loc = 'best', fontsize = 'small') #['line1', 'line2'], ['Functional Model Epochs', 'Prediction'])
    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
    fig.align_titles()
    
    plt.show()
######################################################################################################################

