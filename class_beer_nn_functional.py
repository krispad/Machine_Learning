#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:55:31 2026

@author: krishna
"""

####################                     The Functional Model  
########################################## with Tensor Flow.#######################################

##################### The method detailed below is useful when the number of observations is very large, i.e., several million 

'''
    It's best to re-start the kernel at this point and load all the APIs enabling use of TensorFlow, sklearn, etc. 
    
'''
# API's needed        
import tensorflow as tf # version 2.21 used here 
from keras import layers
import numpy as np 
import  sklearn.metrics as sm     
class func_beer_model(object):
    
      
     def __init__(self, X_train, y_train):        
        
       
        ############################################## TRANSFORMING a .csv DATASET to a tf.data DATASET ###########################################
        ## Use the dictionary method --- construct  symbolic values to keys in the dictionary 'inputs' 
        inputs = {}
        self.beer_features = X_train.copy()
        self.beer_train_label = y_train
        #beer_features['storenum'] = beer_features.storenum.astype(str)
        '''self.beer_test_features = X_test.copy()
        self.beer_test_label = y_test'''
        for name, column in self.beer_features.items(): # note that beer_features.items() is a generator
            dtype = column.dtype
            if  dtype == object:
                dtype = tf.string # continue 
            
            else:
                 dtype = tf.float32
            
            inputs[name] = tf.keras.Input(name = name, shape = (1, ), dtype = dtype)
        self.inputs = inputs    
        # Normalization of the NUMERIC data 
        numeric_inputs = {name:col for name, col in self.inputs.items()
                          if col.dtype==tf.float32}
        
        x = layers.Concatenate()(list(numeric_inputs.values()))
        norm = layers.Normalization()
        norm.adapt(np.array(self.beer_features[numeric_inputs.keys()]))
        all_numeric_inputs = norm(x)
        # all the values in the constructed input  are numeric
        
        preprocessed_inputs = [all_numeric_inputs]
        
        for name , value in self.inputs.items():
            if value.dtype == tf.float32:
                continue
            # for strings use
            else: 
                
                lookup = tf.keras.layers.StringLookup(vocabulary = np.unique(self.beer_features[name]))
                # convert the integer representations of strings to one-hot encoding
                one_hot = layers.CategoryEncoding(num_tokens = lookup.vocabulary_size())
                
                x = lookup(value)
                x = one_hot(x)
                preprocessed_inputs.append(x)
           
    
        # concatenate the preprocessed_inputs together 
        preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
        
        self.beer_preprocessing = tf.keras.Model(self.inputs, preprocessed_inputs_cat)    # inputs
            
     def model_constrn_plots(self):
       
       
       #### Plot a symbolic representation of the model with the input and output ###
    
       # Note : install pydot and graphviz
        image_file = '/home/krishna/Documents/Beer/Beer_data/beer_model_symbolic.png'
            
        tf.keras.utils.plot_model(model= self.beer_preprocessing, rankdir = 'LR', show_shapes = False, dpi = 72, to_file = image_file)
        
       

      # Building the model framework with the symbolic representations of the dataset 'beer'

     def beer_model(self):
       
        body = tf.keras.Sequential([  tf.keras.layers.Dense(30, activation = 'sigmoid')
                                    , tf.keras.layers.Dropout(.1)
                                    , tf.keras.layers.Dense(10, activation = 'sigmoid')
                                    , tf.keras.layers.Dropout(.4)
                                    , tf.keras.layers.Dense(1)
                                   ])
        preprocessed_inputs = self.beer_preprocessing(self.inputs)
        result = body(preprocessed_inputs)
        model = tf.keras.Model(self.inputs, result)
        model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())
                
       
        return model
    
     def test_the_model(self, X_test, y_test):
       
        beer_fn_model = self.beer_model()
        # Fitting the model to the dictionary of tensors , 'beer_features_dict'
        beer_features_dict = {name:np.array(value) for name, value in self.beer_features.items()}   
        beer_fn_model.fit(x = beer_features_dict , y = self.beer_train_label, epochs  = 4)   # beer_features_dict
        
        # Predictions 
        ## Converting to a dictionary of beer features - 'beer_test_features'
        beer_test_features = X_test.copy()
        beer_test_features_dict = {name:np.array(value) for name, value in beer_test_features.items()}
        
        #example_beer_test_features_dict = {name:values[:1] for name, values in self.beer_test_features_dict.items()}
        
        beer_test_pred = beer_fn_model.predict(beer_test_features_dict)
        # Aside
        beer_test_fn = [x[0] for x in beer_test_pred]
        mse  = sm.mean_squared_error(np.array(y_test), np.array(beer_test_fn))
        return f'The mse is {mse}', beer_test_pred 
      
    
     def epoch_plot(self, X_test, y_test):
         mse_fn = []
         beer_fn_model = self.beer_model()
        
         # Fitting the model to the dictionary of tensors , 'beer_features_dict'
         beer_features_dict = {name:np.array(value) for name, value in self.beer_features.items()}   
         for num in range(1, 5, 1):
             beer_fn_model.fit(  x = beer_features_dict
                               , y = self.beer_train_label
                               , epochs = num
                               , verbose = 0
                                  )
             # Predictions 
             
             beer_fn_pred = [x[0] for x in  beer_fn_model.predict(beer_features_dict)]
                      
             mse_fn.append(sm.mean_squared_error(np.array(self.beer_train_label), np.array(beer_fn_pred)))
         _, fn_test_pred = self.test_the_model(X_test, y_test)
         fn_test_pred = [x[0] for x in fn_test_pred]
         return mse_fn, fn_test_pred

#############################################  END  ######################################################################