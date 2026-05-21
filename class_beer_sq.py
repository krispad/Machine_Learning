#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:28:28 2026

@author: krishna
"""
import tensorflow as tf   # version 2.21 is used 
import sklearn.metrics as sm
import pandas as pd 
import numpy as np 

class beer_nn(object):
    def __init__(self):
        
        #numpy values up to 3 decimals
        np.set_printoptions(precision= 3, suppress = True)

        # API's needed
     
        
        self.beer_sq_model = tf.keras.Sequential([tf.keras.layers.Dense(30, activation = 'sigmoid') # 
                                             , tf.keras.layers.Dropout(.1)
                                             , tf.keras.layers.Dense(10, activation = 'sigmoid')
                                             , tf.keras.layers.Dropout(.4)
                                             , tf.keras.layers.Dense(1)
                                                ]
                                               )
             
        self.beer_sq_model.compile(optimizer = 'adam'
                              , loss = tf.keras.losses.MeanSquaredError()
                          )

    def seq_nn_model(self):
            


            ################################# Train data ###################################################
            ########### The training and test data have already been calculated: one-hot encoded data is used for the categorical variables and is stored below  
            #               in the location '~/Documents/Beer/Beer_data/Beer_data_nn'          

            beer_train = pd.read_csv('~/Documents/Beer/Beer_data/Beer_data_nn/X_ohe_train.csv') 
                                       
            beer_train_features = beer_train
            beer_train_label = pd.read_csv('~/Documents/Beer/Beer_data/Beer_data_nn/y_ohe_train.csv')
            beer_train_label = beer_train_label['logistic_target']

            beer_test = pd.read_csv('~/Documents/Beer/Beer_data/Beer_data_nn/X_ohe_test.csv') 
                                       
            beer_test_features = beer_test
            beer_test_label = pd.read_csv('~/Documents/Beer/Beer_data/Beer_data_nn/y_ohe_test.csv')
            beer_test_label = beer_test_label['logistic_target']




            #### The model is developed on the training data; a sequence of partitions of the training data is performed to construct validation sets to 
            ##   validate the model developed and eventually choose a suitable model 
            
            ####  ****** Cross_Validation through sampling ******
            #### Generate data sets based on a k-fold selection, i.e, k datasets of train, validate sets.
            
            # Merge beer_train_features and beer_train_label on common columns to form beer_train
            
            '''beer_train = beer_train_features.merge(beer_train_label, how = 'inner', on = beer_train_label.index)
            beer_train.drop('key_0', axis = 1, inplace = True)
            '''
            return(beer_train_features, beer_train_label, beer_test_features, beer_test_label)
        
 ## The code 'nn_mse' below performs cross-validation on the training set. The training set is subdivided and the current model is tested.        
    def nn_mse(self, split_num):
            '''
               split_num: used in a cross-validation technique, viz.,  'KFold',  representing 
                          the number of folds required.
            '''
           
            nn_train , nn_label = self.seq_nn_model() 
            ''' nn_train: a training dataset that is used in the neural net, e.g. beer_train_features 
                 nn_label: response - the observations e.g. beer_train_label(a series) 
                 
            '''
            from sklearn.model_selection import KFold
            
            Indx = nn_train.index 
            kf = KFold(n_splits = split_num)
            #features = nn_train.drop(labels = 'logistic_target', axis = 1, inplace = False)
            
            model_kfold_mse = []
            model_pred_mse = []
            for trn , validate in kf.split(Indx):
                crstrain = nn_train.loc[trn, :] 
                response= nn_label.loc[trn,]
                assert(crstrain.shape[0]) == len(response), "the length of the response does not equal the row dimension of the feature dataset"
                self.beer_sq_model.fit(x = crstrain
                                     , y = response
                                     , epochs = 10
                                     , verbose = 0
                                      )
                pred_train = [x[0] for x in self.beer_sq_model.predict(crstrain)]
                mse = self.beer_sq_model.loss(list(response),pred_train )
                model_kfold_mse.append(self.tf.get_static_value(mse))
                
                valid_features = nn_train.loc[validate, :]
                valid_response = nn_label.loc[validate, ]
                pred_valid = [x[0] for x in self.beer_sq_model.predict(valid_features)]
                mse_pred = self.beer_sq_model.loss(list(valid_response), pred_valid)
                model_pred_mse.append(self.tf.get_static_value(mse_pred))
                
            mse_train_avg = round(sum(model_kfold_mse)/split_num, ndigits = 2)
            
            mse_valid_avg = round(sum(model_pred_mse)/split_num, ndigits = 2)
            
            return model_kfold_mse, model_pred_mse, mse_train_avg, mse_valid_avg
#### A general function to compare between the observed and the predicted 
    def evaluation(self):
        ''' train: features of training set  or validation feature dataset e.g. beer_train_features
            label : training or validation target (response) e.g. beer_train_label
        '''
        train, label = self.seq_nn_model()
        self.beer_sq_model.fit(  x = train
                               , y = label
                               , epochs = 5
                               , verbose = 0 
                              )

        pred = [x[0] for x in self.beer_sq_model.predict(train)]
        target_difference = [round(x, ndigits = 3) for x in list(label - pred)]
        
        
        mse = self.sm.mean_squared_error(label, pred )  # mean squared error

        return target_difference, round(mse, ndigits = 3)
    
    def epoch_plot(self):
        train_features, train_label, test_features, test_label = self.seq_nn_model()
        mse_sq = []
        model = self.beer_sq_model
        for num in range(1, 5, 1):
                       model.fit(  x = train_features
                                 , y = train_label
                                 , epochs = num
                                 , verbose = 0
                                )
        # Predictions 
                       sq_train_pred = [x[0] for x in  model.predict(train_features)]
           
           
                       mse_sq.append(sm.mean_squared_error(np.array(train_label), np.array(sq_train_pred)))
        model.fit(x = train_features, y = train_label, epochs = 4, verbose = 0)
        sq_test_pred = [x[0] for x in  model.predict(test_features)]  
        return mse_sq  , sq_test_pred
################################################################# END ##########################################################    


    
   
            