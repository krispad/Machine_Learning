#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:59:39 2022

@author: krishna
"""

import pandas as pd  
import numpy as np 

shelves = pd.read_csv('~/Documents/krishna/Contents/Data_Challenge_Work_kp/shelves_relevant.csv',sep = ',')




from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop = 'first') # instantiation
df = {'df_0':X_usw_train, 'df_1':X_usw_test}
df_mdl = {} 
for val in range(0,2, 1):
 
    ohe_rslts = ohe.fit_transform(df[f'df_{val}'][['wks', 'upc_description', 'storenum']]) 
    colnames = list(df[f'df_{val}']['wks'].unique()[1:]) + list(df[f'df_{val}']['upc_description'].unique()[1:])+ ["st" + str(x) for x in list(df[f'df_{val}']['storenum'].unique()[1:])]
    z= pd.DataFrame(ohe_rslts.toarray(),columns = colnames )
    z= x.dropna(axis = 0) # in fact all rows are non-empty 
    z.reindex_like( df[f'df_{val}'], copy = True)
    print(x.index)
    df_mdl[f'df_mdl_{val}'] = x
    
    


x = pd.DataFrame(data = [['strong', 'president', 5, .1], ['medium', 'secretary', 6, .4], ['mild', 'ceo', 7, .5]], index = range(3, 6), columns = ['Type', 'Rank', 'Val1', 'Val2']
)




df = pd.DataFrame({'A': 'a a b'.split(),
                   'B': [1,2,3],
                   'C': [4,6,5]})
g1 = df.groupby('A', group_keys=False)
g2 = df.groupby('A', group_keys=True)


g2[['B', 'C']].apply(lambda x: x / x.sum())


            B    C
A
a 0  0.333333  0.4
  1  0.666667  0.6
b 2  1.000000  1.0



###############################3 Beer Data to tf data : NN constructed #################################3

########################## Input data as a pandas Dataset ###########################3


import pandas as pd 
import numpy as np 
#numpy values up to 3 decimals
np.set_printoptions(precision= 3, suppress = True)


import tensorflow as tf
from tensorflow import keras 
from keras import layers
import sklearn.metrics
from keras import utils 


############## Small datasets with limited range valued features; data type floating point, string #################

#### We can use the dataset in the form of  a pandas dataset 

beer_es_train = pd.read_csv("~/Documents/krishna/Contents/Python/Optimization_Folder/Tensorflow/Keras/Beer/X_train.csv") # "http://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv"
                            
                           

beer_es_train.head()


############# Define the features and the target : Target = 'Age'##################

beer_features = beer_es_train.copy()
beer_label = pd.read_csv("~/Documents/krishna/Contents/Python/Optimization_Folder/Tensorflow/Keras/Beer/yspl_train.csv")
beer_label = beer_label['target_out_stock']

beer_features = np.array(beer_features)

#### PART 1

################# Step 1 : Simple neural net Model ##########################
beer_model = tf.keras.Sequential([  tf.keras.layers.Dense(64, activation = 'relu')
                                     , tf.keras.layers.Dropout(.1)
                                     , tf.keras.layers.Dense(64, activation = 'relu')
                                     , tf.keras.layers.Dropout(.1)
                                     , tf.keras.layers.Dense(1)
                                    ]
                                   )

beer_model.compile(  optimizer = 'adam'
                      , loss = tf.keras.losses.MeanSquaredError()
                     )


beer_model.fit(  x = beer_features
                  , y = beer_label
                  , epochs = 30
                 )

##########################             Test data 

beer_test = pd.read_csv('~/Documents/krishna/Contents/Python/Optimization_Folder/Tensorflow/Keras/Beer/X_test.csv') #"http://storage.googleapis.com/download.tensorflow.org/data/abalone_test.csv"
                           
beer_test_features = beer_test.copy()
beer_test_label = pd.read_csv('~/Documents/krishna/Contents/Python/Optimization_Folder/Tensorflow/Keras/Beer/yspl_test.csv')
beer_test_label = beer_test_label['target_out_stock']

############### list of predictions 
                                  
beer_test_pred = [ x[0] for x in  beer_model.predict(beer_test_features)] # list 

beer_test_diff = np.array(beer_test_label) - np.array(beer_test_pred) #  differences (observed - predicted)

sklearn.metrics.mean_squared_error(np.array(beer_test_label), np.array(beer_test_pred))   # mean squared error


#######################################       PART 2    ############################################################

'''
    It's best to re-start the kernel at this point and load, once again,  all the modules at the beginning of the code.
    Load the X_train and X_test datasets 
'''

##################### The method detailed below is another approach --- useful when the number of observations is very large ~ several million 

############################################## TRANSFORMING a .csv DATASET to a tf.data DATASET ###########################

## Use the dictionary method --- construct  symbolic values to keys in the dictionary 'input' 
input = {}
beer_features = beer_es_train.copy()

for name, column in beer_features.items(): # note that beer_features.items() is a generator
    dtype = column.dtype
    if  dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
        input[name] = tf.keras.Input(name = name, shape = (1, ), dtype = dtype)
    
# Normalization of the NUMERIC data 

# all the values in the constructed input  are numeric
# ( note that the explanatory var. in abalone_train are only numeric)

x = layers.Concatenate()(list(input.values())) # symbolically concatenate the values of input
# into a Keras Tensor with shape (None, 8)
    
norm = layers.Normalization() # normalization layer 
norm.adapt(np.array(beer_features))
numeric_inputs = norm(x)


preprocessed_input = [numeric_inputs]

for name , value in input.items():
    if value.dtype == tf.float32:
        continue
    # for strings use 
    lookup = layers.StringLookup(vocabulary = np.unique(beer_features[name]))
    # convert the integer representations of strings to one-hot encoding
    one_hot = layers.CategoryEncoding(num_tokens = lookup.vocubulary_size())
    
    x = lookup(value)
    x = one_hot(x)
    preprocessed_input.append(x)
    
    # concatenate the preprocessed_inputs together 
preprocessed_input_cat = layers.Concatenate()(preprocessed_input)
beer_preprocessing = tf.keras.Model(input, preprocessed_input_cat)
    
    
    #### Plot a symbolic representation of the model with the input and output ###
    
    # Note : install pydot and graphviz
image_file = '/home/krishna/Documents/krishna/Contents/Python/Optimization_Folder/Tensorflow/Keras/Beer/beer_model.png'
    
tf.keras.utils.plot_model(model= beer_preprocessing, rankdir = 'LR', show_shapes = True, dpi = 72, to_file = image_file)

# Keras models don't automatically convert pandas DataFrames -since it's not clear if it should be converted to one tensor
# or to a dictionary of tensors. So, convert it to a dictionary of tensors:
    
beer_features_dict = {name:np.array(value) for name, value in abalone_features.items()}   


               ############################## Example ##########################
''' The first training observation ( first row of the pandas dataset 'abalone_train') is: '''

example_abalone_features_dict = {name:value[:1] for name, value in abalone_features_dict.items()}
abalone_preprocessing(example_abalone_features_dict)


               ###################### End of Example ###########################
             
# Building the model framework with the symbolic representations of the dataset 'abalone'

def abalone_model(preprocessing_head, input):
    body = tf.keras.Sequential([  tf.keras.layers.Dense(64, activation = 'relu')
                                , tf.keras.layers.Dropout(.1)
                                , tf.keras.layers.Dense(64, activation = 'relu')
                                , tf.keras.layers.Dropout(.1)
                                , tf.keras.layers.Dense(1)
                               ])
    preprocessed_inputs = preprocessing_head(input)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(input, result)
    model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())
    
    return model


abalone_model = abalone_model( abalone_preprocessing, input)
# Fitting the model to the dictionary of tensors , 'abalone_features_dict'
abalone_model.fit(x = abalone_features_dict, y = abalone_train['Age'], epochs  = 30)

# Predictions 
## Converting to a dictionary of abalone features - 'abalone_test_features'
abalone_test_features_dict = {name:value for name, value in abalone_test_features.items()}

abalone_test_pred = abalone_model.predict(abalone_test_features_dict)
sklearn.metrics.mean_squared_error(np.array(abalone_test_label), np.array(abalone_test_pred))

#############################################  END  ###########################################################################

############### Plots of the predicted results (mse) by epoch --- NOTE running the code below takes several minutes ####################

abalone_test_label = abalone_test['Age']

## Converting to a dictionary: abalone_test features --- 'abalone_test_features'
abalone_test_features_dict = {name:value for name, value in abalone_test_features.items()}
mse = []
for num in range(30, 100, 5):
    abalone_model.fit(x = abalone_features_dict, y = abalone_train['Age'], epochs  = num)

    # Predictions 

    abalone_test_pred = abalone_model.predict(abalone_test_features_dict)
    mse.append(sklearn.metrics.mean_squared_error(np.array(abalone_test_label), np.array(abalone_test_pred)))

import matplotlib.pyplot as plt 

fig, ax =plt.subplots(figsize = (10, 10))


ax.plot(range(30, 100, 5), mse)
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Square Error')
ax.set_title('Mean Squared Error: 2-layer model with dropout')

plt.show()