#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:01:12 2020

@author: krishna
"""

 ''' Simulated Data '''   
# Data for Vector Support Machines 
# We generate two datasets ; one from a bivariate distribution and the other 
# from a mixture of a bivariate;  dirichlet and gamma distributio.
  
# 1) Simulating data from a bivariate normal distribution 
def simdat_nn(n):
   ''' Generating random data form a bivariate normal distribution to illustrate 
       vector support machines 
   
    
       arguments : n - the length of a one dimensional array '''
   import numpy as np
   import pandas as pd
   mean_1 = [1, 0]
   cov_1 = [[1, 0], [0, 1]]
   mean_2 = [2.5, 1.2]
   cov_2 = [[1.1, 1], [1, 1]]
   (x_11, x_12)= np.random.multivariate_normal(mean_1, cov_1, n).T
   (x_21, x_22) = np.random.multivariate_normal(mean_2, cov_2, n).T
   y_1 = np.ones(n)
   y_2 = -1*np.ones(n)
   z1 = np.column_stack((x_11,x_12, y_1))
   z2 = np.column_stack((x_21, x_22, y_2))
   df1 = pd.DataFrame(z1, columns = ['x_1', 'x_2', 'y'])
   df2 = pd.DataFrame(z2, columns  = ['x_1', 'x_2', 'y'])
   df = pd.concat([df1, df2], ignore_index = True)
   return(df)
 
 # 1b) Simulating data that contains 'fraudulent' data --- for the 'fraud' version we use distributions that
 #     substantiates Bedford's Law  in order to generate the non- fraudulent data.
 
def sim_frd(n):
    '''Arguments: n --- number of non-suspicious entries
                  m --- number of  suspicious entries 
                  note: n >> m
    '''
    import pandas as pd 
    import numpy as np 
    from numpy.random import Generator
    import math 
    m = math.ceil(.01*n) # pollution amount 
    rng = np.random.default_rng()
    mean_1 = [1, 0]
    cov_1 = [[1.1, 1], [1,  1]]
    (x_11, x_12) = np.random.multivariate_normal(mean_1, cov_1, m).T
    
    lg = rng.lognormal(.1, 1, n) # follows Bedford's Law
    nrml = rng.gamma(1.2, 1, n)  # follows Bedford's Law
    (x_21, x_22) = np.column_stack((lg, nrml)).T
    (x_31, x_32) = rng.dirichlet((7, 7), n).T # follows Bedford's Law 
    
    y_1 = -np.ones(m)
    y_2 =  np.ones(n)
    z1 = np.column_stack((x_11, x_12, y_1))
    z2 = np.column_stack((x_21, x_22, y_2)) # Bedford's Law
    z3 = np.column_stack((x_31, x_32, y_2)) # Bedford's Law
    df1 = pd.DataFrame(z1, columns = ['x_1', 'x_2', 'y'])
    df2 = pd.DataFrame(z2, columns = ['x_1', 'x_2', 'y'])
    df3 = pd.DataFrame(z3, columns = ['x_1', 'x_2', 'y'])
    df = pd.concat([df1, df2, df3], ignore_index = True)
    return(df)
    
  