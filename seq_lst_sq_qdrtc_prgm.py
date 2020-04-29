#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:43:52 2020

@author: krishna
"""

# The code below creates a function object which forms the objective function   
def object_f(a, df, n = 500 ):
    ''' a: an unknown array ; the elements of which are to be determined via constrained optimization 
       df: a data frame containing a response variable 'y' and explanatory variables 'x_1', ..., 'x_k'
       n : the length of the data array generated from the simulated bivariate data 
    '''
    import numpy as np
    #from scipy.optimize import minimize
    #from scipy.optimize import Bounds
    #from scipy.optimize import LinearConstraint
    #import pandas as pd
    #df = simdat_nn(n)
    #df = sim_frd(n)
    I = np.diag(df['y']) # diagonal entries consisting of the response vector df['y']
    X = np.column_stack((df['x_1'], df['x_2'])) 
    IX = np.matmul(I, X)
    tIX = IX.T
    crss_prd = np.matmul(IX, tIX)
    #print(crss_prd)
    a = np.array(a)
    obj = -sum(a) + (1/2)*np.matmul(np.matmul(a.T, crss_prd),a)
  
    return(obj)
  

# Methods
  
# method 'SLSQP' technique - Sequential Least Squares Quadratic Program
  
def object_jac(a, df = df, n = 500):
    '''gradient of the objective function - Jacobian 
    Arguments: a: unknowns - to be determined 
               df: data frame containing variables x_i, i = 1, ..., k and y
    
    '''
    
    import numpy as np
    
    #df = sim_frd(n)
    
    I = np.diag(df['y'])
    X = np.column_stack((df['x_1'], df['x_2'])) 
    IX = np.matmul(I, X)
    tIX = IX.T
    crss_prd = np.matmul(IX, tIX)
   
    a = np.array(a)
   
    jac_obj = - np.ones(np.shape(df)[0]) + np.matmul(crss_prd, a)
    return(jac_obj) 
    


# Begin the Optimization Process 

import numpy as np 
from scipy.optimize import minimize
# An  Optimization function to minimize the Objective subject to Constraints , 
''' Arguments: object_fn -- the objective function 
               x0 : an array representing the starting point-- user input 
               jac : Jacobian of the objective function - the gradient 
               method : method used
               constraints: a list of linear , nonlinear constraints and bounds
'''
# We give an example 
    minimize(object_f, x0 = np.ones(np.shape(df)[0]), method = 'SLSQP', jac = object_jac, constraints = [eq_cons], options={'ftol': 1e-9, 'disp': True}, bounds = Bounds(0, 5))


# the SLSQP method 

''' constraints for the SLSQP method; here we note that the constraints are entered as dictionaries '''
def obj_cnstr_SLSQP(n, df):
   import numpy as np
   #df = sim_frd(n)
   A = np.array(df['y'])
   I = np.diag(A)
   dim_sh  = np.shape(df)[0]
   # equality conditions 
   eq_cons = {'type': 'eq', 'fun': lambda a : np.matmul(np.matmul(I, a).T, np.ones(dim_sh)), 'jac':lambda  a: A }
   # Note 'fun' are the constraints , in this case equality constraints
   #       'jac' the Jacobian of the constraints
   return(eq_cons)
 
  
# The equality constraints  
  
   eq_cons = obj_cnstr_SLSQP(n = 500, df = df) 

# The Bounds  for the 'SLSQP' method

from scipy.optimize import Bounds

# Input Bounds(0, u) --- Note that Bounds(0, u) denotes the bound for each 'x_ij' namely,  0 <= 'x_ij' <= u. 
