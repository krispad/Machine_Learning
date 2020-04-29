#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:14:14 2020

@author: krishna
"""

''' Optimization '''

  
  
  
''' Perceptron 
We use perceptrons to determine the vector w that separates a given set of vectors x_i , i = 1, ..., m . 
To do this , we use the Perceptron Convergence Algorithm. 

In the case where the vectors X_i, i = 1, ...., m are linearly separable , the perceptron algorithm finds a 
hyperplane with vector w so that X*w > 0 where X = [X_1, ..., X_m] and X has shape (n, m)

The perceptron convergence theorem does not indicate that the X_i, i = 1, ...., m are not linearly
separable, except that the algorithm does not comverge. However proof of non-convergence implies non separability
 
We note that a linear programming formulation also solves this problem. The 
problem may be cast as a feasibiliy problem. Primal infeasibility implies linear
dependence (through the unbounded dual) of the vectors X_1, ..., X_m . 
Primal feasibility (that is linear separability) implies linear independence of the vectors X_i, i = 1, ..., m
In the case of linear separability , 
we provide a linear programming solution  given by the function 'linprog_soln'.
The w found using the linprog method is not necessarily the same w as found by 
the perceptron convergence algorithm.
'''
  
def percep_update(w, X, n):
    ''' Arguments: 
         0) n - The row dimension of X
         1) w - vector of numbers with length n
         2) X - n by m array of numbers 
    '''
    import numpy as np 
    X = np.array(X)
    w = np.array(w)
    l = np.dot(w, X) > 0 
    l = l.tolist()
    positions = [i for i , x in enumerate(l) if x == False] # positions where the  X_i give  a non-positive
    # value when multiplied with the current w.
    X_neg = X[:, positions] 
    # tX_neg = np.transpose(X_neg)
    #print(X_neg)
    #print(np.shape(X_neg))
    X_new = np.dot(X_neg, np.ones((len(positions), )))
    #print(X_new)
    #tX_new = np.transpose(X_new)
    
    w = w + X_new
    return(w)
  
 # We use recursion to generate the solution. 
  
def percep_alg(w, X, n):
  ''' w : initial value is 0; vector of zeros of length n
      X : an array of size n by m 
      n: the row dimension of X
      
  '''
  import numpy as np
   
   
  while (np.dot(w, X) > 0).all() == False:
 
    w = percep_update(w, X, n)
  
    w = np.array(w)
   
    
    percep_alg(w , X  , n )
  
  return(w)   

''' Linear Programming '''

def linprog_soln(c, A, b):
    from scipy.optimize import linprog
  
    lin_res = linprog(c = c, A_eq = A, b_eq = b, options = {"disp":True})
    print(lin_res)