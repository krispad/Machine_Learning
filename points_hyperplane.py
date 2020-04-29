#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:56:57 2020

@author: krishna
"""

# A Hyperplane constructed by the 'SLSQP' method to separate the points, 
# that is,  distinguishing anomolous points from those of the regular variety.
  
def hyper_slsqp(object_f, slsqp_cnstr, bds, df ):
   ''' Arguments:
         object_f: the objective function returned by the python function 'object_f'
         slsqp_cnstr: SLSQP constraints in the form of a dictionary - returned by the function obj_cnstr_SLSQP
         bds : bounds
         df : the input dataset with variables x_1, ..., x_m, y
   '''
   from timeit import default_timer as timer
   start = timer()
   lagr_SLSQP = minimize(object_f, np.ones(np.shape(df)[0]), method = 'SLSQP', jac = object_jac, constraints = [eq_cons], options={'ftol': 1e-9, 'disp': True}, bounds = Bounds(0, 5) )
   indx = lagr_SLSQP.x > 0 
   #print(indx)
   I = np.diag(df['y'][indx])
   X = np.column_stack((df['x_1'][indx], df['x_2'][indx])) 
   IX = np.matmul(I, X)
   beta = np.matmul(lagr_SLSQP.x[indx].T, IX) 
   indx2 = lagr_SLSQP.x[indx] < max(lagr_SLSQP.x) - .0001 # ensuring indx2 is non-empty.
   X_new = np.column_stack((X[:, 0][indx2], X[:, 1][indx2]))
   
   #X = np.column_stack((df['x_1'][indx], df['x_2'][indx]))
   beta_0 = 1 - (1/(np.shape(X_new))[0])*(sum(np.matmul(X_new, beta))) # an averaged beta_0; to ensure stability.
   end = timer()
   print(end - start)
   return(beta, beta_0)
   
# The Scikit Learn Package - the SVM ( Support Vector Machine )  and Linear Model module 
 
  
 def svm_coef(df, n_iterations):
   from sklearn import svm
   lin_clf = svm.LinearSVC(max_iter = n_iterations)
   X = np.column_stack((df['x_1'], df['x_2'])) # explanatory variables
   Y = np.array(df['y'])
   svm_beta = lin_clf.fit(X, Y).coef_
   svm_beta_0 = lin_clf.fit(X, Y).intercept_
   return(svm_beta, svm_beta_0)
 
  
 def logregr(df):
   ''' We use logistic regression to determine the hyperplane '''
    from sklearn import linear_model 
    X = np.column_stack((df['x_1'], df['x_2']))
    Y = np.array(df['y'])
    clf = linear_model.LogisticRegression().fit(X, Y)
    log_beta = clf.coef_
    log_beta_0 = clf.intercept_
    return(log_beta, log_beta_0)
  
'''    Plotting the data - Scatter plots  and the fitted hyperplanes   '''

def plot_simdat(df, n_iterations ):
   ''' df is the input data frame with column labels x_i, y_i; i = 1, ..., n. Here n is the length of the one dimensional array
     from simdat_nn() 
     Arguments: df: data frame with variables x_i, i = 1,..., m ; y
                n_iterations - number of iterations for the SVM alg.
   '''
   import numpy as np 
   import matplotlib.pyplot as plt 
   
   fig, ax = plt.subplots()
   #ax.set_xlim = (-2,2)
   #ax.set_ylim = (-2.2)
   ax.scatter(df['x_1'], df['x_2'], s = (np.array(df['y']==1))*8, marker = "o", color = 'r', label = 'uncorr bivariate (lognormal, gamma)')
   ax.scatter(df['x_1'], df['x_2'], s = (np.array(df['y']==-1))*8, marker = "o", color = 'b', label = 'corr bivariate normal')
   # Choose the range of the x-axis for the hyperplane plots
   x = np.arange(0,  6, .1)
   # support vector machine
   svm_beta, svm_beta_0 = svm_coef(df, n_iterations)
   w = (svm_beta_0 + svm_beta[0, 0]*x)/(-svm_beta[0, 1])
   ax.plot(x, w, color = 'k', linewidth = 1, linestyle = 'dashed', label = 'Support VM')
   
   # sequential least squares quadratic program 
   if (slsqp_beta > 1e-05).all():  
     y =  (slsqp_beta_0 + slsqp_beta[0]*x)/(-slsqp_beta[1])
     ax.plot(x,y, color = 'g', linewidth = 1, label = 'Seq. Least SQ Opt. Alg.')
     
   
  
   # logistic regression    
   log_beta, log_beta_0 = logregr(df)
   z  = (log_beta_0 + log_beta[0, 0]*x)/(-log_beta[0, 1])
   ax.plot(x, z, color = 'm', linewidth = 1, linestyle = 'dashed', label = 'Logistic Opt. Alg.')
       
   plt.legend()
   plt.xlabel('x-axis')
   plt.ylabel('y-axis')
   plt.title('Simulated Data')
   plt.show()