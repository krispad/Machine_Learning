# Machine_Learning 
## Contents: 
1. Small Cap Stock and ETFs with 10-Year Treasury Spread
  - Small Caps: An example of an Insurance Stock (Ticker: IYH) overlayed with the 10 year Treasury Spread and a prediction of the average stock price for the next week.
  - 10 year Treasury Spread movements over time
  - The 10 year Treasury Spread overlayed with a Spline fit and the identification of recovery, expansion,and contraction regions.
2. R code for a gradient boosting algorithm (AdaBoost) with a binary response 
  - AdaBoost.R 
  - R Code to generate: 
    1. Important variables from the results of the Adaboost Algorithm
    2. The evolution of errors for the chosen test and training sets. 
  - An application of Adaboost.R to spam data:
    - spam.pdf

3. Separation of Data Points using Hyperplanes
   1. Hyperplanes_Separating_Data_Points_copy1.pdf
   2. Code used in generating PDF hyperplane file
      - perc_linprog.py   - early history of perceptrons 
      -  seq_lst_sq_qdrtc_prgm.py - quadratic program 
      - points_hyperplane.py -- generates separating hyperplane
      - simdata_fraud1.py -- simulated data for fraud application

4. California Housing 
  - 'houses_initial.pdf' displays: 
     1. A map of California with housing values superimposed
        - U.S. Government shapefiles are used to depict Metropolitan Statistical Areas (MSAs) and the MSAs are colour-coded to represent housing values in California. 
     2. The effect of median income
        - Warning: This file is may be too large to view as is, download the file after clicking on the link symbol [the icon to the left of houses_initial.pdf] (once opened, there is a download button on the upper right hand corner) 
     3. California Housing Code
        - github_house_values_cal.py - generates a geospatial graph of California with the 1990 Housing Values superimposed
        - houses_eda2.R -- R code to generate pair/correlation plots and ggplots of the house values against the median income.
        - cal_houses_gamma.ipynb: 
           - modifying the raw data on housing values
           - constructs a modelling data set with NaN entries, imputing the NaN values
           - cross-validating the data and constructing a GLM with a gamma response. 
             - Commentaries on the validation of the model
        -  California_Housing_Bagging_vs_RForests.ipynb
           - compares Bagging and Random Forests for the 1990 California Housing Data
           - 'Ensemble' methods are used to compare Random Forests, Gradient Boosting and AdaBoost.
        -  Raw dataset containing the California Housing Values:
           - houses.txt 
5. Beer Data and Models
   - Description of the Problem
   - Beer Data 
   - Beer data preparation
      - Data_Preparation
      - Beer_Data_Exploration
   - Beer Models
      - Beer_Models.ipynb
         - Contains models of the beer data viz., Poisson, Tweedie, AdaBoost,and Random Forests.
         - Compares the results of the models. 
      - beer2.py contains the python code (using Tensor Flow) for the Neural Network(NN) construction
      - beer_model2_image.png contains a schematic of the NN architecture 


