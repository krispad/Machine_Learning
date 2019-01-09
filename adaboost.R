# ---- part2 ----
gb_ada <- function(dat_orig, M, nfold)
          { library("plyr") # data manipulation library
            library("rpart")
      
      # dat_orig -- the original dataset;  that is, the input data. 
      #  Note that the last variable in the dataset is the response variable, in this case a (-1, +1) variable.
      # M >= 2 --- the number of trees constructed (weak learners)
      # nfold - number of groups dim(dat_orig)[1] is divided into --- nfold cross-validation
      
      # randomly sample (with(out) replacement) the row numbers of dat_orig and attach as a column to the original data
 
      t <- sample(seq(1:dim(dat_orig)[1]), size = dim(dat_orig)[1], replace = F) # really selecting and permuting all the elements in the dataset 
      
      # data set reordered according to random sequence t
      dat_orig <- cbind(dat_orig, t)[order(t), ][, -(dim(dat_orig)[2]+1)] # permuting the order of the rows and deleting the added column denoting the sequence of rows selected
      
      nfold.nbr <- findInterval(x  = seq(from = 1, to = dim(dat_orig)[1], by = 1), vec = seq(from = 1 , to = dim(dat_orig)[1], by = dim(dat_orig)[1]/nfold))
      
     
        # training sample chosen  - the test data is represented by the nfold value
        dat <- dat_orig[(nfold.nbr != nfold), ]   # the training data
       
         a <- paste(names(dat)[1:(dim(dat)[2] -1)], collapse = ' + ')
         model.form = paste(c(names(dat)[dim(dat)[2]], a), collapse = ' ~ ')  # creating the model formula 
      
         # initialization
      w.old = as.vector(rep(1/(dim(dat)[1]), dim(dat)[1])) # initial weights
      
      dat1 <- cbind(dat, w.old)
      
      # initial tree              
      clss_G <- rpart(formula = model.form, data = dat1, weights = w.old,   control = rpart.control(maxdepth = 2,  minsplit = 2))
      
      # class predictions lie in interval (-1,1)
      
      G <- sign(predict(clss_G))  
      y.obs = dat[, dim(dat)[2]] # - observed response vector
      diff <- y.obs == G # logical  ( 0/1) vector depicting the misclassification of the predictions
      # initial misclassification
      err.sum <- sum(!diff)/dim(dat)[1]
    
      alpha.old <- log((1-err.sum)/err.sum)
      
      # weight update 
      w.new= w.old*exp(abs(alpha.old)*(!(y.obs ==G))) # increasing weight on the incorrect classifications 
      
   
      vec = (nfold.nbr != nfold)   # identifying the training data 
      
      # first update on the test and training data 
      G.alphatst <- alpha.old*sign((predict(clss_G, newdata = dat_orig[!vec, ])))    # Test Values lie in (-1, 1) 
      
      G.alphatrn <- alpha.old*sign((predict(clss_G))) # train values in (-1, 1)
      
      # initial setup of the training and test vectors
      train.vec <- vector(mode = "numeric", length =  M)
      test.vec <- vector(mode = "numeric", length = M)
      test.vec[1] = sum(!(dat_orig[!vec,dim(dat_orig)[2]] == sign(G.alphatst)))/sum(!vec) # initial misclassification rate on test set 
      train.vec[1] = sum(!(dat_orig[vec,dim(dat_orig)[2]] == sign(G.alphatrn)))/sum(vec)  # initial misclassification rate on training set 
      
      #variable importance initialization 
     
      mvar_updt <- matrix(as.numeric(clss_G$variable.importance), nrow = 1 , dimnames = list(NULL, names(clss_G$variable.importance)))
      
      
      # Generating a sequence of trees and sequentially boosting and bagging 
      for ( m in seq(from = 1, to = M , by = 1))
      {
        dat1 <- cbind(dat, w.new)
        
        # new tree 
        clss_G.new <- rpart(formula = model.form, data = dat1, control = rpart.control(maxdepth = 3, minsplit = 2),
                            weights = w.new)
        
        G.new <- sign(predict(clss_G.new))  # class estimates 
        
        mvar <- matrix(as.numeric(clss_G.new$variable.importance), nrow = 1, dimnames = list(NULL, names(clss_G.new$variable.importance)))  
      
        diff <- y.obs == G.new
      
        err.sum <- sum(w.new[!diff])/sum(w.new) # weighted 
     
        alpha.new <- log((1 - err.sum)/err.sum)
        w.old <- w.new
        
        # new weights 
        
        w.new <-  w.old*exp(alpha.new*(!(y.obs==G.new)))
        
        vec <- (nfold.nbr != nfold)
        # updates
        G.alphatst <- G.alphatst + alpha.new*sign(predict(clss_G.new, newdata = dat_orig[!vec, ]))
        G.alphatrn <- G.alphatrn + alpha.new*sign(predict(clss_G.new))
      
        alpha.old <- alpha.new
        test.vec[m] = sum(!(dat_orig[!vec,dim(dat_orig)[2]] == sign(G.alphatst)))/sum(!vec) # misclassification on test set 
        train.vec[m] = sum(!(dat_orig[vec,dim(dat_orig)[2]] == sign(G.alphatrn)))/sum(vec)  # misclassification on training set
        #variable importance
        mvar_updt = rbind.fill.matrix(mvar_updt, mvar)
        
      }
      
      pred.df <- matrix(c(train.vec, test.vec), ncol = 2, dimnames =list(NULL,  c("Training Error", "Test Error")))
      
      var.imp <- apply(mvar_updt, MARGIN = 2, FUN = sum, na.rm = T)/M
   
      list(err.sum = err.sum , commitee.testerr = test.vec[M], commitee.trainerr = train.vec[M], errevol = pred.df, variab  = var.imp)
     
      }
     

plt_fn = function(df, M, nfold)
    { # df : error evolution data frame --- use errevol
      plot(x  = seq(from = 1, to = M, by = 1),  ylim = c(0, 1), xlab = "Number of Iterations", ylab = "Error Rate", main = "Error Evolution for Spam Data" , type = "n")
      legend('topright', legend = list("training error", "test error"), col = c("blue", "red"), text.col = c("blue", "red"))
      print(df)
      #print(mode(df[, "train.vec"]))
      points(x = seq(from = 1, to = M, by = 1), y = as.numeric(df[, 1]), type = "l", col = "blue") # train error
      points(x = seq(from = 1, to = M, by = 1), y = as.numeric(df[, 2]), type = "l", col = "red") # test error
    }

    

    
barplt = function(imp_var)
    { 
      len <- length(imp_var)
      a <- order(imp_var, decreasing = T)
      barplot(imp_var[a], horiz = T , col = rainbow(len), xlim = c(-.1, max(imp_var)) , xlab = "Variable Importance", ylim = c(0,len) , space = .275, width = .8, ylab  =  "Variables in Email Messages", 
              main = "Frequency of Variables within Email", sub = "Variables used to detect Spam Email" , cex.names = 1.2)
                                                               
      text(x = rep(-.1, len), y = 1:len , pos = 4, labels = names(imp_var)[a], cex = .4)
      
     }


combined_fn <- function(dat_orig, M, nfold)
{ t = gb_ada(dat_orig, M, nfold)
  
  plt_fn(t$errevol, M, nfold)
  barplt(t$variab)
}



combined_fn(spam, M = 20, nfold= 4)