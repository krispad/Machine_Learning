# California Housing Data( due to Pace and Barry, 1997) - source http://lib.stat.cmu.edu/datasets/     1990 house values across the state California, USA

# Exploratory Data Analysis
# Panel Plots of pairs of the variables


## Panel : histograms on the diagonal
# Taken from the examples on R  help pages for 'pairs' and developed by the R-Core Team. Included here for convenience

panel.hist <- function(x, ...)
{
  usr <- par("usr")
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE, breaks = 'FD')
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}


## (absolute) correlations on the upper panels,
## with size proportional to the correlations.
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}

# Pairs plots of the data 
pairs(houses2.ca[c(1:4, 7:9)], lower.panel = panel.smooth, diag.panel = panel.hist, upper.panel = panel.cor )


# plot of house_value as a function of the median income ; house_values are broken down by median_age and median_income
library(ggplot2)
g1 <- ggplot(data = houses2.ca, mapping  = aes(x = house_value, y = median_income)) + 
  geom_point(aes(col = housing_median_age, size = median_income)) + geom_smooth(mapping = aes(x = house_value, y = median_income), method = "loess", col = "firebrick") + 
  labs(title = "California Housing Values", subtitle = "Year 1990", caption = "California Housing Values against Median Income")

# Modification of the explanatory variables:
houses_cal_mod <- function(dat, varnames, modnames, first) {  
  # dat -- name of the data set 
  # varnames - variable names whose values need to be modified
  # modnames -names of variables that modify variables in varnames ( for the housing data 'modnames' has two entries, 'population', 'households')
  # first - a number indicating the initial r elements of varnames that are modified by modnames[1], that is in the case of the housing data 'population'
  for (k in varnames[1:first])
  {   avg <- paste('avg_', k, sep = '')
  dat[,avg] <- dat[, k]/dat[, modnames[1]]
  
  } 
  for (k in varnames[(first+1):length(varnames)])
  {   
    avg <- paste('avg_', k, sep = '')
    dat[,avg] <- dat[, k]/dat[, modnames[2]]
  }   
  
  dat <- select(dat, subset = -all_of(varnames))         
  return(dat)
}

