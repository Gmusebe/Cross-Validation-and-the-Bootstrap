# Cross Validation

# Import libraries:
library(ISLR)
library(tidyverse)
library(caret)

# Data: The Auto data in the ISLR package
data <- Auto

# Method 1: Validation Set Approach
# Split data into train data and validation data sets.


# Training set and validation set:
set.seed(1)
train <- data$mpg %>% 
  createDataPartition(p = 0.65, list = FALSE)  # method 1

# Alternatively, one can extract train samples by:
# train = sample(1:nrow(data), size = nrow(data)*0.65, replace = FALSE ) # method 2
# Note samples will be different in individual observations selected.

train_set <- Auto[train, ] # The training set
validation_set <- Auto[-train, ] # The validation set

# Fit a model on the training set:
# Linear regression
lm.fit <- lm(mpg~horsepower, data = train_set)


# Now use the model to predict values using the fitted model:
predict <- lm.fit %>% predict(validation_set)

# predict <- predict(lm.fit, validation_set) In case you used method 2

# Model Performance Metrics:
data.frame(R2 = R2(predict, validation_set$mpg),
           RMSE = RMSE(predict, validation_set$mpg),
           MAE = MAE(predict, validation_set$mpg),
           PER = RMSE(predict, validation_set$mpg)/mean(validation_set$mpg))

# Note: The higher the R2, the lower the RMSE and MAE the better the model.

# Fitting a polynomial to compare against the linear model:
lm.fit1 <- lm(mpg~poly(horsepower, 2), data = train_set )

predict1 <- lm.fit1 %>% predict(validation_set)

# Model metrics
data.frame(R2 = R2(predict1, validation_set$mpg),
           RMSE = RMSE(predict1, validation_set$mpg),
           MAE = MAE(predict1, validation_set$mpg),
           PER = RMSE(predict1, validation_set$mpg)/mean(validation_set$mpg))

# Comparing, the polynomial fit(lm.fit1) is a better model than the linear fit(lm.fit)
# A disadvantage is that we build a model on a fraction of the data set only, possibly leaving out some interesting information about data, leading to higher bias.



# Method 2:Leave-One-Out Cross-Validation (LOOCV)
# _______________________________________________
# LOOCV addresses the Validation set approach flaw 
# The LOOCV estimate can be automatically computed for any generalized linear model using the glm() and cv.glm() functions
library(boot)
glm.fit <- glm(mpg~horsepower, data = data)
cv.err <- cv.glm(data, glm.fit)

cv.err$delta # Cross-validation results.

# Fit increasing complex polynomial fits by the for loop
cv.err = rep(0, 5)
for(i in 1:5){
  glm.fit <- glm(mpg~poly(horsepower, i), data = data)
  cv.err[i] <- cv.glm(data, glm.fit)$delta[1]
}
cv.err

# we see a sharp drop in the estimated test MSE between the linear and quadratic fits.
# The quadratic fit is the best having a lower MSE.
# Thus:
train.control <- trainControl(method = "LOOCV")
LOOCV_fit <- train(mpg ~ poly(horsepower,2), data = data,
                   method = "lm",
                   trControl = train.control)
LOOCV_fit
# The advantage of the LOOCV method is that we make use all data points reducing potential bias.


# k-Fold Cross-Validation
# k-fold CV is that it often gives more accurate estimates of the test error rate than does LOOCV
# The cv.glm() function can also be used to implement k-fold CV.
set.seed(17)
cv.err.10 <- rep(0, 10)
for(i in 1:10){
  glm.fit <- glm(mpg~poly(horsepower, i), data = data)
  cv.err.10[i] <- cv.glm(data, glm.fit, K= 10)$delta[1]
}

cv.err.10
# The quadratic fit to the the 9th place it the best fit having the lowest MSE. Thus:
train.control2 <- trainControl(method = "cv", number = 10)
kfold_fit <- train(mpg~poly(horsepower, 9), data = data,
                   method = "lm",
                   trControl = train.control2)

# The Bootstrap
#______________
# This method can be used to measure the accuracy of a predictive model.
# Additionally, it can be used to measure the uncertainty associated with any statistical estimator.
# One of the great advantages of the bootstrap approach is that it can be applied in almost all situations.

# Bootstrap resampling consists of repeatedly selecting a sample of n observations from the original data set, and to evaluate the model on each copy.
# An average standard error is then calculated and the results provide an indication of the overall variance of the model performance.
# a bootstrap analysis in R entails only two steps:
#  1. First, we must create a function that computes the statistic of interest.
#  2. Second, we use the boot() function to perform the bootstrap by repeatedly sampling observations from the data set with replacement.

# 1. Create function: which takes as input the (X, Y) data as well as a vector indicating which observations should be used to estimate ??.
#    The function then outputs the estimate for ?? based on the selected observations.

alpha.fn <- function(data, index){
  X = data$X[index]
  Y = data$Y[index]
  return( (var(Y)-cov(X, Y))/( var(X)+var(Y)-2*cov(X, Y)) )
}

boot(Portfolio, alpha.fn, R = 1000)


# Estimating the Accuracy of a Linear Regression Model
#____________________________________________________
# The bootstrap approach can be used to assess:
# 1. the variability of the coefficient estimates and 
# 2. predictions from a statistical learning method

# 1. !Create a simple function, boot.fn(), which takes in the Auto data set as well as a set of indices for the observations, 
#     and returns the intercept and slope estimates for the linear regression model.
boot.fn <- function(data, index)
  return(coef(lm(mpg~horsepower, data = data[index,])))
#coefficients of the data:

boot.fn(Auto, 1:392)

# Next, we use the boot() function to compute the standard errors of 1,000 bootstrap estimates for the intercept and slope terms.
boot(Auto, boot.fn, 1000)

# This indicates that the bootstrap estimate for SE( ^ ??0) is 0.84192,
# and that the bootstrap estimate for SE( ^ ??1) is 0.007348.

# The bootstrap approach is likely giving a more accurate estimate of the standard errors of ^ ??0 and ^ ??1 than is the summary()