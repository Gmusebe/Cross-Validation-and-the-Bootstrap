# Cross-Validation and the Bootstrap
Also known as Resampling Methods.

**Cross-validation** refers to a set of methods for measuring the performance of a given predictive model validating its effectiveness.

It involves divinding data into two sets and  fitting the same statistical method multiple times using different subsets of these data.

The two sets of data are:

    1) The training set: on which a model is build/fit &
  
    2) Validation/Test se: used to validate the model by estimating the prediction error.

The resampling methods exercised are:

    a) Validation set approach (or data split)
  
    b) Leave One Out Cross Validation
  
    c) k-fold Cross Validation &
  
    d) Bootstrap 

Each of the above methods has its drawbacks and profit in analysis. In statistics the most recommended method is the k-fold Cross Validation.

## The Bootstrap
Similar to cross-validationthe bootstrap resampling method can be used to measure the accuracy of a predictive model. 

Additionally, it can be used to measure the uncertainty associated with any statistical estimator.

Bootstrap resampling consists of repeatedly selecting a sample of n observations from the original data set, and to evaluate the model on each copy. An average standard error is then calculated and the results provide an indication of the overall variance of the model performance.

One of the great advantages of the bootstrap approach is that _it can be applied in **almost all situations**_.

## **Procedure**:
 A bootstrap analysis in R entails only two steps:
 
  1. First, we must create a function that computes the statistic of interest.
  
  2. Second, we use the boot() function to perform the bootstrap by repeatedly sampling observations from the data set with replacement.
