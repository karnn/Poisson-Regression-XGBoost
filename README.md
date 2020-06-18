# Poisson Regression with xgboost

A simple poisson count data model is generated and then fit using an XGBoost regressor
with custom loss function

`obj_func = exp(T(x)) - y T(x)`

where `T(x)` is the output of the boosted tree.

`exp(T(x))` is the estimate of Poisson $\lambda$


