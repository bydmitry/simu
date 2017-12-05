library(survival)
# Create the simplest test data set 
test.d <- as.data.frame(list(
  time   = c(1,2,3,4,5,5,7,1,1), 
  status = c(1,1,0,1,1,1,1,0,1), 
  x      = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
  ))

# Efron
fit <- coxph(Surv(time, status) ~ x, test.d, method = 'efron', init = c(1), iter.max=0) 
round(fit$linear.predictors,3)
fit$loglik


# Exact
fit <- coxph(Surv(time, status) ~ x, test.d, method = 'exact', init = c(1), iter.max=0) 
round(fit$linear.predictors,2)
fit$loglik
