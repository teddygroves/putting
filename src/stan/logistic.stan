functions {
#include custom_functions.stan
}
data {
  int<lower=1> N;
  int<lower=1> N_train;
  int<lower=1> N_test;
  vector[N] x;
  array[N] int y;
  array[N] int batch_size;
  array[N_train] int<lower=1,upper=N> ix_train;
  array[N_test] int<lower=1,upper=N> ix_test;
  int<lower=0,upper=1> likelihood;
}
parameters {
  real a;
  real b;
}
model {
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  if (likelihood){
    y[ix_train] ~ binomial_logit(batch_size[ix_train], a + b * x[ix_train]);
  }
}
generated quantities {
  array[N_test] int yrep;
  vector[N_test] llik;
  for (n in 1:N_test){
    yrep[n] = binomial_rng(batch_size[n], inv_logit(a + b * x[ix_test[n]]));
    llik[n] = binomial_logit_lpmf(y[n] | batch_size[n], inv_logit(a + b * x[ix_test[n]]));
  }
}
