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
  real r;
  real R;
}
parameters {
  real<lower=0> sigma_angle;
}
transformed parameters {
  vector[N_train] p_train = putting_prob_angle(x[ix_train], sigma_angle, r, R);
}
model {
  sigma_angle ~ normal(0, 1);
  if (likelihood){y[ix_train] ~ binomial(batch_size[ix_train], p_train);}
}
generated quantities {
  real sigma_angle_degrees = sigma_angle * 180 / pi();
  vector[N_test] p_test = putting_prob_angle(x[ix_test], sigma_angle, r, R);
  array[N_test] int yrep;
  vector[N_test] llik;
  for (n in 1:N_test){
    yrep[n] = binomial_rng(batch_size[ix_test[n]], p_test[n]);
    llik[n] = binomial_logit_lpmf(y[ix_test[n]] | batch_size[ix_test[n]], p_test[n]);
  }
}
