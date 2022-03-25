/* This file is for your stan functions */

vector standardise_vector(vector v, real mu, real s){
    return (v - mu) / (2 * s);
}

matrix standardise_cols(matrix m, vector mu, vector s){
  matrix[rows(m), cols(m)] out;
  for (c in 1:cols(m))
    out[,c] = standardise_vector(m[,c], mu[c], s[c]);
  return out;
}

vector unstandardise_vector(vector v, real m, real s){
  return m + v * 2 * s;
}

vector col_means(matrix m){
  int C = cols(m);
  vector[C] out;
  for (c in 1:C)
    out[c] = mean(m[,c]);
  return out;
}

vector col_sds(matrix m){
  int C = cols(m);
  vector[C] out;
  for (c in 1:C)
    out[c] = sd(m[,c]);
  return out;
}

vector putting_prob_angle(vector x, real sigma, real r, real R){
  /* Putting success probability assuming perfect force, angle noise sigma, ball
     radius r and hole success radius R. */
  return 2 * Phi(asin((R-r) ./ x) / sigma) - 1;
}

vector putting_prob_force(vector x, real sigma, real overshot, real distance_tolerance){
  /* Putting success probability assuming perfect angle, distance noise sigma,
     target distance x+overshot and success zone [x, x+distance_tolerance]. */
  return Phi((distance_tolerance - overshot) ./ ((x + overshot) * sigma)) -
    Phi((- overshot) ./ ((x + overshot) * sigma));

}

vector putting_prob_angle_and_force(vector x,
                                    real sigma_angle,
                                    real sigma_distance,
                                    real r,
                                    real R,
                                    real overshot,
                                    real distance_tolerance){
  /* Putting success probability with independently noisy angle and distance. */
  return putting_prob_angle(x, sigma_angle, r, R)
    .* putting_prob_force(x, sigma_distance, overshot, distance_tolerance);
}
