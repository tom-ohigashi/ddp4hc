// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// sampling new claster
int sample_new_cls_logp(
    arma::vec proposal_logp
){
  unsigned int ncl = proposal_logp.n_elem;
  double u = R::runif(0.0, 1.0);
  arma::vec cum_p = arma::cumsum(arma::exp(proposal_logp));
  
  int ret = 0;
  for(unsigned int i=0; i<ncl; i++){
    if(u > cum_p(i)){ret += 1;}
  }
  return(ret);
}

// sampling new claster
int sample_new_cls_p(
    arma::vec proposal_p
){
  unsigned int ncl = proposal_p.n_elem;
  double u = R::runif(0.0, 1.0);
  arma::vec cum_p = arma::cumsum(proposal_p);
  
  int ret = 0;
  for(unsigned int i=0; i<ncl; i++){
    if(u > cum_p(i)){ret += 1;}
  }
  return(ret);
}

arma::uvec tableC(arma::uvec x) {
  arma::uvec unique_vals = arma::unique(x);
  unsigned int num_unique = unique_vals.n_elem;
  
  arma::uvec counts(num_unique);
  for (unsigned int i = 0; i < num_unique; i++) {
    double val = unique_vals(i);
    counts(i) = sum(x == val);
  }
  arma::uvec ret = counts;
  
  return ret;
}

arma::uvec tableC2(arma::uvec z, unsigned int J) {
  arma::uvec unique_vals = arma::unique(z);
  unsigned int num_unique = unique_vals.n_elem;
  
  arma::uvec counts = tableC(z);
  
  arma::uvec m(J);
  for (unsigned int i = 0; i < num_unique; i++) {
    double val = unique_vals(i);
    unsigned int index = val; 
    m(index) = counts(i);
  }
  
  return m;
}

// random sample from truncated normal distribution
double ran_truncnorm(
    double mu,
    double sigma,
    double lower,
    double upper
){
  double y;
  double ry;
  double u;
  double ret;
  
  do{
    y = R::runif(lower, upper);
    ry = R::dnorm(y, mu, sigma, false);
    u = R::runif(lower, upper);
  } while (ry < u);
  ret = y;
  return(ret);
}

// probability density function of truncated normal distribution
double den_truncnorm1(
    double x,
    double mu,
    double sigma,
    double lower,
    double upper
){
  double ret = (exp(-(pow((x-mu),2))/(2*pow(sigma,2)))) / (sqrt(2*M_PI)*sigma*(R::pnorm(((upper-mu)/sigma), 0.0, 1.0, true, false) - R::pnorm((lower-mu)/sigma, 0.0, 1.0, true, false)));
  return(ret);
}

// probability density function of truncated normal distribution
double den_truncnorm(
    double x,
    double mu,
    double sigma,
    double lower,
    double upper
){
  double ret = 0;
  if((x < lower) || (x > upper)){
  }else{
    ret = den_truncnorm1(x, mu, sigma, lower, upper);
  }
  return(ret);
}

arma::vec ran_gamma(arma::vec shape, double scale) {
  int n = shape.n_elem;
  arma::vec ret(n, arma::fill::zeros);
  
  for (int i = 0; i < n; i++) {
    double shape_val = shape(i);
    // ret(i) = R::rgamma(shape_val, scale);
    ret(i) = arma::randg(arma::distr_param(shape_val, scale));
  }
  
  return ret;
}


double ran_half_cauchy(double scale) {
  double u = R::runif(0, 1);
  double ret = scale * std::tan(M_PI*u/2);
  return ret;
}

double den_half_cauchy(double x, double scale) {
  double ret = 2*scale/(M_PI * (std::pow(x, 2.0) + std::pow(scale, 2.0)));
  return ret;
}


arma::vec ran_beta_post(arma::vec y, arma::vec m, double a0, double b0) {
  int n = y.n_elem;
  arma::vec ret(n, arma::fill::zeros);
  
  for (int i = 0; i < n; i++) {
    double a_val = y(i) + a0;
    double b_val = m(i) - y(i) + b0;
    ret(i) = R::rbeta(a_val, b_val);
  }
  
  return ret;
}


double log_sum_w_dbinom(int y, int m, arma::vec th_j, arma::vec w) {
  int n = th_j.n_elem;
  double ret = 0;

  arma::vec wp(n, arma::fill::zeros);
  for(int i=0; i<n; i++){
    wp(i) = w(i) * R::dbinom(y, m, th_j(i), false);
  }
  ret = log(sum(wp));
  return ret;
}

// sum residual2 
double residual2(
    const arma::vec y_C,
    const arma::mat x_C,
    const unsigned int n_C,
    arma::vec beta
){
  arma::vec tmp1(n_C), tmp2(n_C);
  tmp2 = y_C - x_C * beta;
  tmp1 = arma::pow(tmp2, 2);
  return arma::accu(tmp1);
}

double log_normpdf_each(
    const arma::vec y_C,
    const arma::mat x_C,
    const unsigned int n_C,
    arma::vec beta,
    double sigma
){
  arma::vec tmp1(n_C), tmp2(n_C), tmp3(n_C);
  tmp2 = x_C * beta;
  tmp3.fill(sigma);
  tmp1 = arma::log_normpdf(y_C, tmp2, tmp3);
  return arma::accu(tmp1);
}

double normpdf_each(
    const arma::vec y_C,
    const arma::mat x_C,
    const unsigned int n_C,
    arma::vec beta,
    double sigma
){
  arma::vec tmp1(n_C), tmp2(n_C), tmp3(n_C);
  tmp2 = x_C * beta;
  tmp3.fill(sigma);
  tmp1 = arma::normpdf(y_C, tmp2, tmp3);
  return arma::prod(tmp1);
}

double log_sum_w_dnorm(const arma::vec y_C,
                         const arma::mat x_C,
                         const unsigned int n_C,
                         arma::mat beta,
                         arma::vec sigma,
                         arma::vec w) {
  int n = w.n_elem;
  double ret = 0;
  arma::vec tmp1(n_C), tmp2(n_C), tmp3(n_C);
  arma::vec wp(n, arma::fill::zeros);
  for(int i=0; i<n; i++){
    tmp2 = x_C * beta.row(i).t();
    tmp3.fill(sigma(i));
    tmp1 = arma::normpdf(y_C, tmp2, tmp3);
    wp(i) = w(i) * arma::prod(tmp1);
  }
  ret = log(sum(wp));
  return ret;
}

double log_sum_w_dnorm2(const arma::vec y_C,
                       const arma::mat x_C,
                       const unsigned int n_C,
                       arma::mat beta,
                       const double sigma,
                       arma::vec w) {
  int n = w.n_elem;
  double ret = 0;
  arma::vec tmp1(n_C), tmp2(n_C), tmp3(n_C);
  arma::vec wp(n, arma::fill::zeros);
  for(int i=0; i<n; i++){
    tmp2 = x_C * beta.row(i).t();
    tmp3.fill(sigma);
    tmp1 = arma::normpdf(y_C, tmp2, tmp3);
    wp(i) = w(i) * arma::prod(tmp1);
  }
  ret = log(sum(wp));
  return ret;
}



int sample_index_from_prob_cont(const arma::vec& p) {
  double u = R::runif(0.0, 1.0);
  double cs = 0.0;
  for (arma::uword i = 0; i < p.n_elem; ++i) {
    cs += p(i);
    if (u <= cs) return static_cast<int>(i);
  }
  return static_cast<int>(p.n_elem - 1);
}

arma::uvec tabulate_alloc_cont(const arma::uvec& z, const unsigned int K) {
  arma::uvec out(K, arma::fill::zeros);
  for (arma::uword i = 0; i < z.n_elem; ++i) {
    if (z(i) < K) out(z(i))++;
  }
  return out;
}

unsigned int n_occupied_cont(const arma::uvec& z) {
  arma::uvec z_unique = arma::unique(z);
  return z_unique.n_elem;
}

double sample_dp_precision_cont(
    const double M,
    const unsigned int n_obs,
    const unsigned int n_cl,
    const double hyper_gamma_shape,
    const double hyper_gamma_scale
) {
  // Escobar-West update: prior M ~ Gamma(shape, scale).
  double eta = R::rbeta(M + 1.0, static_cast<double>(n_obs));
  double rate = 1.0 / hyper_gamma_scale - std::log(eta);
  double mix_num = hyper_gamma_shape + static_cast<double>(n_cl) - 1.0;
  double mix_den = mix_num + static_cast<double>(n_obs) * rate;
  double prob = mix_num / mix_den;
  
  double shape = (R::runif(0.0, 1.0) < prob)
    ? hyper_gamma_shape + static_cast<double>(n_cl)
      : hyper_gamma_shape + static_cast<double>(n_cl) - 1.0;
  
  return R::rgamma(shape, 1.0 / rate);
}

void update_component_means_cont(
    arma::vec& mu_star,
    const arma::uvec& z,
    const arma::vec& ybar,
    const arma::vec& var_y,
    const double mu_G0,
    const double tau2_G0
) {
  for (arma::uword c = 0; c < mu_star.n_elem; ++c) {
    double prec = 1.0 / tau2_G0;
    double num = mu_G0 / tau2_G0;
    
    for (arma::uword j = 0; j < z.n_elem; ++j) {
      if (z(j) == c) {
        prec += 1.0 / var_y(j);
        num  += ybar(j) / var_y(j);
      }
    }
    
    double post_var = 1.0 / prec;
    double post_mean = post_var * num;
    mu_star(c) = R::rnorm(post_mean, std::sqrt(post_var));
  }
}

void recompute_stick_weights_cont(
    arma::vec& w,
    const arma::vec& v
) {
  w.set_size(v.n_elem);
  double rem = 1.0;
  for (arma::uword c = 0; c < v.n_elem; ++c) {
    w(c) = rem * v(c);
    rem *= (1.0 - v(c));
  }
}


int sample_index_from_prob_ddpm_cont(const arma::vec& p) {
  double u = R::runif(0.0, 1.0);
  double cs = 0.0;
  for (arma::uword i = 0; i < p.n_elem; ++i) {
    cs += p(i);
    if (u <= cs) return static_cast<int>(i);
  }
  return static_cast<int>(p.n_elem - 1);
}

unsigned int n_occupied_ddpm_cont(const arma::uvec& z) {
  arma::uvec z_unique = arma::unique(z);
  return z_unique.n_elem;
}

double log_beta_fn_ddpm_cont(double a, double b) {
  return R::lbeta(a, b);
}

double sample_dp_precision_ddpm_cont(
    const double M,
    const unsigned int n_obs,
    const unsigned int n_cl,
    const double hyper_gamma_shape,
    const double hyper_gamma_scale
) {
  double eta = R::rbeta(M + 1.0, static_cast<double>(n_obs));
  double rate = 1.0 / hyper_gamma_scale - std::log(eta);
  double mix_num = hyper_gamma_shape + static_cast<double>(n_cl) - 1.0;
  double mix_den = mix_num + static_cast<double>(n_obs) * rate;
  double prob = mix_num / mix_den;
  
  double shape = (R::runif(0.0, 1.0) < prob)
    ? hyper_gamma_shape + static_cast<double>(n_cl)
      : hyper_gamma_shape + static_cast<double>(n_cl) - 1.0;
  
  return R::rgamma(shape, 1.0 / rate);
}

void update_component_means_ddpm_cont(
    arma::vec& mu_star,
    const arma::uvec& z,
    const arma::vec& ybar,
    const arma::vec& var_y,
    const double mu_G0,
    const double tau2_G0
) {
  for (arma::uword c = 0; c < mu_star.n_elem; ++c) {
    double prec = 1.0 / tau2_G0;
    double num = mu_G0 / tau2_G0;
    
    for (arma::uword j = 0; j < z.n_elem; ++j) {
      if (z(j) == c) {
        prec += 1.0 / var_y(j);
        num  += ybar(j) / var_y(j);
      }
    }
    
    double post_var = 1.0 / prec;
    double post_mean = post_var * num;
    mu_star(c) = R::rnorm(post_mean, std::sqrt(post_var));
  }
}

void recompute_weights_ddpm_cont(
    arma::vec& w,
    const arma::vec& v
) {
  w.set_size(v.n_elem);
  double rem = 1.0;
  for (arma::uword c = 0; c < v.n_elem; ++c) {
    w(c) = rem * v(c);
    rem *= (1.0 - v(c));
  }
}

double logit_ddpm_cont(double x) {
  return std::log(x) - std::log(1.0 - x);
}

double invlogit_ddpm_cont(double x) {
  if (x > 35.0) return 1.0 - 1e-12;
  if (x < -35.0) return 1e-12;
  return 1.0 / (1.0 + std::exp(-x));
}

