// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

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

