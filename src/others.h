#ifndef OTHERS_H
#define OTHERS_H

#include <RcppArmadillo.h>

// sampling
int sample_new_cls_logp(arma::vec proposal_logp);
int sample_new_cls_p(arma::vec proposal_p);

// table
arma::uvec tableC(arma::uvec x);
arma::uvec tableC2(arma::uvec z, unsigned int J);

// truncated normal
double ran_truncnorm(double mu, double sigma, double lower, double upper);
double den_truncnorm1(double x, double mu, double sigma, double lower, double upper);
double den_truncnorm(double x, double mu, double sigma, double lower, double upper);

// gamma
arma::vec ran_gamma(arma::vec shape, double scale);

// half-cauchy
double ran_half_cauchy(double scale);
double den_half_cauchy(double x, double scale);

// beta
arma::vec ran_beta_post(arma::vec y, arma::vec m, double a0, double b0);

// binomial log-sum
double log_sum_w_dbinom(int y, int m, arma::vec th_j, arma::vec w);

// normal residual
double residual2(const arma::vec y_C, const arma::mat x_C, const unsigned int n_C, arma::vec beta);
double log_normpdf_each(const arma::vec y_C, const arma::mat x_C, const unsigned int n_C, arma::vec beta, double sigma);
double normpdf_each(const arma::vec y_C, const arma::mat x_C, const unsigned int n_C, arma::vec beta, double sigma);

// normal log-sum
double log_sum_w_dnorm(const arma::vec y_C, const arma::mat x_C, const unsigned int n_C,
                       arma::mat beta, arma::vec sigma, arma::vec w);

double log_sum_w_dnorm2(const arma::vec y_C, const arma::mat x_C, const unsigned int n_C,
                        arma::mat beta, const double sigma, arma::vec w);

#endif  // OTHERS_H
