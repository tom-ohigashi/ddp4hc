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

// -----------------------------------------------------------------------------
// continuous summary-data helpers for DPM_Cont.cpp
// -----------------------------------------------------------------------------

int sample_index_from_prob_cont(const arma::vec& p);

arma::uvec tabulate_alloc_cont(
    const arma::uvec& z,
    const unsigned int K
);

unsigned int n_occupied_cont(const arma::uvec& z);

double sample_dp_precision_cont(
    const double M,
    const unsigned int n_obs,
    const unsigned int n_cl,
    const double hyper_gamma_shape,
    const double hyper_gamma_scale
);

void update_component_means_cont(
    arma::vec& mu_star,
    const arma::uvec& z,
    const arma::vec& ybar,
    const arma::vec& var_y,
    const double mu_G0,
    const double tau2_G0
);

void recompute_stick_weights_cont(
    arma::vec& w,
    const arma::vec& v
);


// -----------------------------------------------------------------------------
// continuous summary-data helpers for DDPM_Cont.cpp
// -----------------------------------------------------------------------------

int sample_index_from_prob_ddpm_cont(const arma::vec& p);

unsigned int n_occupied_ddpm_cont(const arma::uvec& z);

double log_beta_fn_ddpm_cont(
    double a,
    double b
);

double sample_dp_precision_ddpm_cont(
    const double M,
    const unsigned int n_obs,
    const unsigned int n_cl,
    const double hyper_gamma_shape,
    const double hyper_gamma_scale
);

void update_component_means_ddpm_cont(
    arma::vec& mu_star,
    const arma::uvec& z,
    const arma::vec& ybar,
    const arma::vec& var_y,
    const double mu_G0,
    const double tau2_G0
);

void recompute_weights_ddpm_cont(
    arma::vec& w,
    const arma::vec& v
);

double logit_ddpm_cont(double x);

double invlogit_ddpm_cont(double x);

#endif  // OTHERS_H
