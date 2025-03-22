// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "others.h"

// [[Rcpp::export]]
Rcpp::List Simple_Linear(
    const arma::vec y,
    const arma::mat x,
    const arma::vec g,
    const double mu_beta_G0 = 0,
    const double tau_beta_G0 = 10000,
    const double alpha_sigma_G0 = 1, 
    const double beta_sigma_G0 = 1,
    const unsigned int NBURN = 4000,
    const unsigned int NTHIN = 10,
    const unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  arma::mat X = arma::join_rows(x, g);
  unsigned int ncol_x = X.n_cols, n = X.n_rows;
  
  // hyperparameter
  // base distribution
  arma::vec beta_G0 = arma::vec(ncol_x, arma::fill::value(mu_beta_G0));
  arma::vec B_G0_vec = arma::vec(ncol_x, arma::fill::value(tau_beta_G0));
  arma::mat B_G0 = tau_beta_G0*arma::mat(ncol_x, ncol_x, arma::fill::eye);
  
  // MCMC output parameters
  arma::mat beta_C_out(NOUTSAMPLE, ncol_x);
  arma::vec sigma_C_out(NOUTSAMPLE);
  
  arma::vec beta_star(ncol_x, arma::fill::zeros);
  double sigma_star = 1;
  
  unsigned int count = 0;
  
  // variables for MCMC
  unsigned int nit, diff;
  double tmp1;
  arma::vec beta_post_mu;
  arma::mat beta_post_var;
  
  // Rcpp::Rcout << "start !!"  << "\n";
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update each cluster parameter
    // update sigma by Gibbs
    tmp1 = residual2(y, X, n, beta_star);
    sigma_star = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(n/2 + alpha_sigma_G0, 1/(tmp1/2 + beta_sigma_G0)))));
    
    // update beta by Gibbs
    beta_post_var = inv((inv(B_G0) + X.t()*X/pow(sigma_star,2)));
    beta_post_mu = beta_post_var * (inv(B_G0)*beta_G0 + X.t()*y/pow(sigma_star,2));
    beta_star = arma::mvnrnd(beta_post_mu, beta_post_var);
    
    // save output
    diff = nit - NBURN + 1;
    if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
      beta_C_out.row(count) = beta_star.t();
      sigma_C_out(count) = sigma_star;
      count++;
    }
    
    // print
    if(print == 1.0){
      if(nit + 1 <= NBURN){
        if ( 10*(nit+1)/(1.0*NBURN)==round(10*(nit+1)/(NBURN)) ){
          Rcpp::Rcout << "Burn-in " << (100*(nit+1)/(NBURN)) << "% completed \n";
        }
      }else{
        if ( (10*(nit+1-NBURN)/(1.0*NTHIN*NOUTSAMPLE)) == round(10*(nit+1-NBURN)/(NTHIN*NOUTSAMPLE)) ){
          Rcpp::Rcout << "MCMC " << (100*(nit+1-NBURN)/(NTHIN*NOUTSAMPLE)) << "% completed \n";
        }
      }
    }
    
  }
  
  Rcpp::List ret = Rcpp::List::create(Rcpp::_["beta"] = beta_C_out, 
                                      Rcpp::_["sigma"] = sigma_C_out
  );
  return(ret);
}

