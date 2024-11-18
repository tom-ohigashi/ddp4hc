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


// MCMC sampling for DDPM method
// [[Rcpp::export]]
Rcpp::List DDPM_Linear(
    const arma::vec y_C,
    const arma::mat x_C,
    const arma::vec g_C,
    const unsigned int n_C,
    const arma::vec y_h,
    const arma::mat x_h,
    const arma::vec study_id_h,
    const arma::vec n_h,
    const double hyper_gamma_shape = 1, 
    const double hyper_gamma_scale = 1,
    const double mu_beta_G0 = 0,
    const double tau_beta_G0 = 10000,
    const double alpha_sigma_G0 = 1, 
    const double beta_sigma_G0 = 1,
    const double proposal_phi_sd = 0.1,
    const unsigned int NBURN = 4000,
    const unsigned int NTHIN = 10,
    const unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  arma::mat X_C = arma::join_rows(x_C, g_C);
  arma::vec g_h(arma::accu(n_h), arma::fill::zeros);
  arma::mat X_h = arma::join_rows(x_h, g_h);
  unsigned int ncol_x = X_h.n_cols;
  unsigned int J = study_id_h.max()+1;
  arma::uvec idHC(J-1, arma::fill::zeros);
  for(unsigned int i=0; i<(J-1); i++){
    idHC(i) = i;
  }
  unsigned int idCC = J-1;
  arma::vec study_id_h2 = study_id_h - 1;

  // hyperparameter
  // base distribution
  arma::vec beta_G0 = arma::vec(ncol_x, arma::fill::value(mu_beta_G0));
  arma::vec B_G0_vec = arma::vec(ncol_x, arma::fill::value(tau_beta_G0));
  arma::mat B_G0 = tau_beta_G0*arma::mat(ncol_x, ncol_x, arma::fill::eye);
  // DP precision parameters
  double M_DP = 1.0;
  double pi = 0.5;
  int s = 0;
  
  double phi = 0.5;
  double phip = 0.55;
  // hyperparameters for phi_gamma ~ Beta(phi_gamma1, phi_gamma2)
  double phi_gamma1 = 2.0;
  double phi_gamma2 = 2.0;
  
  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  // initialization for each cluster parameter
  arma::mat beta_star(maxcl+1, ncol_x, arma::fill::zeros);
  // arma::vec sigma_star(maxcl+1, arma::fill::ones);
  arma::mat beta_star_old(maxcl+1, ncol_x, arma::fill::zeros);
  // arma::vec sigma_star_old(maxcl+1, arma::fill::ones);
  arma::mat beta_star_add(maxcl+1, ncol_x, arma::fill::zeros);
  // arma::vec sigma_star_add(maxcl+1, arma::fill::ones);
  
  double sigma;
  
  arma::vec vHC(maxcl*3);
  arma::vec vHCp(maxcl*3);
  arma::vec vCC(maxcl*3);
  arma::vec vCCp(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    vHC(i) = R::rbeta(1.0, M_DP);
    vHCp(i) = R::rbeta(1.0, M_DP);
    vCC(i) = R::rbeta(1.0, M_DP);
    vCCp(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec wHC, wHCp, wCC, wCCp;
  wHC.zeros(maxcl+1);
  wHCp.zeros(maxcl+1);
  wCC.zeros(maxcl+1);
  wCCp.zeros(maxcl+1);
  wHC(0) = vHC(0);
  wHCp(0) = vHCp(0);
  wCC(0) = vCC(0);
  wCCp(0) = vCCp(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      wHC(c) = vHC(c);
      wHCp(c) = vHCp(c);
      wCC(c) = vCC(c);
      wCCp(c) = vCCp(c);
      for(c_=0; c_<c; c_++){
        wHC(c) *= (1-vHC(c_));
        wHCp(c) *= (1-vHCp(c_));
        wCC(c) *= (1-vCC(c_));
        wCCp(c) *= (1-vCCp(c_));
      }
    }
  }
  
  
  // MCMC output parameters
  arma::mat beta_C_out(NOUTSAMPLE, ncol_x);
  // arma::vec sigma_C_out(NOUTSAMPLE);
  arma::vec sigma_out(NOUTSAMPLE);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec phi_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;

  // variables for MCMC
  unsigned int nit, i, ii, cc, j, I1, I2, zCC, ncl_new, prop_zCC, diff, countwHC, countwCC, countth_j;
  arma::vec sumy, sumn, uHC, uCC, ll, like, proposal_p, proposal_p1, beta_aux, 
  wHC_old, wCC_old, wHC_add, wCC_add, tmpvec, beta_post_mu;
  arma::uvec zHC, indices, indices2, candi_clHC, candi_clCC, prop_zHC, 
  proposal_p_, candi_clHCj, candi_clCCj, clid0, clid00, z_old, uindices;
  double betaprior, aj, bj, lq01, lq01p, lq11, lq11p, q01, q11, q01p, q11p, p01, p01p, uu,
  lq0n, lq0np, lq1n, lq1np, q0n, q1n, p0n, q0np, q1np, p0np, sumtemp, sumtempp, 
  Jtp, Jt, lnr, prophi, uHC_star, uCC_star, wHC_, wCC_, vHC_, vCC_,
  tmp1, tmp2;
  arma::uword len1 = 0;
  arma::mat beta_post_var, tmpmat;

  // Rcpp::Rcout << "start !!"  << "\n";
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    pi = R::rbeta(M_DP+1, J);
    s = R::rbinom(1, 1-(hyper_gamma_shape+ncl-1)/(J*(hyper_gamma_scale-log(pi))+hyper_gamma_shape+ncl-1));
    M_DP = arma::randg(arma::distr_param(hyper_gamma_shape+ncl-s, 1.0/((1.0/hyper_gamma_scale)-log(pi))));
    
    
    // update each cluster parameter
    // update sigma by Gibbs
    // for(i=0; i<clid.n_elem; i++){
    //   c = clid(i);
    //   indices = arma::find(z == c) ;
    //   tmp1 = 0; tmp2 = 0;
    //   for(ii=0; ii<indices.n_elem; ii++){
    //     cc = indices(ii);
    //     if(cc == idCC){
    //       tmp2 += n_C;
    //       tmp1 += residual2(y_C, X_C, n_C, beta_star.row(c).t());
    //     }else{
    //       indices2 = arma::find(study_id_h2 == cc);
    //       tmp2 += indices2.n_elem;
    //       tmp1 += residual2(y_h(indices2), X_h.rows(indices2), indices2.n_elem, beta_star.row(c).t());
    //     }
    //   }
    //   sigma_star(c) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(tmp2/2 + alpha_sigma_G0, 1/(tmp1/2 + beta_sigma_G0)))));
    // }
    
    tmp1 = 0; tmp2 = 0;
    for(i=0; i<idHC.n_elem; i++){
      indices2 = arma::find(study_id_h2 == i);
      tmp2 += indices2.n_elem;
      tmp1 += residual2(y_h(indices2), X_h.rows(indices2), indices2.n_elem, beta_star.row(z(i)).t());
    }      
    tmp2 += n_C;
    tmp1 += residual2(y_C, X_C, n_C, beta_star.row(z(idCC)).t());
    sigma = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(tmp2/2 + alpha_sigma_G0, 1/(tmp1/2 + beta_sigma_G0)))));
    
    // update beta by Gibbs
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      tmpvec.zeros(0);
      tmpmat.zeros(0);
      uindices.zeros(0);
      for(ii=0; ii<indices.n_elem; ii++){
        uindices = arma::join_cols(uindices, arma::find(study_id_h2 == indices(ii)));
      }
      if(arma::any(indices == idCC)){
        tmpmat = arma::join_cols(X_h.rows(uindices), X_C);
        tmpvec = arma::join_cols(y_h(uindices), y_C);
      }else{
        tmpmat = X_h.rows(uindices);
        tmpvec = y_h(uindices);
      }
      // beta_post_var = inv(inv(B_G0) + tmpmat.t()*tmpmat/pow(sigma_star(c),2));
      // beta_post_mu = beta_post_var * (inv(B_G0)*beta_G0 + tmpmat.t()*tmpvec/pow(sigma_star(c),2));
      beta_post_var = inv(inv(B_G0) + tmpmat.t()*tmpmat/pow(sigma,2));
      beta_post_mu = beta_post_var * (inv(B_G0)*beta_G0 + tmpmat.t()*tmpvec/pow(sigma,2));
      beta_star.row(c) = arma::mvnrnd(beta_post_mu, beta_post_var).t();
    }
    
    phip = ran_truncnorm(phi, proposal_phi_sd, 0, 1);
    for(c=0; c<(maxcl+1); c++){
      if(m(c) == 0){
        betaprior = R::rbeta(1.0, M_DP);
        vHC(c) = betaprior;
        vHCp(c) = betaprior;
      }else{
        zHC = z.subvec(0,max(idHC));
        indices = arma::find((zHC) == c);
        indices2 = arma::find((zHC) > c);
        I1 = indices.n_elem;
        I2 = indices2.n_elem;
        aj = 1 + I1;
        bj = M_DP + I2;
        lq01 = log(phi) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCC(c)) - lgamma(aj+bj);
        lq01p = log(phip) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCCp(c)) - lgamma(aj+bj);
        lq11 = log(1-phi) + log(M_DP) + I1*log(vCC(c)) + (M_DP-1+I2)*log(1-vCC(c));
        lq11p = log(1-phip) + log(M_DP) + I1*log(vCCp(c)) + (M_DP-1+I2)*log(1-vCCp(c));
        q01 = exp(lq01);
        q11 = exp(lq11);
        q01p = exp(lq01p);
        q11p = exp(lq11p);
        p01 = q01/(q01+q11);
        p01p = q01p/(q01p+q11p);
        uu = R::runif(0.0, 1.0);
        if(p01>uu){
          vHC(c) = R::rbeta(aj, bj);
        }else{
          vHC(c) = vCC(c);
        }
        if(p01p>uu){
          vHCp(c) = R::rbeta(aj, bj);
        }else{
          vHCp(c) = vCCp(c);
        }
      }
    }
    for(c=0; c<(maxcl+1); c++){
      if(vHC(c) >= 0.9999){
        vHC(c) = 0.9999;
      }
      if(vHCp(c) >= 0.9999){
        vHCp(c) = 0.9999;
      }
    }
    
    // update v for current control
    for(c=0; c<(maxcl+1); c++){
      if(m(c) == 0){
        betaprior = R::rbeta(1.0, M_DP);
        vCC(c) = betaprior;
        vCCp(c) = betaprior;
      }else{
        zCC = z(idCC);
        I1 = ((zCC)==c)*1;
        I2 = ((zCC)>c)*1;
        aj = 1 + I1;
        bj = M_DP + I2;
        lq0n = log(phi) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
        lq0np = log(phip) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
        lq1n = log(1-phi) + I1*log(vHC(c)) + I2*log(1-vHC(c));
        lq1np = log(1-phip) + I1*log(vHCp(c)) + I2*log(1-vHCp(c));
        q0n = exp(lq0n);
        q1n = exp(lq1n);
        p0n = q0n/(q0n+q1n);
        q0np = exp(lq0np);
        q1np = exp(lq1np);
        p0np = q0np/(q0np+q1np);
        uu = R::runif(0.0, 1.0);
        if(p0n>uu){
          vCC(c) = R::rbeta(aj, bj);
        }else{
          vCC(c) = vHC(c);
        }
        if(p0np>uu){
          vCCp(c) = R::rbeta(aj, bj);
        }else{
          vCCp(c) = vHCp(c);
        }
      }
    }
    for(c=0; c<(maxcl+1); c++){
      if(vCC(c) >= 0.9999){
        vCC(c) = 0.9999;
      }
      if(vCCp(c) >= 0.9999){
        vCCp(c) = 0.9999;
      }
    }
    
    // update weight
    wHC.zeros(maxcl+1);
    wHCp.zeros(maxcl+1);
    wCC.zeros(maxcl+1);
    wCCp.zeros(maxcl+1);
    wHC(0) = vHC(0);
    wHCp(0) = vHCp(0);
    wCC(0) = vCC(0);
    wCCp(0) = vCCp(0);
    if(maxcl != 0){
      for(c=1; c<(maxcl+1);c++){
        wHC(c) = vHC(c);
        wHCp(c) = vHCp(c);
        wCC(c) = vCC(c);
        wCCp(c) = vCCp(c);
        for(c_=0; c_<c; c_++){
          wHC(c) *= (1-vHC(c_));
          wHCp(c) *= (1-vHCp(c_));
          wCC(c) *= (1-vCC(c_));
          wCCp(c) *= (1-vCCp(c_));
        }
      }
    }
    
    // update phi
    sumtemp = 0;
    sumtempp = 0;
    for(i=0; i<idHC.n_elem; i++){
      indices2 = arma::find(study_id_h2 == i);
      tmp2 = indices2.n_elem;
      // sumtemp += log_sum_w_dnorm(y_h(indices2), X_h.rows(indices2), tmp2, beta_star, sigma_star, wHC);
      // sumtempp += log_sum_w_dnorm(y_h(indices2), X_h.rows(indices2), tmp2, beta_star, sigma_star, wHCp);
      sumtemp += log_sum_w_dnorm2(y_h(indices2), X_h.rows(indices2), tmp2, beta_star, sigma, wHC);
      sumtempp += log_sum_w_dnorm2(y_h(indices2), X_h.rows(indices2), tmp2, beta_star, sigma, wHCp);
    }
    // sumtemp += log_sum_w_dnorm(y_C, X_C, n_C, beta_star, sigma_star, wCC);
    // sumtempp += log_sum_w_dnorm(y_C, X_C, n_C, beta_star, sigma_star, wCCp);
    sumtemp += log_sum_w_dnorm2(y_C, X_C, n_C, beta_star, sigma, wCC);
    sumtempp += log_sum_w_dnorm2(y_C, X_C, n_C, beta_star, sigma, wCCp);
    Jtp = den_truncnorm(phip, phi, proposal_phi_sd, 0, 1);
    Jt = den_truncnorm(phi, phip, proposal_phi_sd, 0, 1);
    lnr = ((phi_gamma1-1)*log(phip) + (phi_gamma2-1)*log(1-phip) + sumtempp) - log(Jtp)
      - ((phi_gamma1-1)*log(phi) + (phi_gamma2-1)*log(1-phi) + sumtemp) + log(Jt);
    prophi = exp(std::min(lnr, 0.0));
    uu = R::runif(0.0, 1.0);
    if(prophi >= uu){
      phi = phip;
      wHC = wHCp;
      wCC = wCCp;
    }else{
      phi = phi;
      wHC = wHC;
      wCC = wCC;
    }
    
    // update u
    uHC.zeros(idHC.n_elem);
    uCC.zeros(1);
    for(i=0; i<idHC.n_elem; i++){
      uHC(i) = R::runif(0.0, 1.0) * wHC(z(i));
    }
    uCC(0) = R::runif(0.0, 1.0) * wCC(z(idCC));
    uHC_star = arma::min(uHC);
    uCC_star = uCC(0);
    
    
    // candidates for new clusters
    wHC_ = 1 - sum(wHC);
    wCC_ = 1 - sum(wCC);
    clid0 = arma::find(m == 0);
    if(clid0.n_elem > 0){
      wHC_old = wHC;
      wCC_old = wCC;
      beta_star_old = beta_star;
      // sigma_star_old = sigma_star;
      if(max(clid0) >= wHC.n_elem){
        len1 = max(clid0) - wHC.n_elem + 1;
        wHC_add.zeros(len1);
        wCC_add.zeros(len1);
        beta_star_add.zeros(len1, ncol_x);
        // sigma_star_add.ones(len1);
      }
      countwHC = 0;
      countwCC = 0;
      countth_j = 0;
      ncl_new = 0;
      
      while ( ((wHC_ > uHC_star) || (wCC_ > uCC_star)) && (ncl_new < clid0.n_elem)  ){
        vHC_ = R::rbeta(1.0, M_DP);
        if(clid0(ncl_new) < wHC.n_elem){
          wHC_old(clid0(ncl_new)) = wHC_*vHC_;
        }else{
          wHC_add(countwHC) = wHC_*vHC_;
          countwHC++;
        }
        wHC_ *= (1-vHC_);
        
        vCC_ = R::rbeta(1.0, M_DP);
        if(clid0(ncl_new) < wCC.n_elem){
          wCC_old(clid0(ncl_new)) = wCC_*vCC_;
        }else{
          wCC_add(countwCC) = wCC_*vCC_;
          countwCC++;
        }
        wCC_ = wCC_*(1-vCC_);
        
        ncl_new++;
      }
      
      if(max(clid0) >= wHC_old.n_elem){
        wHC = arma::join_cols(wHC_old, wHC_add);
        wCC = arma::join_cols(wCC_old, wCC_add);
      }else{
        wHC = wHC_old;
        wCC = wCC_old;
      }
      
      if(ncl_new != 0){
        clid00 = clid0(arma::span(0, (ncl_new-1)));
        for(i=0; i<clid00.n_elem; i++){
          j = clid00(i);
          if(j < beta_star_old.n_rows){
            beta_star_old.row(j) = arma::mvnrnd(beta_G0, B_G0).t();
            // sigma_star_old(j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
          }else{
            beta_star_add.row(countth_j) = arma::mvnrnd(beta_G0, B_G0).t();
            // sigma_star_add(countth_j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
            countth_j++;
          }
        }
        if(max(clid00) >= beta_star_old.n_rows){
          beta_star = arma::join_cols(beta_star_old, beta_star_add);
          // sigma_star = arma::join_cols(sigma_star_old, sigma_star_add);
        }else{
          beta_star = beta_star_old;
          // sigma_star = sigma_star_old;
        }
      }
    }
    
    
    // update z
    candi_clHC = arma::find(wHC != 0);
    candi_clCC = arma::find(wCC != 0);
    prop_zHC.zeros(idHC.n_elem);
    prop_zCC = 0;
    z_old = z;
    
    if(candi_clHC.n_elem > 1){
      for(i=0; i<idHC.n_elem; i++){
        ll.zeros(candi_clHC.n_elem);
        like.zeros(candi_clHC.n_elem);
        indices2 = arma::find(study_id_h2 == i);
        tmp2 = indices2.n_elem;
        for(c=0; c<candi_clHC.n_elem; c++){
          ll(c) = (uHC(i) < wHC(candi_clHC(c)))*1;
          // like(c) = normpdf_each(y_h(indices2), X_h.rows(indices2), tmp2, beta_star.row(candi_clHC(c)).t(), sigma_star(candi_clHC(c)));
          like(c) = normpdf_each(y_h(indices2), X_h.rows(indices2), tmp2, beta_star.row(candi_clHC(c)).t(), sigma);
        }
        proposal_p = like % ll;
        proposal_p_ = arma::find(proposal_p != 0);
        candi_clHCj = candi_clHC(proposal_p_);
        proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
        len1 = candi_clHCj.n_elem;
        if(len1 == 1){
          prop_zHC(i) = candi_clHCj(0);
        }else if(len1 == 0){
          prop_zHC(i) = z_old(i);
        }else{
          prop_zHC(i) = candi_clHCj(sample_new_cls_p(proposal_p1));
        }
      }
    }else{
      prop_zHC.fill(candi_clHC(0));
    }
    for(i=0; i<idHC.n_elem; i++){
      z(i) = prop_zHC(i);
    }
    
    if(candi_clCC.n_elem > 1){
      ll.zeros(candi_clCC.n_elem);
      like.zeros(candi_clCC.n_elem);
      for(c=0; c<candi_clCC.n_elem; c++){
        ll(c) = (uCC(0) < wCC(candi_clCC(c)))*1;
        // like(c) = normpdf_each(y_C, X_C, n_C, beta_star.row(candi_clCC(c)).t(), sigma_star(candi_clCC(c)));
        like(c) = normpdf_each(y_C, X_C, n_C, beta_star.row(candi_clCC(c)).t(), sigma);
      }
      proposal_p = like % ll;
      proposal_p_ = arma::find(proposal_p != 0);
      candi_clCCj = candi_clCC(proposal_p_);
      proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
      len1 = candi_clCCj.n_elem;
      if(len1 == 1){
        prop_zCC = candi_clCCj(0);
      }else if(len1 == 0){
        prop_zCC = z_old(idCC);
      }else{
        prop_zCC = candi_clCCj(sample_new_cls_p(proposal_p1));
      }
    }else{
      prop_zCC = candi_clCC(0);
    }
    z(idCC) = prop_zCC;
    
    unique_z = arma::unique(z);
    ncl = unique_z.n_elem;
    maxcl = max(arma::unique(z));
    m = tableC2(z, J);
    clid = arma::find(m != 0);
    
    // save output
    diff = nit - NBURN + 1;
    if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
      beta_C_out.row(count) = beta_star.row(z(idCC));
      // sigma_C_out(count) = sigma_star(z(idCC));
      sigma_out(count) = sigma;
      
      z_out.row(count) = z.t() + 1;
      indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
      M_DP_out(count) = M_DP;
      phi_out(count) = phi;
      ncl_out(count) = ncl;
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
  
  // Rcpp::_["sigma_C"] = sigma_C_out, 
    Rcpp::List ret = Rcpp::List::create(Rcpp::_["beta_C"] = beta_C_out, 
                                      Rcpp::_["sigma_C"] = sigma_out, 
                                      Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["phi"] = phi_out,
                                      Rcpp::_["ncl"] = ncl_out
  );
  return(ret);
}


// MCMC sampling for DDPM method
// [[Rcpp::export]]
Rcpp::List DDPM_Linear_Ca(
    const arma::vec y_C,
    const arma::mat x_C,
    const arma::vec g_C,
    const unsigned int n_C,
    const arma::vec y_h,
    const arma::mat x_h,
    const arma::vec study_id_h,
    const arma::vec n_h,
    const double hcauchy_scale = 1,
    const double mu_beta_G0 = 0,
    const double tau_beta_G0 = 10000,
    const double alpha_sigma_G0 = 1, 
    const double beta_sigma_G0 = 1,
    const double proposal_phi_sd = 0.1,
    const double proposal_M_DP_sd = 1,
    const unsigned int NBURN = 4000,
    const unsigned int NTHIN = 10,
    const unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  arma::mat X_C = arma::join_rows(x_C, g_C);
  arma::vec g_h(arma::accu(n_h), arma::fill::zeros);
  arma::mat X_h = arma::join_rows(x_h, g_h);
  unsigned int ncol_x = X_h.n_cols;
  unsigned int J = study_id_h.max()+1;
  arma::uvec idHC(J-1, arma::fill::zeros);
  for(unsigned int i=0; i<(J-1); i++){
    idHC(i) = i;
  }
  unsigned int idCC = J-1;
  arma::vec study_id_h2 = study_id_h - 1;
  
  // hyperparameter
  // base distribution
  arma::vec beta_G0 = arma::vec(ncol_x, arma::fill::value(mu_beta_G0));
  arma::vec B_G0_vec = arma::vec(ncol_x, arma::fill::value(tau_beta_G0));
  arma::mat B_G0 = tau_beta_G0*arma::mat(ncol_x, ncol_x, arma::fill::eye);
  // DP precision parameters
  double M_DP = 1.0;

  double phi = 0.5;
  double phip = 0.55;
  // hyperparameters for phi_gamma ~ Beta(phi_gamma1, phi_gamma2)
  double phi_gamma1 = 2.0;
  double phi_gamma2 = 2.0;
  
  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  // initialization for each cluster parameter
  arma::mat beta_star(maxcl+1, ncol_x, arma::fill::zeros);
  arma::vec sigma_star(maxcl+1, arma::fill::ones);
  arma::mat beta_star_old(maxcl+1, ncol_x, arma::fill::zeros);
  arma::vec sigma_star_old(maxcl+1, arma::fill::ones);
  arma::mat beta_star_add(maxcl+1, ncol_x, arma::fill::zeros);
  arma::vec sigma_star_add(maxcl+1, arma::fill::ones);
  
  arma::vec vHC(maxcl*3);
  arma::vec vHCp(maxcl*3);
  arma::vec vCC(maxcl*3);
  arma::vec vCCp(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    vHC(i) = R::rbeta(1.0, M_DP);
    vHCp(i) = R::rbeta(1.0, M_DP);
    vCC(i) = R::rbeta(1.0, M_DP);
    vCCp(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec wHC, wHCp, wCC, wCCp;
  wHC.zeros(maxcl+1);
  wHCp.zeros(maxcl+1);
  wCC.zeros(maxcl+1);
  wCCp.zeros(maxcl+1);
  wHC(0) = vHC(0);
  wHCp(0) = vHCp(0);
  wCC(0) = vCC(0);
  wCCp(0) = vCCp(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      wHC(c) = vHC(c);
      wHCp(c) = vHCp(c);
      wCC(c) = vCC(c);
      wCCp(c) = vCCp(c);
      for(c_=0; c_<c; c_++){
        wHC(c) *= (1-vHC(c_));
        wHCp(c) *= (1-vHCp(c_));
        wCC(c) *= (1-vCC(c_));
        wCCp(c) *= (1-vCCp(c_));
      }
    }
  }
  
  
  // MCMC output parameters
  arma::vec gamma_out(NOUTSAMPLE);
  arma::mat beta_C_out(NOUTSAMPLE, ncol_x);
  arma::vec sigma_C_out(NOUTSAMPLE);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec phi_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;

  // variables for MCMC
  unsigned int nit, i, ii, cc, j, I1, I2, zCC, ncl_new, prop_zCC, diff, countwHC, countwCC, countth_j;
  arma::vec sumy, sumn, uHC, uCC, ll, like, proposal_p, proposal_p1, beta_aux, 
  wHC_old, wCC_old, wHC_add, wCC_add, tmpvec, beta_post_mu;
  arma::uvec zHC, indices, indices2, candi_clHC, candi_clCC, prop_zHC, 
  proposal_p_, candi_clHCj, candi_clCCj, clid0, clid00, z_old, uindices;
  double betaprior, aj, bj, lq01, lq01p, lq11, lq11p, q01, q11, q01p, q11p, p01, p01p, uu,
  lq0n, lq0np, lq1n, lq1np, q0n, q1n, p0n, q0np, q1np, p0np, sumtemp, sumtempp, 
  Jtp, Jt, lnr, prophi, uHC_star, uCC_star, wHC_, wCC_, vHC_, vCC_,
  tmp1, tmp2, M_DP_prop, lr;
  arma::uword len1 = 0;
  arma::mat beta_post_var, tmpmat;
  
  // Rcpp::Rcout << "start !!"  << "\n";
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    M_DP_prop = (M_DP + arma::randu() *2.0 - 1.0) * proposal_M_DP_sd;
    if(M_DP_prop < 0){
      M_DP_prop = -M_DP_prop;
    }
    lr = ncl * log(M_DP_prop) + lgamma(M_DP_prop) + log(den_half_cauchy(M_DP_prop, hcauchy_scale))
      - lgamma(M_DP_prop + J)
      - ncl * log(M_DP) - lgamma(M_DP) - log(den_half_cauchy(M_DP, hcauchy_scale))
      + lgamma(M_DP + J);
      if(log(R::runif(0.0, 1.0))<lr){
        M_DP = M_DP_prop;
      }

    
    // update each cluster parameter
    // update sigma by Gibbs
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      tmp1 = 0; tmp2 = 0;
      for(ii=0; ii<indices.n_elem; ii++){
        cc = indices(ii);
        if(cc == idCC){
          tmp2 += n_C;
          tmp1 += residual2(y_C, X_C, n_C, beta_star.row(c).t());
        }else{
          indices2 = arma::find(study_id_h2 == cc);
          tmp2 += indices2.n_elem;
          tmp1 += residual2(y_h(indices2), X_h.rows(indices2), indices2.n_elem, beta_star.row(c).t());
        }
      }
      sigma_star(c) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(tmp2/2 + alpha_sigma_G0, 1/(tmp1/2 + beta_sigma_G0)))));
    }
    
    // update beta by Gibbs
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      tmpvec.zeros(0);
      tmpmat.zeros(0);
      uindices.zeros(0);
      for(ii=0; ii<indices.n_elem; ii++){
        uindices = arma::join_cols(uindices, arma::find(study_id_h2 == indices(ii)));
      }
      if(arma::any(indices == idCC)){
        tmpmat = arma::join_cols(X_h.rows(uindices), X_C);
        tmpvec = arma::join_cols(y_h(uindices), y_C);
      }else{
        tmpmat = X_h.rows(uindices);
        tmpvec = y_h(uindices);
      }
      beta_post_var = inv(inv(B_G0) + tmpmat.t()*tmpmat/pow(sigma_star(c),2));
      beta_post_mu = beta_post_var * (inv(B_G0)*beta_G0 + tmpmat.t()*tmpvec/pow(sigma_star(c),2));
      beta_star.row(c) = arma::mvnrnd(beta_post_mu, beta_post_var).t();
    }
    
    phip = ran_truncnorm(phi, proposal_phi_sd, 0, 1);
      for(c=0; c<(maxcl+1); c++){
        if(m(c) == 0){
          betaprior = R::rbeta(1.0, M_DP);
          vHC(c) = betaprior;
          vHCp(c) = betaprior;
        }else{
          zHC = z.subvec(0,max(idHC));
          indices = arma::find((zHC) == c);
          indices2 = arma::find((zHC) > c);
          I1 = indices.n_elem;
          I2 = indices2.n_elem;
          aj = 1 + I1;
          bj = M_DP + I2;
          lq01 = log(phi) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCC(c)) - lgamma(aj+bj);
          lq01p = log(phip) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCCp(c)) - lgamma(aj+bj);
          lq11 = log(1-phi) + log(M_DP) + I1*log(vCC(c)) + (M_DP-1+I2)*log(1-vCC(c));
          lq11p = log(1-phip) + log(M_DP) + I1*log(vCCp(c)) + (M_DP-1+I2)*log(1-vCCp(c));
          q01 = exp(lq01);
          q11 = exp(lq11);
          q01p = exp(lq01p);
          q11p = exp(lq11p);
          p01 = q01/(q01+q11);
          p01p = q01p/(q01p+q11p);
          uu = R::runif(0.0, 1.0);
          if(p01>uu){
            vHC(c) = R::rbeta(aj, bj);
          }else{
            vHC(c) = vCC(c);
          }
          if(p01p>uu){
            vHCp(c) = R::rbeta(aj, bj);
          }else{
            vHCp(c) = vCCp(c);
          }
        }
      }
      for(c=0; c<(maxcl+1); c++){
        if(vHC(c) >= 0.9999){
          vHC(c) = 0.9999;
        }
        if(vHCp(c) >= 0.9999){
          vHCp(c) = 0.9999;
        }
      }
      
      // update v for current control
      for(c=0; c<(maxcl+1); c++){
        if(m(c) == 0){
          betaprior = R::rbeta(1.0, M_DP);
          vCC(c) = betaprior;
          vCCp(c) = betaprior;
        }else{
          zCC = z(idCC);
          I1 = ((zCC)==c)*1;
          I2 = ((zCC)>c)*1;
          aj = 1 + I1;
          bj = M_DP + I2;
          lq0n = log(phi) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
          lq0np = log(phip) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
          lq1n = log(1-phi) + I1*log(vHC(c)) + I2*log(1-vHC(c));
          lq1np = log(1-phip) + I1*log(vHCp(c)) + I2*log(1-vHCp(c));
          q0n = exp(lq0n);
          q1n = exp(lq1n);
          p0n = q0n/(q0n+q1n);
          q0np = exp(lq0np);
          q1np = exp(lq1np);
          p0np = q0np/(q0np+q1np);
          uu = R::runif(0.0, 1.0);
          if(p0n>uu){
            vCC(c) = R::rbeta(aj, bj);
          }else{
            vCC(c) = vHC(c);
          }
          if(p0np>uu){
            vCCp(c) = R::rbeta(aj, bj);
          }else{
            vCCp(c) = vHCp(c);
          }
        }
      }
      for(c=0; c<(maxcl+1); c++){
        if(vCC(c) >= 0.9999){
          vCC(c) = 0.9999;
        }
        if(vCCp(c) >= 0.9999){
          vCCp(c) = 0.9999;
        }
      }
      
      // update weight
      wHC.zeros(maxcl+1);
      wHCp.zeros(maxcl+1);
      wCC.zeros(maxcl+1);
      wCCp.zeros(maxcl+1);
      wHC(0) = vHC(0);
      wHCp(0) = vHCp(0);
      wCC(0) = vCC(0);
      wCCp(0) = vCCp(0);
      if(maxcl != 0){
        for(c=1; c<(maxcl+1);c++){
          wHC(c) = vHC(c);
          wHCp(c) = vHCp(c);
          wCC(c) = vCC(c);
          wCCp(c) = vCCp(c);
          for(c_=0; c_<c; c_++){
            wHC(c) *= (1-vHC(c_));
            wHCp(c) *= (1-vHCp(c_));
            wCC(c) *= (1-vCC(c_));
            wCCp(c) *= (1-vCCp(c_));
          }
        }
      }
      
      // update phi
      sumtemp = 0;
      sumtempp = 0;
      for(i=0; i<idHC.n_elem; i++){
        indices2 = arma::find(study_id_h2 == i);
        tmp2 = indices2.n_elem;
        sumtemp += log_sum_w_dnorm(y_h(indices2), X_h.rows(indices2), tmp2, beta_star, sigma_star, wHC);
        sumtempp += log_sum_w_dnorm(y_h(indices2), X_h.rows(indices2), tmp2, beta_star, sigma_star, wHCp);
      }
      sumtemp += log_sum_w_dnorm(y_C, X_C, n_C, beta_star, sigma_star, wCC);
      sumtempp += log_sum_w_dnorm(y_C, X_C, n_C, beta_star, sigma_star, wCCp);
      Jtp = den_truncnorm(phip, phi, proposal_phi_sd, 0, 1);
      Jt = den_truncnorm(phi, phip, proposal_phi_sd, 0, 1);
      lnr = ((phi_gamma1-1)*log(phip) + (phi_gamma2-1)*log(1-phip) + sumtempp) - log(Jtp)
        - ((phi_gamma1-1)*log(phi) + (phi_gamma2-1)*log(1-phi) + sumtemp) + log(Jt);
      prophi = exp(std::min(lnr, 0.0));
      uu = R::runif(0.0, 1.0);
      if(prophi >= uu){
        phi = phip;
        wHC = wHCp;
        wCC = wCCp;
      }else{
        phi = phi;
        wHC = wHC;
        wCC = wCC;
      }
      
      // update u
      uHC.zeros(idHC.n_elem);
      uCC.zeros(1);
      for(i=0; i<idHC.n_elem; i++){
        uHC(i) = R::runif(0.0, 1.0) * wHC(z(i));
      }
      uCC(0) = R::runif(0.0, 1.0) * wCC(z(idCC));
      uHC_star = arma::min(uHC);
      uCC_star = uCC(0);
      
      
      // candidates for new clusters
      wHC_ = 1 - sum(wHC);
      wCC_ = 1 - sum(wCC);
      clid0 = arma::find(m == 0);
      if(clid0.n_elem > 0){
        wHC_old = wHC;
        wCC_old = wCC;
        beta_star_old = beta_star;
        sigma_star_old = sigma_star;
        if(max(clid0) >= wHC.n_elem){
          len1 = max(clid0) - wHC.n_elem + 1;
          wHC_add.zeros(len1);
          wCC_add.zeros(len1);
          beta_star_add.zeros(len1, ncol_x);
          sigma_star_add.ones(len1);
        }
        countwHC = 0;
        countwCC = 0;
        countth_j = 0;
        ncl_new = 0;
        
        while ( ((wHC_ > uHC_star) || (wCC_ > uCC_star)) && (ncl_new < clid0.n_elem)  ){
          vHC_ = R::rbeta(1.0, M_DP);
          if(clid0(ncl_new) < wHC.n_elem){
            wHC_old(clid0(ncl_new)) = wHC_*vHC_;
          }else{
            wHC_add(countwHC) = wHC_*vHC_;
            countwHC++;
          }
          wHC_ *= (1-vHC_);
          
          vCC_ = R::rbeta(1.0, M_DP);
          if(clid0(ncl_new) < wCC.n_elem){
            wCC_old(clid0(ncl_new)) = wCC_*vCC_;
          }else{
            wCC_add(countwCC) = wCC_*vCC_;
            countwCC++;
          }
          wCC_ = wCC_*(1-vCC_);
          
          ncl_new++;
        }
        
        if(max(clid0) >= wHC_old.n_elem){
          wHC = arma::join_cols(wHC_old, wHC_add);
          wCC = arma::join_cols(wCC_old, wCC_add);
        }else{
          wHC = wHC_old;
          wCC = wCC_old;
        }
        
        if(ncl_new != 0){
          clid00 = clid0(arma::span(0, (ncl_new-1)));
          for(i=0; i<clid00.n_elem; i++){
            j = clid00(i);
            if(j < beta_star_old.n_rows){
              beta_star_old.row(j) = arma::mvnrnd(beta_G0, B_G0).t();
              sigma_star_old(j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
            }else{
              beta_star_add.row(countth_j) = arma::mvnrnd(beta_G0, B_G0).t();
              sigma_star_add(countth_j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
              countth_j++;
            }
          }
          if(max(clid00) >= beta_star_old.n_rows){
            beta_star = arma::join_cols(beta_star_old, beta_star_add);
            sigma_star = arma::join_cols(sigma_star_old, sigma_star_add);
          }else{
            beta_star = beta_star_old;
            sigma_star = sigma_star_old;
          }
        }
      }
      
      
      // update z
      candi_clHC = arma::find(wHC != 0);
      candi_clCC = arma::find(wCC != 0);
      prop_zHC.zeros(idHC.n_elem);
      prop_zCC = 0;
      z_old = z;
      z.zeros(J);
      if(candi_clHC.n_elem > 1){
        for(i=0; i<idHC.n_elem; i++){
          ll.zeros(candi_clHC.n_elem);
          like.zeros(candi_clHC.n_elem);
          indices2 = arma::find(study_id_h2 == i);
          tmp2 = indices2.n_elem;
          for(c=0; c<candi_clHC.n_elem; c++){
            ll(c) = (uHC(i) < wHC(candi_clHC(c)))*1;
            like(c) = normpdf_each(y_h(indices2), X_h.rows(indices2), tmp2, beta_star.row(candi_clHC(c)).t(), sigma_star(candi_clHC(c)));
          }
          proposal_p = like % ll;
          proposal_p_ = arma::find(proposal_p != 0);
          candi_clHCj = candi_clHC(proposal_p_);
          proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
          len1 = candi_clHCj.n_elem;
          if(len1 == 1){
            prop_zHC(i) = candi_clHCj(0);
          }else if(len1 == 0){
            prop_zHC(i) = z_old(i);
          }else{
            prop_zHC(i) = candi_clHCj(sample_new_cls_p(proposal_p1));
          }
        }
      }else{
        prop_zHC.fill(candi_clHC(0));
      }
      for(i=0; i<idHC.n_elem; i++){
        z(i) = prop_zHC(i);
      }
      
      if(candi_clCC.n_elem > 1){
        ll.zeros(candi_clCC.n_elem);
        like.zeros(candi_clCC.n_elem);
        for(c=0; c<candi_clCC.n_elem; c++){
          ll(c) = (uCC(0) < wCC(candi_clCC(c)))*1;
          like(c) = normpdf_each(y_C, X_C, n_C, beta_star.row(candi_clCC(c)).t(), sigma_star(candi_clCC(c)));
        }
        proposal_p = like % ll;
        proposal_p_ = arma::find(proposal_p != 0);
        candi_clCCj = candi_clCC(proposal_p_);
        proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
        len1 = candi_clCCj.n_elem;
        if(len1 == 1){
          prop_zCC = candi_clCCj(0);
        }else if(len1 == 0){
          prop_zCC = z_old(idCC);
        }else{
          prop_zCC = candi_clCCj(sample_new_cls_p(proposal_p1));
        }
      }else{
        prop_zCC = candi_clCC(0);
      }
      z(idCC) = prop_zCC;
      
      unique_z = arma::unique(z);
      ncl = unique_z.n_elem;
      maxcl = max(arma::unique(z));
      m = tableC2(z, J);
      clid = arma::find(m != 0);
      
    // save output
    diff = nit - NBURN + 1;
    if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
      beta_C_out.row(count) = beta_star.row(z(idCC));
      sigma_C_out(count) = sigma_star(z(idCC));
      
      z_out.row(count) = z.t() + 1;
      indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
      M_DP_out(count) = M_DP;
      phi_out(count) = phi;
      ncl_out(count) = ncl;
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
  
  Rcpp::List ret = Rcpp::List::create(Rcpp::_["beta_C"] = beta_C_out, 
                                      Rcpp::_["sigma_C"] = sigma_C_out, 
                                      Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["phi"] = phi_out,
                                      Rcpp::_["ncl"] = ncl_out
  );
  return(ret);
}



// MCMC sampling for DPM method
// [[Rcpp::export]]
Rcpp::List DPM_Linear(
    const arma::vec y_C,
    const arma::mat x_C,
    const arma::vec g_C,
    const unsigned int n_C,
    const arma::vec y_h,
    const arma::mat x_h,
    const arma::vec study_id_h,
    const arma::vec n_h,
    const double hyper_gamma_shape = 1, 
    const double hyper_gamma_scale = 1,
    const double mu_beta_G0 = 0,
    const double tau_beta_G0 = 10000,
    const double alpha_sigma_G0 = 1, 
    const double beta_sigma_G0 = 1,
    const unsigned int NBURN = 4000,
    const unsigned int NTHIN = 10,
    const unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  arma::mat X_C = arma::join_rows(x_C, g_C);
  arma::vec g_h(arma::accu(n_h), arma::fill::zeros);
  arma::mat X_h = arma::join_rows(x_h, g_h);
  unsigned int ncol_x = X_h.n_cols;
  unsigned int J = study_id_h.max()+1;
  arma::uvec id(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    id(i) = i;
  }
  arma::vec study_id_h2 = study_id_h - 1;
  
  // hyperparameter
  // base distribution
  arma::vec beta_G0 = arma::vec(ncol_x, arma::fill::value(mu_beta_G0));
  arma::vec B_G0_vec = arma::vec(ncol_x, arma::fill::value(tau_beta_G0));
  arma::mat B_G0 = tau_beta_G0*arma::mat(ncol_x, ncol_x, arma::fill::eye);
  // DP precision parameters
  double M_DP = 1.0;
  double pi = 0.5;
  int s = 0;
  
  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  // initialization for each cluster parameter
  arma::mat beta_star(maxcl+1, ncol_x, arma::fill::zeros);
  // arma::vec sigma_star(maxcl+1, arma::fill::ones);
  arma::mat beta_star_old(maxcl+1, ncol_x, arma::fill::zeros);
  // arma::vec sigma_star_old(maxcl+1, arma::fill::ones);
  arma::mat beta_star_add(maxcl+1, ncol_x, arma::fill::zeros);
  // arma::vec sigma_star_add(maxcl+1, arma::fill::ones);
  
  double sigma;
  
  arma::vec v(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    v(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec w;
  w.zeros(maxcl+1);
  w(0) = v(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      w(c) = v(c);
      for(c_=0; c_<c; c_++){
        w(c) *= (1-v(c_));
      }
    }
  }
  double w_ = v(ncl); // weight for potential new clusters
  
  // MCMC output parameters
  arma::mat beta_C_out(NOUTSAMPLE, ncol_x);
  // arma::vec sigma_C_out(NOUTSAMPLE);
  arma::vec sigma_out(NOUTSAMPLE);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;

  // variables for MCMC
  unsigned int nit, i, ii, cc, j, ncl_new, diff, countw, countth_j;
  arma::vec sumy, sumn, v1, u, ll, like, proposal_p, proposal_p1, beta_aux, 
  w_old, w_add, tmpvec, beta_post_mu;
  arma::uvec indices, indices2, candi_cl, candi_clj, prop_z, 
  proposal_p_, clid0, clid00, z_old, uindices;
  double u_star, v_, tmp1, tmp2;
  arma::uword len1 = 0;
  arma::mat beta_post_var, tmpmat;
  
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    pi = R::rbeta(M_DP+1, J);
    s = R::rbinom(1, 1-(hyper_gamma_shape+ncl-1)/(J*(hyper_gamma_scale-log(pi))+hyper_gamma_shape+ncl-1));
    M_DP = arma::randg(arma::distr_param(hyper_gamma_shape+ncl-s, 1.0/((1.0/hyper_gamma_scale)-log(pi))));
    
    // update each cluster parameter
    // update sigma by Gibbs
    // for(i=0; i<clid.n_elem; i++){
    //   c = clid(i);
    //   indices = arma::find(z == c) ;
    //   tmp1 = 0; tmp2 = 0;
    //   for(ii=0; ii<indices.n_elem; ii++){
    //     cc = indices(ii);
    //     if(cc == id(J-1)){
    //       tmp2 += n_C;
    //       tmp1 += residual2(y_C, X_C, n_C, beta_star.row(c).t());
    //     }else{
    //       indices2 = arma::find(study_id_h2 == cc);
    //       tmp2 += indices2.n_elem;
    //       tmp1 += residual2(y_h(indices2), X_h.rows(indices2), indices2.n_elem, beta_star.row(c).t());
    //     }
    //   }
    //   sigma_star(c) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(tmp2/2 + alpha_sigma_G0, 1/(tmp1/2 + beta_sigma_G0)))));
    // }
    tmp1 = 0; tmp2 = 0;
    for(i=0; i<J-1; i++){
      indices2 = arma::find(study_id_h2 == i);
      tmp2 += indices2.n_elem;
      tmp1 += residual2(y_h(indices2), X_h.rows(indices2), indices2.n_elem, beta_star.row(z(i)).t());
    }      
    tmp2 += n_C;
    tmp1 += residual2(y_C, X_C, n_C, beta_star.row(z(J-1)).t());
    sigma = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(tmp2/2 + alpha_sigma_G0, 1/(tmp1/2 + beta_sigma_G0)))));
    
    // update beta by Gibbs
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      tmpvec.zeros(0);
      tmpmat.zeros(0);
      uindices.zeros(0);
      for(ii=0; ii<indices.n_elem; ii++){
        uindices = arma::join_cols(uindices, arma::find(study_id_h2 == indices(ii)));
      }
      if(arma::any(indices == id(J-1))){
        tmpmat = arma::join_cols(X_h.rows(uindices), X_C);
        tmpvec = arma::join_cols(y_h(uindices), y_C);
      }else{
        tmpmat = X_h.rows(uindices);
        tmpvec = y_h(uindices);
      }
      // beta_post_var = inv(inv(B_G0) + tmpmat.t()*tmpmat/pow(sigma_star(c),2));
      // beta_post_mu = beta_post_var * (inv(B_G0)*beta_G0 + tmpmat.t()*tmpvec/pow(sigma_star(c),2));
      beta_post_var = inv(inv(B_G0) + tmpmat.t()*tmpmat/pow(sigma,2));
      beta_post_mu = beta_post_var * (inv(B_G0)*beta_G0 + tmpmat.t()*tmpvec/pow(sigma,2));
      beta_star.row(c) = arma::mvnrnd(beta_post_mu, beta_post_var).t();
    }

    // update v
    w.zeros(maxcl + 1);
    v1.zeros(ncl + 1);
    for(c=0; c<ncl; c++){
      v1(c) = arma::randg(arma::distr_param(m(clid(c)), 1.0));
    }
    v1(ncl) = arma::randg(arma::distr_param(M_DP, 1.0));
    for(c=0; c<(ncl+1); c++){
      v(c) = v1(c)/sum(v1);
    }
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      w(c) = v(i); // weight for each cluster 
    }
    w_ = v(ncl); // weight for potential new clusters
    

    // update u
    u.zeros(J);
    for(i=0; i<J; i++){
      u(i) = R::runif(0.0, 1.0) * w(z(i));
    }
    u_star = arma::min(u);
    
    // candidates for new clusters
    clid0 = arma::find(m == 0);
    if(clid0.n_elem > 0){
      w_old = w;
      beta_star_old = beta_star;
      // sigma_star_old = sigma_star;
      if(max(clid0) >= w.n_elem){
        len1 = max(clid0) - w.n_elem + 1;
        w_add.zeros(len1);
        beta_star_add.zeros(len1, ncol_x);
        // sigma_star_add.ones(len1);
      }
      countw = 0;
      countth_j = 0;
      ncl_new = 0;
      
      while ( (w_ > u_star) && (ncl_new < clid0.n_elem)  ){
        v_ = R::rbeta(1.0, M_DP);
        if(clid0(ncl_new) < w.n_elem){
          w_old(clid0(ncl_new)) = w_*v_;
        }else{
          w_add(countw) = w_*v_;
          countw++;
        }
        w_ *= (1-v_);
        ncl_new++;
      }
      
      if(max(clid0) >= w_old.n_elem){
        w = arma::join_cols(w_old, w_add);
      }else{
        w = w_old;
      }
      
      if(ncl_new != 0){
        clid00 = clid0(arma::span(0, (ncl_new-1)));
        for(i=0; i<clid00.n_elem; i++){
          j = clid00(i);
          if(j < beta_star_old.n_rows){
            beta_star_old.row(j) = arma::mvnrnd(beta_G0, B_G0).t();
            // sigma_star_old(j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
          }else{
            beta_star_add.row(countth_j) = arma::mvnrnd(beta_G0, B_G0).t();
            // sigma_star_add(countth_j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
            countth_j++;
          }
        }
        if(max(clid00) >= beta_star_old.n_rows){
          beta_star = arma::join_cols(beta_star_old, beta_star_add);
          // sigma_star = arma::join_cols(sigma_star_old, sigma_star_add);
        }else{
          beta_star = beta_star_old;
          // sigma_star = sigma_star_old;
        }
      }
    }
    

    // update z
    candi_cl = arma::find(w != 0);
    prop_z.zeros(id.n_elem);
    z_old = z;
    z.zeros(J);
    if(candi_cl.n_elem > 1){
      for(i=0; i<J-1; i++){
        ll.zeros(candi_cl.n_elem);
        like.zeros(candi_cl.n_elem);
        indices2 = arma::find(study_id_h2 == i);
        tmp2 = indices2.n_elem;
        for(c=0; c<candi_cl.n_elem; c++){
          ll(c) = (u(i) < w(candi_cl(c)))*1;
          // like(c) = normpdf_each(y_h(indices2), X_h.rows(indices2), tmp2, beta_star.row(candi_cl(c)).t(), sigma_star(candi_cl(c)));
          like(c) = normpdf_each(y_h(indices2), X_h.rows(indices2), tmp2, beta_star.row(candi_cl(c)).t(), sigma);
        }
        proposal_p = like % ll;
        proposal_p_ = arma::find(proposal_p != 0);
        candi_clj = candi_cl(proposal_p_);
        proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
        len1 = candi_clj.n_elem;
        if(len1 == 1){
          prop_z(i) = candi_clj(0);
        }else if(len1 == 0){
          prop_z(i) = z_old(i);
        }else{
          prop_z(i) = candi_clj(sample_new_cls_p(proposal_p1));
        }
      }
      i = J-1;
      ll.zeros(candi_cl.n_elem);
      like.zeros(candi_cl.n_elem);
      indices2 = arma::find(study_id_h2 == i);
      tmp2 = indices2.n_elem;
      for(c=0; c<candi_cl.n_elem; c++){
        ll(c) = (u(i) < w(candi_cl(c)))*1;
        // like(c) = normpdf_each(y_C, X_C, n_C, beta_star.row(candi_cl(c)).t(), sigma_star(candi_cl(c)));
        like(c) = normpdf_each(y_C, X_C, n_C, beta_star.row(candi_cl(c)).t(), sigma);
      }
      proposal_p = like % ll;
      proposal_p_ = arma::find(proposal_p != 0);
      candi_clj = candi_cl(proposal_p_);
      proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
      len1 = candi_clj.n_elem;
      if(len1 == 1){
        prop_z(i) = candi_clj(0);
      }else if(len1 == 0){
        prop_z(i) = z_old(i);
      }else{
        prop_z(i) = candi_clj(sample_new_cls_p(proposal_p1));
      }
    }else{
      prop_z.fill(candi_cl(0));
    }
    for(i=0; i<id.n_elem; i++){
      z(i) = prop_z(i);
    }
    
    unique_z = arma::unique(z);
    ncl = unique_z.n_elem;
    maxcl = max(arma::unique(z));
    m = tableC2(z, J);
    clid = arma::find(m != 0);
    

    // save output
    diff = nit - NBURN + 1;
    if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
      beta_C_out.row(count) = beta_star.row(z(id(J-1)));
      // sigma_C_out(count) = sigma_star(z(id(J-1)));
      sigma_out(count) = sigma;
      
      z_out.row(count) = z.t() + 1;
      indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
      M_DP_out(count) = M_DP;
      ncl_out(count) = ncl;
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
  
  // Rcpp::_["sigma_C"] = sigma_C_out, 
    Rcpp::List ret = Rcpp::List::create(Rcpp::_["beta_C"] = beta_C_out, 
                                        Rcpp::_["sigma_C"] = sigma_out,
                                        Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["ncl"] = ncl_out
  );
  return(ret);
}


// MCMC sampling for DPM method
// [[Rcpp::export]]
Rcpp::List DPM_Linear_Ca(
    const arma::vec y_C,
    const arma::mat x_C,
    const arma::vec g_C,
    const unsigned int n_C,
    const arma::vec y_h,
    const arma::mat x_h,
    const arma::vec study_id_h,
    const arma::vec n_h,
    const double hcauchy_scale = 1,
    const double mu_beta_G0 = 0,
    const double tau_beta_G0 = 10000,
    const double alpha_sigma_G0 = 1, 
    const double beta_sigma_G0 = 1,
    const double proposal_M_DP_sd = 1,
    const unsigned int NBURN = 4000,
    const unsigned int NTHIN = 10,
    const unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  arma::mat X_C = arma::join_rows(x_C, g_C);
  arma::vec g_h(arma::accu(n_h), arma::fill::zeros);
  arma::mat X_h = arma::join_rows(x_h, g_h);
  unsigned int ncol_x = X_h.n_cols;
  unsigned int J = study_id_h.max()+1;
  arma::uvec id(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    id(i) = i;
  }
  arma::vec study_id_h2 = study_id_h - 1;
  
  // hyperparameter
  // base distribution
  arma::vec beta_G0 = arma::vec(ncol_x, arma::fill::value(mu_beta_G0));
  arma::vec B_G0_vec = arma::vec(ncol_x, arma::fill::value(tau_beta_G0));
  arma::mat B_G0 = tau_beta_G0*arma::mat(ncol_x, ncol_x, arma::fill::eye);
  // DP precision parameters
  double M_DP = 1.0;

  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  // initialization for each cluster parameter
  arma::mat beta_star(maxcl+1, ncol_x, arma::fill::zeros);
  arma::vec sigma_star(maxcl+1, arma::fill::ones);
  arma::mat beta_star_old(maxcl+1, ncol_x, arma::fill::zeros);
  arma::vec sigma_star_old(maxcl+1, arma::fill::ones);
  arma::mat beta_star_add(maxcl+1, ncol_x, arma::fill::zeros);
  arma::vec sigma_star_add(maxcl+1, arma::fill::ones);
  
  arma::vec v(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    v(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec w;
  w.zeros(maxcl+1);
  w(0) = v(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      w(c) = v(c);
      for(c_=0; c_<c; c_++){
        w(c) *= (1-v(c_));
      }
    }
  }
  double w_ = v(ncl); // weight for potential new clusters
  
  // MCMC output parameters
  arma::mat beta_C_out(NOUTSAMPLE, ncol_x);
  arma::vec sigma_C_out(NOUTSAMPLE);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;

  // variables for MCMC
  unsigned int nit, i, ii, cc, j, ncl_new, diff, countw, countth_j;
  arma::vec sumy, sumn, v1, u, ll, like, proposal_p, proposal_p1, beta_aux, 
  w_old, w_add, tmpvec, beta_post_mu;
  arma::uvec indices, indices2, candi_cl, candi_clj, prop_z, 
  proposal_p_, clid0, clid00, z_old, uindices;
  double u_star, v_, tmp1, tmp2, M_DP_prop, lr;
  arma::uword len1 = 0;
  arma::mat beta_post_var, tmpmat;
  
  // Rcpp::Rcout << "start !!"  << "\n";
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    M_DP_prop = (M_DP + arma::randu() *2.0 - 1.0) * proposal_M_DP_sd;
    if(M_DP_prop < 0){
      M_DP_prop = -M_DP_prop;
    }
    lr = ncl * log(M_DP_prop) + lgamma(M_DP_prop) + log(den_half_cauchy(M_DP_prop, hcauchy_scale))
      - lgamma(M_DP_prop + J)
      - ncl * log(M_DP) - lgamma(M_DP) - log(den_half_cauchy(M_DP, hcauchy_scale))
      + lgamma(M_DP + J);
      if(log(R::runif(0.0, 1.0))<lr){
        M_DP = M_DP_prop;
      }
      
    
    // update each cluster parameter
    // update sigma by Gibbs
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      tmp1 = 0; tmp2 = 0;
      for(ii=0; ii<indices.n_elem; ii++){
        cc = indices(ii);
        if(cc == id(J-1)){
          tmp2 += n_C;
          tmp1 += residual2(y_C, X_C, n_C, beta_star.row(c).t());
        }else{
          indices2 = arma::find(study_id_h2 == cc);
          tmp2 += indices2.n_elem;
          tmp1 += residual2(y_h(indices2), X_h.rows(indices2), indices2.n_elem, beta_star.row(c).t());
        }
      }
      sigma_star(c) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(tmp2/2 + alpha_sigma_G0, 1/(tmp1/2 + beta_sigma_G0)))));
    }
    
    // update beta by Gibbs
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      tmpvec.zeros(0);
      tmpmat.zeros(0);
      uindices.zeros(0);
      for(ii=0; ii<indices.n_elem; ii++){
        uindices = arma::join_cols(uindices, arma::find(study_id_h2 == indices(ii)));
      }
      if(arma::any(indices == id(J-1))){
        tmpmat = arma::join_cols(X_h.rows(uindices), X_C);
        tmpvec = arma::join_cols(y_h(uindices), y_C);
      }else{
        tmpmat = X_h.rows(uindices);
        tmpvec = y_h(uindices);
      }
      beta_post_var = inv(inv(B_G0) + tmpmat.t()*tmpmat/pow(sigma_star(c),2));
      beta_post_mu = beta_post_var * (inv(B_G0)*beta_G0 + tmpmat.t()*tmpvec/pow(sigma_star(c),2));
      beta_star.row(c) = arma::mvnrnd(beta_post_mu, beta_post_var).t();
    }
    
    // update v
    w.zeros(maxcl + 1);
    v1.zeros(ncl + 1);
    for(c=0; c<ncl; c++){
      v1(c) = arma::randg(arma::distr_param(m(clid(c)), 1.0));
    }
    v1(ncl) = arma::randg(arma::distr_param(M_DP, 1.0));
    for(c=0; c<(ncl+1); c++){
      v(c) = v1(c)/sum(v1);
    }
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      w(c) = v(i); // weight for each cluster 
    }
    w_ = v(ncl); // weight for potential new clusters
    
    
    // update u
    u.zeros(J);
    for(i=0; i<J; i++){
      u(i) = R::runif(0.0, 1.0) * w(z(i));
    }
    u_star = arma::min(u);
    
    
    // candidates for new clusters
    clid0 = arma::find(m == 0);
    if(clid0.n_elem > 0){
      w_old = w;
      beta_star_old = beta_star;
      sigma_star_old = sigma_star;
      if(max(clid0) >= w.n_elem){
        len1 = max(clid0) - w.n_elem + 1;
        w_add.zeros(len1);
        beta_star_add.zeros(len1, ncol_x);
        sigma_star_add.ones(len1);
      }
      countw = 0;
      countth_j = 0;
      ncl_new = 0;
      
      while ( (w_ > u_star) && (ncl_new < clid0.n_elem)  ){
        v_ = R::rbeta(1.0, M_DP);
        if(clid0(ncl_new) < w.n_elem){
          w_old(clid0(ncl_new)) = w_*v_;
        }else{
          w_add(countw) = w_*v_;
          countw++;
        }
        w_ *= (1-v_);
        ncl_new++;
      }
      
      if(max(clid0) >= w_old.n_elem){
        w = arma::join_cols(w_old, w_add);
      }else{
        w = w_old;
      }
      
      if(ncl_new != 0){
        clid00 = clid0(arma::span(0, (ncl_new-1)));
        for(i=0; i<clid00.n_elem; i++){
          j = clid00(i);
          if(j < beta_star_old.n_rows){
            beta_star_old.row(j) = arma::mvnrnd(beta_G0, B_G0).t();
            sigma_star_old(j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
          }else{
            beta_star_add.row(countth_j) = arma::mvnrnd(beta_G0, B_G0).t();
            sigma_star_add(countth_j) = as_scalar(arma::sqrt(1/arma::randg(1, arma::distr_param(alpha_sigma_G0, 1/beta_sigma_G0))));
            countth_j++;
          }
        }
        if(max(clid00) >= beta_star_old.n_rows){
          beta_star = arma::join_cols(beta_star_old, beta_star_add);
          sigma_star = arma::join_cols(sigma_star_old, sigma_star_add);
        }else{
          beta_star = beta_star_old;
          sigma_star = sigma_star_old;
        }
      }
    }
    
    
    // update z
    candi_cl = arma::find(w != 0);
    prop_z.zeros(id.n_elem);
    z_old = z;
    z.zeros(J);
    if(candi_cl.n_elem > 1){
      for(i=0; i<J-1; i++){
        ll.zeros(candi_cl.n_elem);
        like.zeros(candi_cl.n_elem);
        indices2 = arma::find(study_id_h2 == i);
        tmp2 = indices2.n_elem;
        for(c=0; c<candi_cl.n_elem; c++){
          ll(c) = (u(i) < w(candi_cl(c)))*1;
          like(c) = normpdf_each(y_h(indices2), X_h.rows(indices2), tmp2, beta_star.row(candi_cl(c)).t(), sigma_star(candi_cl(c)));
        }
        proposal_p = like % ll;
        proposal_p_ = arma::find(proposal_p != 0);
        candi_clj = candi_cl(proposal_p_);
        proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
        len1 = candi_clj.n_elem;
        if(len1 == 1){
          prop_z(i) = candi_clj(0);
        }else if(len1 == 0){
          prop_z(i) = z_old(i);
        }else{
          prop_z(i) = candi_clj(sample_new_cls_p(proposal_p1));
        }
      }
      i = J-1;
      ll.zeros(candi_cl.n_elem);
      like.zeros(candi_cl.n_elem);
      indices2 = arma::find(study_id_h2 == i);
      tmp2 = indices2.n_elem;
      for(c=0; c<candi_cl.n_elem; c++){
        ll(c) = (u(i) < w(candi_cl(c)))*1;
        like(c) = normpdf_each(y_C, X_C, n_C, beta_star.row(candi_cl(c)).t(), sigma_star(candi_cl(c)));
      }
      proposal_p = like % ll;
      proposal_p_ = arma::find(proposal_p != 0);
      candi_clj = candi_cl(proposal_p_);
      proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
      len1 = candi_clj.n_elem;
      if(len1 == 1){
        prop_z(i) = candi_clj(0);
      }else if(len1 == 0){
        prop_z(i) = z_old(i);
      }else{
        prop_z(i) = candi_clj(sample_new_cls_p(proposal_p1));
      }
    }else{
      prop_z.fill(candi_cl(0));
    }
    for(i=0; i<id.n_elem; i++){
      z(i) = prop_z(i);
    }
    
    unique_z = arma::unique(z);
    ncl = unique_z.n_elem;
    maxcl = max(arma::unique(z));
    m = tableC2(z, J);
    clid = arma::find(m != 0);
    
    // save output
    diff = nit - NBURN + 1;
    if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
      beta_C_out.row(count) = beta_star.row(z(id(J-1)));
      sigma_C_out(count) = sigma_star(z(id(J-1)));
      
      z_out.row(count) = z.t() + 1;
      indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
      M_DP_out(count) = M_DP;
      ncl_out(count) = ncl;
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
  
  Rcpp::List ret = Rcpp::List::create(Rcpp::_["beta_C"] = beta_C_out, 
                                      Rcpp::_["sigma_C"] = sigma_C_out, 
                                      Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["ncl"] = ncl_out
  );
  return(ret);
}



// MCMC sampling for DDPM method
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



// MCMC sampling for DDPM method
// [[Rcpp::export]]
Rcpp::List DDPM_Bin(
    arma::vec y,
    arma::vec n,
    double hyper_gamma_shape = 1, 
    double hyper_gamma_scale = 1,
    double proposal_phi_sd = 0.1,
    unsigned int NBURN = 4000,
    unsigned int NTHIN = 10,
    unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  unsigned int J = n.n_elem;
  arma::uvec idHC(J-1, arma::fill::zeros);
  for(unsigned int i=0; i<(J-1); i++){
    idHC(i) = i;
  }
  unsigned int idCC = J-1;
  
  // hyperparameter
  // base distribution
  double a_G0 = 0.5;
  double b_G0 = 0.5;
  // DP precision parameters
  double M_DP = 1.0;
  double pi = 0.5;
  int s = 0;
  
  double phi = 0.5;
  double phip = 0.55;
  // hyperparameters for phi_gamma ~ Beta(phi_gamma1, phi_gamma2)
  double phi_gamma1 = 2.0;
  double phi_gamma2 = 2.0;
  
  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  arma::vec th_j(maxcl+1, arma::fill::zeros);
  for(unsigned int i=0; i<th_j.n_elem; i++){
    th_j(i) = R::rbeta(a_G0, b_G0);
  }
  arma::vec vHC(maxcl*3);
  arma::vec vHCp(maxcl*3);
  arma::vec vCC(maxcl*3);
  arma::vec vCCp(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    vHC(i) = R::rbeta(1.0, M_DP);
    vHCp(i) = R::rbeta(1.0, M_DP);
    vCC(i) = R::rbeta(1.0, M_DP);
    vCCp(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec wHC, wHCp, wCC, wCCp;
  wHC.zeros(maxcl+1);
  wHCp.zeros(maxcl+1);
  wCC.zeros(maxcl+1);
  wCCp.zeros(maxcl+1);
  wHC(0) = vHC(0);
  wHCp(0) = vHCp(0);
  wCC(0) = vCC(0);
  wCCp(0) = vCCp(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      wHC(c) = vHC(c);
      wHCp(c) = vHCp(c);
      wCC(c) = vCC(c);
      wCCp(c) = vCCp(c);
      for(c_=0; c_<c; c_++){
        wHC(c) *= (1-vHC(c_));
        wHCp(c) *= (1-vHCp(c_));
        wCC(c) *= (1-vCC(c_));
        wCCp(c) *= (1-vCCp(c_));
      }
    }
  }
  
  
  // MCMC output parameters
  arma::mat p_out(NOUTSAMPLE, J);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec phi_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;
  
  // variables for MCMC
  unsigned int nit, i, j, I1, I2, zCC, ncl_new, prop_zCC, diff, countwHC, countwCC, countth_j;
  arma::vec sumy, sumn, uHC, uCC, ll, like, proposal_p, proposal_p1, p_aux, 
  wHC_old, wCC_old, wHC_add, wCC_add, th_j_old, th_j_add;
  arma::uvec zHC, indices, indices2, candi_clHC, candi_clCC, prop_zHC, 
  proposal_p_, candi_clHCj, candi_clCCj, clid0, clid00, z_old;
  double betaprior, aj, bj, lq01, lq01p, lq11, lq11p, q01, q11, q01p, q11p, p01, p01p, uu,
  lq0n, lq0np, lq1n, lq1np, q0n, q1n, p0n, q0np, q1np, p0np, sumtemp, sumtempp, 
  Jtp, Jt, lnr, prophi, uHC_star, uCC_star, wHC_, wCC_, vHC_, vCC_;
  arma::uword len1;
  
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    pi = R::rbeta(M_DP+1, J);
    s = R::rbinom(1, 1-(hyper_gamma_shape+ncl-1)/(J*(hyper_gamma_scale-log(pi))+hyper_gamma_shape+ncl-1));
    M_DP = arma::randg(arma::distr_param(hyper_gamma_shape+ncl-s, 1.0/((1.0/hyper_gamma_scale)-log(pi))));
    
    
    // update theta
    sumy.zeros(maxcl+1);
    sumn.zeros(maxcl+1);
    th_j.zeros(maxcl+1);
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      sumy(c) = sum(y(indices));
      sumn(c) = sum(n(indices));
    }
    th_j = ran_beta_post(sumy, sumn, a_G0, b_G0);
    
    
    // update v for historical controls
    phip = ran_truncnorm(phi, proposal_phi_sd, 0, 1);
    for(c=0; c<(maxcl+1); c++){
      if(m(c) == 0){
        betaprior = R::rbeta(1.0, M_DP);
        vHC(c) = betaprior;
        vHCp(c) = betaprior;
      }else{
        zHC = z.subvec(0,max(idHC));
        indices = arma::find((zHC) == c);
        indices2 = arma::find((zHC) > c);
        I1 = indices.n_elem;
        I2 = indices2.n_elem;
        aj = 1 + I1;
        bj = M_DP + I2;
        lq01 = log(phi) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCC(c)) - lgamma(aj+bj);
        lq01p = log(phip) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCCp(c)) - lgamma(aj+bj);
        lq11 = log(1-phi) + log(M_DP) + I1*log(vCC(c)) + (M_DP-1+I2)*log(1-vCC(c));
        lq11p = log(1-phip) + log(M_DP) + I1*log(vCCp(c)) + (M_DP-1+I2)*log(1-vCCp(c));
        q01 = exp(lq01);
        q11 = exp(lq11);
        q01p = exp(lq01p);
        q11p = exp(lq11p);
        p01 = q01/(q01+q11);
        p01p = q01p/(q01p+q11p);
        uu = R::runif(0.0, 1.0);
        if(p01>uu){
          vHC(c) = R::rbeta(aj, bj);
        }else{
          vHC(c) = vCC(c);
        }
        if(p01p>uu){
          vHCp(c) = R::rbeta(aj, bj);
        }else{
          vHCp(c) = vCCp(c);
        }
      }
    }
    for(c=0; c<(maxcl+1); c++){
      if(vHC(c) >= 0.9999){
        vHC(c) = 0.9999;
      }
      if(vHCp(c) >= 0.9999){
        vHCp(c) = 0.9999;
      }
    }
    
    // update v for current control
    for(c=0; c<(maxcl+1); c++){
      if(m(c) == 0){
        betaprior = R::rbeta(1.0, M_DP);
        vCC(c) = betaprior;
        vCCp(c) = betaprior;
      }else{
        zCC = z(idCC);
        I1 = ((zCC)==c)*1;
        I2 = ((zCC)>c)*1;
        aj = 1 + I1;
        bj = M_DP + I2;
        lq0n = log(phi) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
        lq0np = log(phip) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
        lq1n = log(1-phi) + I1*log(vHC(c)) + I2*log(1-vHC(c));
        lq1np = log(1-phip) + I1*log(vHCp(c)) + I2*log(1-vHCp(c));
        q0n = exp(lq0n);
        q1n = exp(lq1n);
        p0n = q0n/(q0n+q1n);
        q0np = exp(lq0np);
        q1np = exp(lq1np);
        p0np = q0np/(q0np+q1np);
        uu = R::runif(0.0, 1.0);
        if(p0n>uu){
          vCC(c) = R::rbeta(aj, bj);
        }else{
          vCC(c) = vHC(c);
        }
        if(p0np>uu){
          vCCp(c) = R::rbeta(aj, bj);
        }else{
          vCCp(c) = vHCp(c);
        }
      }
    }
    for(c=0; c<(maxcl+1); c++){
      if(vCC(c) >= 0.9999){
        vCC(c) = 0.9999;
      }
      if(vCCp(c) >= 0.9999){
        vCCp(c) = 0.9999;
      }
    }
    
    // update weight
    wHC.zeros(maxcl+1);
    wHCp.zeros(maxcl+1);
    wCC.zeros(maxcl+1);
    wCCp.zeros(maxcl+1);
    wHC(0) = vHC(0);
    wHCp(0) = vHCp(0);
    wCC(0) = vCC(0);
    wCCp(0) = vCCp(0);
    if(maxcl != 0){
      for(c=1; c<(maxcl+1);c++){
        wHC(c) = vHC(c);
        wHCp(c) = vHCp(c);
        wCC(c) = vCC(c);
        wCCp(c) = vCCp(c);
        for(c_=0; c_<c; c_++){
          wHC(c) *= (1-vHC(c_));
          wHCp(c) *= (1-vHCp(c_));
          wCC(c) *= (1-vCC(c_));
          wCCp(c) *= (1-vCCp(c_));
        }
      }
    }
    
    // update phi
    sumtemp = 0;
    sumtempp = 0;
    for(i=0; i<idHC.n_elem; i++){
      sumtemp += log_sum_w_dbinom(y(i), n(i), th_j, wHC);
      sumtempp += log_sum_w_dbinom(y(i), n(i), th_j, wHCp);
    }
    sumtemp += log_sum_w_dbinom(y(idCC), n(idCC), th_j, wCC);
    sumtempp += log_sum_w_dbinom(y(idCC), n(idCC), th_j, wCCp);
    Jtp = den_truncnorm(phip, phi, proposal_phi_sd, 0, 1);
    Jt = den_truncnorm(phi, phip, proposal_phi_sd, 0, 1);
    lnr = ((phi_gamma1-1)*log(phip) + (phi_gamma2-1)*log(1-phip) + sumtempp) - log(Jtp)
      - ((phi_gamma1-1)*log(phi) + (phi_gamma2-1)*log(1-phi) + sumtemp) + log(Jt);
    prophi = exp(std::min(lnr, 0.0));
    uu = R::runif(0.0, 1.0);
    if(prophi >= uu){
      phi = phip;
      wHC = wHCp;
      wCC = wCCp;
    }else{
      phi = phi;
      wHC = wHC;
      wCC = wCC;
    }
    
    // update u
    uHC.zeros(idHC.n_elem);
    uCC.zeros(1);
    for(i=0; i<idHC.n_elem; i++){
      uHC(i) = R::runif(0.0, 1.0) * wHC(z(i));
    }
    uCC(0) = R::runif(0.0, 1.0) * wCC(z(idCC));
    uHC_star = arma::min(uHC);
    uCC_star = uCC(0);
    
    
    // candidates for new clusters
    wHC_ = 1 - sum(wHC);
    wCC_ = 1 - sum(wCC);
    clid0 = arma::find(m == 0);
    if(clid0.n_elem > 0){
      wHC_old = wHC;
      wCC_old = wCC;
      th_j_old = th_j;
      if(max(clid0) >= wHC.n_elem){
        len1 = max(clid0) - wHC.n_elem + 1;
        wHC_add.zeros(len1);
        wCC_add.zeros(len1);
        th_j_add.zeros(len1);
      }
      countwHC = 0;
      countwCC = 0;
      countth_j = 0;
      ncl_new = 0;
      
      while ( ((wHC_ > uHC_star) || (wCC_ > uCC_star)) && (ncl_new < clid0.n_elem)  ){
        vHC_ = R::rbeta(1.0, M_DP);
        if(clid0(ncl_new) < wHC.n_elem){
          wHC_old(clid0(ncl_new)) = wHC_*vHC_;
        }else{
          wHC_add(countwHC) = wHC_*vHC_;
          countwHC++;
        }
        wHC_ *= (1-vHC_);
        
        vCC_ = R::rbeta(1.0, M_DP);
        if(clid0(ncl_new) < wCC.n_elem){
          wCC_old(clid0(ncl_new)) = wCC_*vCC_;
        }else{
          wCC_add(countwCC) = wCC_*vCC_;
          countwCC++;
        }
        wCC_ = wCC_*(1-vCC_);
        
        ncl_new++;
      }
      
      if(max(clid0) >= wHC_old.n_elem){
        wHC = arma::join_cols(wHC_old, wHC_add);
        wCC = arma::join_cols(wCC_old, wCC_add);
      }else{
        wHC = wHC_old;
        wCC = wCC_old;
      }
      
      if(ncl_new != 0){
        clid00 = clid0(arma::span(0, (ncl_new-1)));
        for(i=0; i<clid00.n_elem; i++){
          j = clid00(i);
          if(j < th_j_old.n_elem){
            th_j_old(j) = R::rbeta(a_G0, b_G0);
          }else{
            th_j_add(countth_j) = R::rbeta(a_G0, b_G0);
            countth_j++;
          }
        }
        if(max(clid00) >= th_j_old.n_elem){
          th_j = arma::join_cols(th_j_old, th_j_add);
        }else{
          th_j = th_j_old;
        }
      }
    }
    
    for(c=0; c<th_j.n_elem; c++){
      if(th_j(c) == 0.0){
        th_j(c) = 0.0001;
      }
      if(th_j(c) == 1.0){
        th_j(c) = 0.9999;
      }
    }
    
    // update z
    candi_clHC = arma::find(wHC != 0);
    candi_clCC = arma::find(wCC != 0);
    prop_zHC.zeros(idHC.n_elem);
    prop_zCC = 0;
    z_old = z;
    z.zeros(J);
    if(candi_clHC.n_elem > 1){
      for(i=0; i<idHC.n_elem; i++){
        ll.zeros(candi_clHC.n_elem);
        like.zeros(candi_clHC.n_elem);
        for(c=0; c<candi_clHC.n_elem; c++){
          ll(c) = (uHC(i) < wHC(candi_clHC(c)))*1;
          like(c) = R::dbinom(y(i), n(i), th_j(candi_clHC(c)), false);
        }
        proposal_p = like % ll;
        proposal_p_ = arma::find(proposal_p != 0);
        candi_clHCj = candi_clHC(proposal_p_);
        proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
        len1 = candi_clHCj.n_elem;
        if(len1 == 1){
          prop_zHC(i) = candi_clHCj(0);
        }else if(len1 == 0){
          prop_zHC(i) = z_old(i);
        }else{
          prop_zHC(i) = candi_clHCj(sample_new_cls_p(proposal_p1));
        }
      }
    }else{
      prop_zHC.fill(candi_clHC(0));
    }
    for(i=0; i<idHC.n_elem; i++){
      z(i) = prop_zHC(i);
    }
    
    if(candi_clCC.n_elem > 1){
      ll.zeros(candi_clCC.n_elem);
      like.zeros(candi_clCC.n_elem);
      for(c=0; c<candi_clCC.n_elem; c++){
        ll(c) = (uCC(0) < wCC(candi_clCC(c)))*1;
        like(c) = R::dbinom(y(idCC), n(idCC), th_j(candi_clCC(c)), false);
      }
      proposal_p = like % ll;
      proposal_p_ = arma::find(proposal_p != 0);
      candi_clCCj = candi_clCC(proposal_p_);
      proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
      len1 = candi_clCCj.n_elem;
      if(len1 == 1){
        prop_zCC = candi_clCCj(0);
      }else if(len1 == 0){
        prop_zCC = z_old(idCC);
      }else{
        prop_zCC = candi_clCCj(sample_new_cls_p(proposal_p1));
      }
    }else{
      prop_zCC = candi_clCC(0);
    }
    z(idCC) = prop_zCC;
    
    unique_z = arma::unique(z);
    ncl = unique_z.n_elem;
    maxcl = max(arma::unique(z));
    m = tableC2(z, J);
    clid = arma::find(m != 0);
    
    
    // save output
    diff = nit - NBURN + 1;
    if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
      p_aux.zeros(J);
      for(i=0; i<J; i++){
        p_aux(i) = th_j(z(i));
      }
      p_out.row(count) = p_aux.t();
      z_out.row(count) = z.t() + 1;
      indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
      M_DP_out(count) = M_DP;
      phi_out(count) = phi;
      ncl_out(count) = ncl;
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
  
  Rcpp::List ret = Rcpp::List::create(Rcpp::_["p"] = p_out, 
                                      Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["phi"] = phi_out,
                                      Rcpp::_["ncl"] = ncl_out
  );
  return(ret);
}





// MCMC sampling for DDPM method with M_DP ~ C+
// [[Rcpp::export]]
Rcpp::List DDPM_Bin_Ca(
    arma::vec y,
    arma::vec n,
    double hcauchy_scale = 1,
    double proposal_M_DP_sd = 1,
    double proposal_phi_sd = 0.1,
    unsigned int NBURN = 4000,
    unsigned int NTHIN = 10,
    unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  unsigned int J = n.n_elem;
  arma::uvec idHC(J-1, arma::fill::zeros);
  for(unsigned int i=0; i<(J-1); i++){
    idHC(i) = i;
  }
  unsigned int idCC = J-1;
  
  // hyperparameter
  // base distribution
  double a_G0 = 0.5;
  double b_G0 = 0.5;
  // DP precision parameters
  double M_DP = 1.0;
  
  double phi = 0.5;
  double phip = 0.55;
  // hyperparameters for phi_gamma ~ Beta(phi_gamma1, phi_gamma2)
  double phi_gamma1 = 2.0;
  double phi_gamma2 = 2.0;
  
  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  arma::vec th_j(maxcl+1, arma::fill::zeros);
  for(unsigned int i=0; i<th_j.n_elem; i++){
    th_j(i) = R::rbeta(a_G0, b_G0);
  }
  arma::vec vHC(maxcl*3);
  arma::vec vHCp(maxcl*3);
  arma::vec vCC(maxcl*3);
  arma::vec vCCp(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    vHC(i) = R::rbeta(1.0, M_DP);
    vHCp(i) = R::rbeta(1.0, M_DP);
    vCC(i) = R::rbeta(1.0, M_DP);
    vCCp(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec wHC, wHCp, wCC, wCCp;
  wHC.zeros(maxcl+1);
  wHCp.zeros(maxcl+1);
  wCC.zeros(maxcl+1);
  wCCp.zeros(maxcl+1);
  wHC(0) = vHC(0);
  wHCp(0) = vHCp(0);
  wCC(0) = vCC(0);
  wCCp(0) = vCCp(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      wHC(c) = vHC(c);
      wHCp(c) = vHCp(c);
      wCC(c) = vCC(c);
      wCCp(c) = vCCp(c);
      for(c_=0; c_<c; c_++){
        wHC(c) *= (1-vHC(c_));
        wHCp(c) *= (1-vHCp(c_));
        wCC(c) *= (1-vCC(c_));
        wCCp(c) *= (1-vCCp(c_));
      }
    }
  }
  
  
  // MCMC output parameters
  arma::mat p_out(NOUTSAMPLE, J);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec phi_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;

  // variables for MCMC
  unsigned int nit, i, j, I1, I2, zCC, ncl_new, prop_zCC, diff, countwHC, countwCC, countth_j;
  arma::vec sumy, sumn, uHC, uCC, ll, like, proposal_p, proposal_p1, p_aux, 
  wHC_old, wCC_old, wHC_add, wCC_add, th_j_old, th_j_add;
  arma::uvec zHC, indices, indices2, candi_clHC, candi_clCC, prop_zHC, 
  proposal_p_, candi_clHCj, candi_clCCj, clid0, clid00, z_old;
  double betaprior, aj, bj, lq01, lq01p, lq11, lq11p, q01, q11, q01p, q11p, p01, p01p, uu,
  lq0n, lq0np, lq1n, lq1np, q0n, q1n, p0n, q0np, q1np, p0np, sumtemp, sumtempp, 
  Jtp, Jt, lnr, prophi, uHC_star, uCC_star, wHC_, wCC_, vHC_, vCC_, M_DP_prop, lr;
  arma::uword len1;
  
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    M_DP_prop = (M_DP + arma::randu() *2.0 - 1.0) * proposal_M_DP_sd;
    if(M_DP_prop < 0){
      M_DP_prop = -M_DP_prop;
    }
    lr = ncl * log(M_DP_prop) + lgamma(M_DP_prop) + log(den_half_cauchy(M_DP_prop, hcauchy_scale))
      - lgamma(M_DP_prop + J)
      - ncl * log(M_DP) - lgamma(M_DP) - log(den_half_cauchy(M_DP, hcauchy_scale))
      + lgamma(M_DP + J);
      if(log(R::runif(0.0, 1.0))<lr){
        M_DP = M_DP_prop;
      }
      
      
      // update theta
      sumy.zeros(maxcl+1);
      sumn.zeros(maxcl+1);
      th_j.zeros(maxcl+1);
      for(i=0; i<clid.n_elem; i++){
        c = clid(i);
        indices = arma::find(z == c) ;
        sumy(c) = sum(y(indices));
        sumn(c) = sum(n(indices));
      }
      th_j = ran_beta_post(sumy, sumn, a_G0, b_G0);
      
      // update v for historical controls
      phip = ran_truncnorm(phi, proposal_phi_sd, 0, 1);
      for(c=0; c<(maxcl+1); c++){
        if(m(c) == 0){
          betaprior = R::rbeta(1.0, M_DP);
          vHC(c) = betaprior;
          vHCp(c) = betaprior;
        }else{
          zHC = z.subvec(0,max(idHC));
          indices = arma::find((zHC) == c);
          indices2 = arma::find((zHC) > c);
          I1 = indices.n_elem;
          I2 = indices2.n_elem;
          aj = 1 + I1;
          bj = M_DP + I2;
          lq01 = log(phi) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCC(c)) - lgamma(aj+bj);
          lq01p = log(phip) + 2*log(M_DP) + lgamma(aj) + lgamma(bj) + (M_DP-1)*log(1-vCCp(c)) - lgamma(aj+bj);
          lq11 = log(1-phi) + log(M_DP) + I1*log(vCC(c)) + (M_DP-1+I2)*log(1-vCC(c));
          lq11p = log(1-phip) + log(M_DP) + I1*log(vCCp(c)) + (M_DP-1+I2)*log(1-vCCp(c));
          q01 = exp(lq01);
          q11 = exp(lq11);
          q01p = exp(lq01p);
          q11p = exp(lq11p);
          p01 = q01/(q01+q11);
          p01p = q01p/(q01p+q11p);
          uu = R::runif(0.0, 1.0);
          if(p01>uu){
            vHC(c) = R::rbeta(aj, bj);
          }else{
            vHC(c) = vCC(c);
          }
          if(p01p>uu){
            vHCp(c) = R::rbeta(aj, bj);
          }else{
            vHCp(c) = vCCp(c);
          }
        }
      }
      for(c=0; c<(maxcl+1); c++){
        if(vHC(c) >= 0.9999){
          vHC(c) = 0.9999;
        }
        if(vHCp(c) >= 0.9999){
          vHCp(c) = 0.9999;
        }
      }
      
      // update v for current control
      for(c=0; c<(maxcl+1); c++){
        if(m(c) == 0){
          betaprior = R::rbeta(1.0, M_DP);
          vCC(c) = betaprior;
          vCCp(c) = betaprior;
        }else{
          zCC = z(idCC);
          I1 = ((zCC)==c)*1;
          I2 = ((zCC)>c)*1;
          aj = 1 + I1;
          bj = M_DP + I2;
          lq0n = log(phi) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
          lq0np = log(phip) + log(M_DP) + lgamma(aj) + lgamma(bj) - lgamma(aj+bj);
          lq1n = log(1-phi) + I1*log(vHC(c)) + I2*log(1-vHC(c));
          lq1np = log(1-phip) + I1*log(vHCp(c)) + I2*log(1-vHCp(c));
          q0n = exp(lq0n);
          q1n = exp(lq1n);
          p0n = q0n/(q0n+q1n);
          q0np = exp(lq0np);
          q1np = exp(lq1np);
          p0np = q0np/(q0np+q1np);
          uu = R::runif(0.0, 1.0);
          if(p0n>uu){
            vCC(c) = R::rbeta(aj, bj);
          }else{
            vCC(c) = vHC(c);
          }
          if(p0np>uu){
            vCCp(c) = R::rbeta(aj, bj);
          }else{
            vCCp(c) = vHCp(c);
          }
        }
      }
      for(c=0; c<(maxcl+1); c++){
        if(vCC(c) >= 0.9999){
          vCC(c) = 0.9999;
        }
        if(vCCp(c) >= 0.9999){
          vCCp(c) = 0.9999;
        }
      }
      
      // update weight
      wHC.zeros(maxcl+1);
      wHCp.zeros(maxcl+1);
      wCC.zeros(maxcl+1);
      wCCp.zeros(maxcl+1);
      wHC(0) = vHC(0);
      wHCp(0) = vHCp(0);
      wCC(0) = vCC(0);
      wCCp(0) = vCCp(0);
      if(maxcl != 0){
        for(c=1; c<(maxcl+1);c++){
          wHC(c) = vHC(c);
          wHCp(c) = vHCp(c);
          wCC(c) = vCC(c);
          wCCp(c) = vCCp(c);
          for(c_=0; c_<c; c_++){
            wHC(c) *= (1-vHC(c_));
            wHCp(c) *= (1-vHCp(c_));
            wCC(c) *= (1-vCC(c_));
            wCCp(c) *= (1-vCCp(c_));
          }
        }
      }
      
      // update phi
      sumtemp = 0;
      sumtempp = 0;
      for(i=0; i<idHC.n_elem; i++){
        sumtemp += log_sum_w_dbinom(y(i), n(i), th_j, wHC);
        sumtempp += log_sum_w_dbinom(y(i), n(i), th_j, wHCp);
      }
      sumtemp += log_sum_w_dbinom(y(idCC), n(idCC), th_j, wCC);
      sumtempp += log_sum_w_dbinom(y(idCC), n(idCC), th_j, wCCp);
      Jtp = den_truncnorm(phip, phi, proposal_phi_sd, 0, 1);
      Jt = den_truncnorm(phi, phip, proposal_phi_sd, 0, 1);
      lnr = ((phi_gamma1-1)*log(phip) + (phi_gamma2-1)*log(1-phip) + sumtempp) - log(Jtp)
        - ((phi_gamma1-1)*log(phi) + (phi_gamma2-1)*log(1-phi) + sumtemp) + log(Jt);
      prophi = exp(std::min(lnr, 0.0));
      uu = R::runif(0.0, 1.0);
      if(prophi >= uu){
        phi = phip;
        wHC = wHCp;
        wCC = wCCp;
      }else{
        phi = phi;
        wHC = wHC;
        wCC = wCC;
      }
      
      // update u
      uHC.zeros(idHC.n_elem);
      uCC.zeros(1);
      for(i=0; i<idHC.n_elem; i++){
        uHC(i) = R::runif(0.0, 1.0) * wHC(z(i));
      }
      uCC(0) = R::runif(0.0, 1.0) * wCC(z(idCC));
      uHC_star = arma::min(uHC);
      uCC_star = uCC(0);
      
      
      // candidates for new clusters
      wHC_ = 1 - sum(wHC);
      wCC_ = 1 - sum(wCC);
      clid0 = arma::find(m == 0);
      if(clid0.n_elem > 0){
        wHC_old = wHC;
        wCC_old = wCC;
        th_j_old = th_j;
        if(max(clid0) >= wHC.n_elem){
          len1 = max(clid0) - wHC.n_elem + 1;
          wHC_add.zeros(len1);
          wCC_add.zeros(len1);
          th_j_add.zeros(len1);
        }
        countwHC = 0;
        countwCC = 0;
        countth_j = 0;
        ncl_new = 0;
        
        while ( ((wHC_ > uHC_star) || (wCC_ > uCC_star)) && (ncl_new < clid0.n_elem)  ){
          vHC_ = R::rbeta(1.0, M_DP);
          if(clid0(ncl_new) < wHC.n_elem){
            wHC_old(clid0(ncl_new)) = wHC_*vHC_;
          }else{
            wHC_add(countwHC) = wHC_*vHC_;
            countwHC++;
          }
          wHC_ *= (1-vHC_);
          
          vCC_ = R::rbeta(1.0, M_DP);
          if(clid0(ncl_new) < wCC.n_elem){
            wCC_old(clid0(ncl_new)) = wCC_*vCC_;
          }else{
            wCC_add(countwCC) = wCC_*vCC_;
            countwCC++;
          }
          wCC_ = wCC_*(1-vCC_);
          
          ncl_new++;
        }
        
        if(max(clid0) >= wHC_old.n_elem){
          wHC = arma::join_cols(wHC_old, wHC_add);
          wCC = arma::join_cols(wCC_old, wCC_add);
        }else{
          wHC = wHC_old;
          wCC = wCC_old;
        }
        
        if(ncl_new != 0){
          clid00 = clid0(arma::span(0, (ncl_new-1)));
          for(i=0; i<clid00.n_elem; i++){
            j = clid00(i);
            if(j < th_j_old.n_elem){
              th_j_old(j) = R::rbeta(a_G0, b_G0);
            }else{
              th_j_add(countth_j) = R::rbeta(a_G0, b_G0);
              countth_j++;
            }
          }
          if(max(clid00) >= th_j_old.n_elem){
            th_j = arma::join_cols(th_j_old, th_j_add);
          }else{
            th_j = th_j_old;
          }
        }
      }
      
      for(c=0; c<th_j.n_elem; c++){
        if(th_j(c) == 0.0){
          th_j(c) = 0.0001;
        }
        if(th_j(c) == 1.0){
          th_j(c) = 0.9999;
        }
      }
      
      // update z
      candi_clHC = arma::find(wHC != 0);
      candi_clCC = arma::find(wCC != 0);
      prop_zHC.zeros(idHC.n_elem);
      prop_zCC = 0;
      z_old = z;
      z.zeros(J);
      if(candi_clHC.n_elem > 1){
        for(i=0; i<idHC.n_elem; i++){
          ll.zeros(candi_clHC.n_elem);
          like.zeros(candi_clHC.n_elem);
          for(c=0; c<candi_clHC.n_elem; c++){
            ll(c) = (uHC(i) < wHC(candi_clHC(c)))*1;
            like(c) = R::dbinom(y(i), n(i), th_j(candi_clHC(c)), false);
          }
          proposal_p = like % ll;
          proposal_p_ = arma::find(proposal_p != 0);
          candi_clHCj = candi_clHC(proposal_p_);
          proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
          len1 = candi_clHCj.n_elem;
          if(len1 == 1){
            prop_zHC(i) = candi_clHCj(0);
          }else if(len1 == 0){
            prop_zHC(i) = z_old(i);
          }else{
            prop_zHC(i) = candi_clHCj(sample_new_cls_p(proposal_p1));
          }
        }
      }else{
        prop_zHC.fill(candi_clHC(0));
      }
      for(i=0; i<idHC.n_elem; i++){
        z(i) = prop_zHC(i);
      }
      
      if(candi_clCC.n_elem > 1){
        ll.zeros(candi_clCC.n_elem);
        like.zeros(candi_clCC.n_elem);
        for(c=0; c<candi_clCC.n_elem; c++){
          ll(c) = (uCC(idCC) < wCC(candi_clCC(c)))*1;
          like(c) = R::dbinom(y(idCC), n(idCC), th_j(candi_clCC(c)), false);
        }
        proposal_p = like % ll;
        proposal_p_ = arma::find(proposal_p != 0);
        candi_clCCj = candi_clCC(proposal_p_);
        proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
        len1 = candi_clCCj.n_elem;
        if(len1 == 1){
          prop_zCC = candi_clCCj(0);
        }else if(len1 == 0){
          prop_zCC = z_old(idCC);
        }else{
          prop_zCC = candi_clCCj(sample_new_cls_p(proposal_p1));
        }
      }else{
        prop_zCC = candi_clCC(0);
      }
      z(idCC) = prop_zCC;
      
      unique_z = arma::unique(z);
      ncl = unique_z.n_elem;
      maxcl = max(arma::unique(z));
      m = tableC2(z, J);
      clid = arma::find(m != 0);
      
      
      // save output
      diff = nit - NBURN + 1;
      if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
        p_aux.zeros(J);
        for(i=0; i<J; i++){
          p_aux(i) = th_j(z(i));
        }
        p_out.row(count) = p_aux.t();
        z_out.row(count) = z.t() + 1;
        indices = arma::regspace<arma::uvec>(0, J - 1);
        sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
        M_DP_out(count) = M_DP;
        phi_out(count) = phi;
        ncl_out(count) = ncl;
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
  
  Rcpp::List ret = Rcpp::List::create(Rcpp::_["p"] = p_out, 
                                      Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["phi"] = phi_out,
                                      Rcpp::_["ncl"] = ncl_out
  );
  return(ret);
}




// MCMC sampling for DPM method
// [[Rcpp::export]]
Rcpp::List DPM_Bin(
    arma::vec y,
    arma::vec n,
    double hyper_gamma_shape = 1,
    double hyper_gamma_scale = 1,
    unsigned int NBURN = 4000,
    unsigned int NTHIN = 10,
    unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  unsigned int J = n.n_elem;
  arma::uvec id(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    id(i) = i;
  }
  
  // hyperparameter
  // base distribution
  double a_G0 = 0.5;
  double b_G0 = 0.5;
  // DP precision parameters
  double M_DP = 1.0;
  double pi = 0.5;
  int s = 0;
  
  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  arma::vec th_j(maxcl+1, arma::fill::zeros);
  for(unsigned int i=0; i<th_j.n_elem; i++){
    th_j(i) = R::rbeta(a_G0, b_G0);
  }
  arma::vec v(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    v(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec w;
  w.zeros(maxcl+1);
  w(0) = v(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      w(c) = v(c);
      for(c_=0; c_<c; c_++){
        w(c) *= (1-v(c_));
      }
    }
  }
  double w_ = v(ncl); // weight for potential new clusters
  
  
  // MCMC output parameters
  arma::mat p_out(NOUTSAMPLE, J);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;
  
  // variables for MCMC
  unsigned int countw, nit, i, j, ncl_new, diff, countth_j;
  arma::vec v1, u, w_old, w_add, sumy, sumn, ll, like, proposal_p, proposal_p1, p_aux, th_j_old, th_j_add;
  arma::uvec candi_cl, prop_z, candi_clj, indices, indices2, proposal_p_, clid0, clid00, z_old;
  double u_star, v_;
  arma::uword len1;
  
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    pi = R::rbeta(M_DP+1, J);
    s = R::rbinom(1, 1-(hyper_gamma_shape+ncl-1)/(J*(hyper_gamma_scale-log(pi))+hyper_gamma_shape+ncl-1));
    M_DP = arma::randg(arma::distr_param(hyper_gamma_shape+ncl-s, 1.0/((1.0/hyper_gamma_scale)-log(pi))));
    
    
    // update theta
    sumy.zeros(maxcl+1);
    sumn.zeros(maxcl+1);
    th_j.zeros(maxcl+1);
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      indices = arma::find(z == c) ;
      sumy(c) = sum(y(indices));
      sumn(c) = sum(n(indices));
    }
    th_j = ran_beta_post(sumy, sumn, a_G0, b_G0);
    
    
    // update v
    w.zeros(maxcl + 1);
    v1.zeros(ncl + 1);
    for(c=0; c<ncl; c++){
      // v1(c) = R::rgamma(m(clid(c)), 1);
      v1(c) = arma::randg(arma::distr_param(m(clid(c)), 1.0));
    }
    // v1(ncl) = R::rgamma(M_DP, 1);
    v1(ncl) = arma::randg(arma::distr_param(M_DP, 1.0));
    for(c=0; c<(ncl+1); c++){
      v(c) = v1(c)/sum(v1);
    }
    for(i=0; i<clid.n_elem; i++){
      c = clid(i);
      w(c) = v(i); // weight for each cluster 
    }
    w_ = v(ncl); // weight for potential new clusters
    
    
    // update u
    u.zeros(J);
    for(i=0; i<J; i++){
      u(i) = R::runif(0.0, 1.0) * w(z(i));
    }
    u_star = arma::min(u);
    
    
    // candidates for new clusters
    clid0 = arma::find(m == 0);
    if(clid0.n_elem > 0){
      w_old = w;
      th_j_old = th_j;
      if(max(clid0) >= w.n_elem){
        len1 = max(clid0) - w.n_elem + 1;
        w_add.zeros(len1);
        th_j_add.zeros(len1);
      }
      countw = 0;
      countth_j = 0;
      ncl_new = 0;
      
      while ( (w_ > u_star) && (ncl_new < clid0.n_elem)  ){
        v_ = R::rbeta(1.0, M_DP);
        if(clid0(ncl_new) < w.n_elem){
          w_old(clid0(ncl_new)) = w_*v_;
        }else{
          w_add(countw) = w_*v_;
          countw++;
        }
        w_ *= (1-v_);
        ncl_new++;
      }
      
      if(max(clid0) >= w_old.n_elem){
        w = arma::join_cols(w_old, w_add);
      }else{
        w = w_old;
      }
      
      if(ncl_new != 0){
        clid00 = clid0(arma::span(0, (ncl_new-1)));
        for(i=0; i<clid00.n_elem; i++){
          j = clid00(i);
          if(j < th_j_old.n_elem){
            th_j_old(j) = R::rbeta(a_G0, b_G0);
          }else{
            th_j_add(countth_j) = R::rbeta(a_G0, b_G0);
            countth_j++;
          }
        }
        if(max(clid00) >= th_j_old.n_elem){
          th_j = arma::join_cols(th_j_old, th_j_add);
        }else{
          th_j = th_j_old;
        }
      }
    }
    
    for(c=0; c<th_j.n_elem; c++){
      if(th_j(c) == 0.0){
        th_j(c) = 0.0001;
      }
      if(th_j(c) == 1.0){
        th_j(c) = 0.9999;
      }
    }
    
    // update z
    candi_cl = arma::find(w != 0);
    prop_z.zeros(id.n_elem);
    z_old = z;
    z.zeros(J);
    if(candi_cl.n_elem > 1){
      for(i=0; i<id.n_elem; i++){
        ll.zeros(candi_cl.n_elem);
        like.zeros(candi_cl.n_elem);
        for(c=0; c<candi_cl.n_elem; c++){
          ll(c) = (u(i) < w(candi_cl(c)))*1;
          like(c) = R::dbinom(y(i), n(i), th_j(candi_cl(c)), false);
        }
        proposal_p = like % ll;
        proposal_p_ = arma::find(proposal_p != 0);
        candi_clj = candi_cl(proposal_p_);
        proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
        len1 = candi_clj.n_elem;
        if(len1 == 1){
          prop_z(i) = candi_clj(0);
        }else if(len1 == 0){
          prop_z(i) = z_old(i);
        }else{
          prop_z(i) = candi_clj(sample_new_cls_p(proposal_p1));
        }
      }
    }else{
      prop_z.fill(candi_cl(0));
    }
    for(i=0; i<id.n_elem; i++){
      z(i) = prop_z(i);
    }
    
    unique_z = arma::unique(z);
    ncl = unique_z.n_elem;
    maxcl = max(arma::unique(z));
    m = tableC2(z, J);
    clid = arma::find(m != 0);
    
    
    // save output
    diff = nit - NBURN + 1;
    if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
      p_aux.zeros(J);
      for(i=0; i<J; i++){
        p_aux(i) = th_j(z(i));
      }
      p_out.row(count) = p_aux.t();
      z_out.row(count) = z.t() + 1;
      indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
      M_DP_out(count) = M_DP;
      ncl_out(count) = ncl;
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
  
  Rcpp::List ret = Rcpp::List::create(Rcpp::_["p"] = p_out, 
                                      Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["ncl"] = ncl_out
  );
  return(ret);
}




// MCMC sampling for DPM method with C+
// [[Rcpp::export]]
Rcpp::List DPM_Bin_Ca(
    arma::vec y,
    arma::vec n,
    double hcauchy_scale = 1,
    double proposal_M_DP_sd = 1,
    unsigned int NBURN = 4000,
    unsigned int NTHIN = 10,
    unsigned int NOUTSAMPLE = 4000,
    const int print = 0
){
  unsigned int J = n.n_elem;
  arma::uvec id(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    id(i) = i;
  }
  
  // hyperparameter
  // base distribution
  double a_G0 = 0.5;
  double b_G0 = 0.5;
  // DP precision parameters
  double M_DP = 1.0;
  
  // initialization for each study allocation
  arma::uvec z(J, arma::fill::zeros);
  for(unsigned int i=0; i<J; i++){
    z(i) = i;
  }
  
  arma::uvec unique_z = arma::unique(z);
  unsigned int ncl = unique_z.n_elem;
  unsigned int maxcl = max(arma::unique(z));
  arma::uvec m = tableC2(z, J);
  arma::uvec clid = arma::find(m != 0);
  
  arma::vec th_j(maxcl+1, arma::fill::zeros);
  for(unsigned int i=0; i<th_j.n_elem; i++){
    th_j(i) = R::rbeta(a_G0, b_G0);
  }
  arma::vec v(maxcl*3);
  for(unsigned int i=0; i<(maxcl*3); i++){
    v(i) = R::rbeta(1.0, M_DP);
  }
  arma::vec w;
  w.zeros(maxcl+1);
  w(0) = v(0);
  unsigned int c, c_;
  if(maxcl != 0){
    for(c=1; c<(maxcl+1);c++){
      w(c) = v(c);
      for(c_=0; c_<c; c_++){
        w(c) *= (1-v(c_));
      }
    }
  }
  double w_ = v(ncl); // weight for potential new clusters
  
  
  // MCMC output parameters
  arma::mat p_out(NOUTSAMPLE, J);
  arma::umat z_out(NOUTSAMPLE, J);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);
  
  unsigned int count = 0;

  // variables for MCMC
  unsigned int countw, nit, i, j, ncl_new, diff, countth_j;
  arma::vec v1, u, w_old, w_add, sumy, sumn, ll, like, proposal_p, proposal_p1, p_aux, th_j_old, th_j_add;
  arma::uvec candi_cl, prop_z, candi_clj, indices, indices2, proposal_p_, clid0, clid00, z_old;
  double u_star, v_, M_DP_prop, lr;
  arma::uword len1;
  
  for(nit=0; nit<(NBURN+NTHIN*NOUTSAMPLE); nit++){
    
    if (nit % 100 == 0)
      Rcpp::checkUserInterrupt();    
    
    // update M_DP
    M_DP_prop = (M_DP + arma::randu() *2.0 - 1.0) * proposal_M_DP_sd;
    if(M_DP_prop < 0){
      M_DP_prop = -M_DP_prop;
    }
    lr = ncl * log(M_DP_prop) + lgamma(M_DP_prop) + log(den_half_cauchy(M_DP_prop, hcauchy_scale))
      - lgamma(M_DP_prop + J)
      - ncl * log(M_DP) - lgamma(M_DP) - log(den_half_cauchy(M_DP, hcauchy_scale))
      + lgamma(M_DP + J);
      if(log(R::runif(0.0, 1.0))<lr){
        M_DP = M_DP_prop;
      }
      
      
      // update theta
      sumy.zeros(maxcl+1);
      sumn.zeros(maxcl+1);
      th_j.zeros(maxcl+1);
      for(i=0; i<clid.n_elem; i++){
        c = clid(i);
        indices = arma::find(z == c) ;
        sumy(c) = sum(y(indices));
        sumn(c) = sum(n(indices));
      }
      th_j = ran_beta_post(sumy, sumn, a_G0, b_G0);
      
      
      // update v
      w.zeros(maxcl + 1);
      v1.zeros(ncl + 1);
      for(c=0; c<ncl; c++){
        // v1(c) = R::rgamma(m(clid(c)), 1);
        v1(c) = arma::randg(arma::distr_param(m(clid(c)), 1.0));
      }
      // v1(ncl) = R::rgamma(M_DP, 1);
      v1(ncl) = arma::randg(arma::distr_param(M_DP, 1.0));
      for(c=0; c<(ncl+1); c++){
        v(c) = v1(c)/sum(v1);
      }
      for(i=0; i<clid.n_elem; i++){
        c = clid(i);
        w(c) = v(i); // weight for each cluster 
      }
      w_ = v(ncl); // weight for potential new clusters
      
      
      // update u
      u.zeros(J);
      for(i=0; i<J; i++){
        u(i) = R::runif(0.0, 1.0) * w(z(i));
      }
      u_star = arma::min(u);
      
      
      // candidates for new clusters
      clid0 = arma::find(m == 0);
      if(clid0.n_elem > 0){
        w_old = w;
        th_j_old = th_j;
        if(max(clid0) >= w.n_elem){
          len1 = max(clid0) - w.n_elem + 1;
          w_add.zeros(len1);
          th_j_add.zeros(len1);
        }
        countw = 0;
        countth_j = 0;
        ncl_new = 0;
        
        while ( (w_ > u_star) && (ncl_new < clid0.n_elem)  ){
          v_ = R::rbeta(1.0, M_DP);
          if(clid0(ncl_new) < w.n_elem){
            w_old(clid0(ncl_new)) = w_*v_;
          }else{
            w_add(countw) = w_*v_;
            countw++;
          }
          w_ *= (1-v_);
          ncl_new++;
        }
        
        if(max(clid0) >= w_old.n_elem){
          w = arma::join_cols(w_old, w_add);
        }else{
          w = w_old;
        }
        
        if(ncl_new != 0){
          clid00 = clid0(arma::span(0, (ncl_new-1)));
          for(i=0; i<clid00.n_elem; i++){
            j = clid00(i);
            if(j < th_j_old.n_elem){
              th_j_old(j) = R::rbeta(a_G0, b_G0);
            }else{
              th_j_add(countth_j) = R::rbeta(a_G0, b_G0);
              countth_j++;
            }
          }
          if(max(clid00) >= th_j_old.n_elem){
            th_j = arma::join_cols(th_j_old, th_j_add);
          }else{
            th_j = th_j_old;
          }
        }
      }
      
      for(c=0; c<th_j.n_elem; c++){
        if(th_j(c) == 0.0){
          th_j(c) = 0.0001;
        }
        if(th_j(c) == 1.0){
          th_j(c) = 0.9999;
        }
      }
      
      // update z
      candi_cl = arma::find(w != 0);
      prop_z.zeros(id.n_elem);
      z_old = z;
      z.zeros(J);
      if(candi_cl.n_elem > 1){
        for(i=0; i<id.n_elem; i++){
          ll.zeros(candi_cl.n_elem);
          like.zeros(candi_cl.n_elem);
          for(c=0; c<candi_cl.n_elem; c++){
            ll(c) = (u(i) < w(candi_cl(c)))*1;
            like(c) = R::dbinom(y(i), n(i), th_j(candi_cl(c)), false);
          }
          proposal_p = like % ll;
          proposal_p_ = arma::find(proposal_p != 0);
          candi_clj = candi_cl(proposal_p_);
          proposal_p1 = proposal_p(proposal_p_)/sum(proposal_p);
          len1 = candi_clj.n_elem;
          if(len1 == 1){
            prop_z(i) = candi_clj(0);
          }else if(len1 == 0){
            prop_z(i) = z_old(i);
          }else{
            prop_z(i) = candi_clj(sample_new_cls_p(proposal_p1));
          }
        }
      }else{
        prop_z.fill(candi_cl(0));
      }
      for(i=0; i<id.n_elem; i++){
        z(i) = prop_z(i);
      }
      
      unique_z = arma::unique(z);
      ncl = unique_z.n_elem;
      maxcl = max(arma::unique(z));
      m = tableC2(z, J);
      clid = arma::find(m != 0);
      
      
      // save output
      diff = nit - NBURN + 1;
      if((nit + 1 > NBURN) && ((diff / NTHIN)*NTHIN == diff)){
        p_aux.zeros(J);
        for(i=0; i<J; i++){
          p_aux(i) = th_j(z(i));
        }
        p_out.row(count) = p_aux.t();
        z_out.row(count) = z.t() + 1;
        indices = arma::regspace<arma::uvec>(0, J - 1);
        sim_mat = sim_mat + (arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1));
        M_DP_out(count) = M_DP;
        ncl_out(count) = ncl;
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
  
  Rcpp::List ret = Rcpp::List::create(Rcpp::_["p"] = p_out, 
                                      Rcpp::_["z"] = z_out, 
                                      Rcpp::_["sim_mat"] = sim_mat,
                                      Rcpp::_["M_DP"] = M_DP_out, 
                                      Rcpp::_["ncl"] = ncl_out
                                        );
  return(ret);
}


