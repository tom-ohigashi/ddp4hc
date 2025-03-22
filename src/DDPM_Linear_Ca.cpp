// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "others.h"

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
