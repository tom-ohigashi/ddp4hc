// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "others.h"

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
