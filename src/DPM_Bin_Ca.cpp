// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "others.h"

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
