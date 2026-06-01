
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "others.h"
using namespace Rcpp;

// -----------------------------------------------------------------------------
// DPM_Cont.cpp
// Slice-sampler DPM for continuous endpoints from arm-level summary statistics.
// This follows the retrospective slice-sampler structure used in DPM_Bin.cpp,
// replacing the beta-binomial component by a normal summary-likelihood component.
//
// Input convention:
//   ybar = c(historical control means..., current control mean)
//   sd   = c(historical control SDs...,    current control SD)
//   n    = c(historical control ns...,     current control n)
//
// The last element is treated as the current control.
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List DPM_Cont(
    arma::vec ybar,
    arma::vec sd,
    arma::vec n,
    double hyper_gamma_shape = 1,
    double hyper_gamma_scale = 1,
    double mu_G0 = 0,
    double tau2_G0 = 10000,
    unsigned int NBURN = 4000,
    unsigned int NTHIN = 10,
    unsigned int NOUTSAMPLE = 4000,
    const int print = 0
) {
  unsigned int J = ybar.n_elem;
  if (sd.n_elem != J || n.n_elem != J) {
    Rcpp::stop("ybar, sd, and n must have the same length.");
  }
  if (J < 2) {
    Rcpp::stop("At least one historical control and one current control are required.");
  }
  if (arma::any(sd <= 0) || arma::any(n <= 0)) {
    Rcpp::stop("All sd and n values must be positive.");
  }
  if (tau2_G0 <= 0) {
    Rcpp::stop("tau2_G0 must be positive.");
  }

  arma::vec var_y = arma::square(sd) / n;
  unsigned int idCC = J - 1;

  // Initialization: one occupied cluster per source.
  arma::uvec z(J, arma::fill::zeros);
  for (unsigned int i = 0; i < J; i++) z(i) = i;

  double M_DP = 1.0;
  unsigned int K = J;

  arma::vec v(K, arma::fill::zeros);
  arma::vec w(K, arma::fill::zeros);
  double rem = 1.0;
  for (unsigned int c = 0; c < K; c++) {
    v(c) = R::rbeta(1.0, M_DP);
    if (v(c) > 0.9999) v(c) = 0.9999;
    w(c) = rem * v(c);
    rem *= (1.0 - v(c));
  }

  arma::vec mu_star(K, arma::fill::zeros);
  for (unsigned int c = 0; c < K; c++) {
    mu_star(c) = R::rnorm(mu_G0, std::sqrt(tau2_G0));
  }

  arma::vec mu_C_out(NOUTSAMPLE, arma::fill::zeros);
  arma::mat z_out(NOUTSAMPLE, J, arma::fill::zeros);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);

  unsigned int count = 0;

  for (unsigned int nit = 0; nit < NBURN + NTHIN * NOUTSAMPLE; nit++) {
    if (nit % 100 == 0) Rcpp::checkUserInterrupt();

    unsigned int maxcl = z.max();
    if (maxcl + 1 > v.n_elem) {
      Rcpp::stop("Internal error: allocation index exceeds instantiated sticks.");
    }

    // 1. Update v and w for instantiated clusters up to max occupied label.
    arma::uvec m = tabulate_alloc_cont(z, maxcl + 1);
    v.set_size(maxcl + 1);
    w.set_size(maxcl + 1);

    for (unsigned int c = 0; c < maxcl + 1; c++) {
      unsigned int I1 = m(c);
      unsigned int I2 = 0;
      for (unsigned int cc = c + 1; cc < maxcl + 1; cc++) I2 += m(cc);

      v(c) = R::rbeta(1.0 + static_cast<double>(I1),
                      M_DP + static_cast<double>(I2));
      if (v(c) >= 0.9999) v(c) = 0.9999;
      if (v(c) <= 0.0001) v(c) = 0.0001;
    }
    recompute_stick_weights_cont(w, v);

    // Truncate atom vector to current instantiated length if needed.
    mu_star = mu_star.head(maxcl + 1);

    // 2. Update component means.
    update_component_means_cont(mu_star, z, ybar, var_y, mu_G0, tau2_G0);

    // 3. Update slice variables.
    arma::vec u(J, arma::fill::zeros);
    for (unsigned int i = 0; i < J; i++) {
      u(i) = R::runif(0.0, w(z(i)));
    }
    double u_star = u.min();

    // 4. Retrospectively instantiate new clusters until remaining stick is below u_star.
    double w_tail = 1.0 - arma::accu(w);
    if (w_tail < 0.0) w_tail = 0.0;

    while (w_tail > u_star) {
      double v_new = R::rbeta(1.0, M_DP);
      if (v_new >= 0.9999) v_new = 0.9999;
      if (v_new <= 0.0001) v_new = 0.0001;

      double w_new = w_tail * v_new;
      w_tail *= (1.0 - v_new);

      v = arma::join_cols(v, arma::vec(1).fill(v_new));
      w = arma::join_cols(w, arma::vec(1).fill(w_new));
      mu_star = arma::join_cols(
        mu_star,
        arma::vec(1).fill(R::rnorm(mu_G0, std::sqrt(tau2_G0)))
      );
    }

    // 5. Update allocations using the slice restriction u_i < w_c.
    for (unsigned int i = 0; i < J; i++) {
      arma::uvec candi = arma::find(w > u(i));
      arma::vec logp(candi.n_elem, arma::fill::zeros);

      for (arma::uword h = 0; h < candi.n_elem; h++) {
        unsigned int c = candi(h);
        logp(h) = R::dnorm(ybar(i), mu_star(c), std::sqrt(var_y(i)), 1);
      }

      logp -= logp.max();
      arma::vec p = arma::exp(logp);
      p /= arma::accu(p);

      int sampled = sample_index_from_prob_cont(p);
      z(i) = candi(sampled);
    }

    // 6. Update DP precision.
    unsigned int ncl = n_occupied_cont(z);
    M_DP = sample_dp_precision_cont(
      M_DP, J, ncl, hyper_gamma_shape, hyper_gamma_scale
    );

    // 7. Save.
    unsigned int diff = nit - NBURN + 1;
    if ((nit + 1 > NBURN) && ((diff / NTHIN) * NTHIN == diff)) {
      mu_C_out(count) = mu_star(z(idCC));
      z_out.row(count) = arma::conv_to<arma::rowvec>::from(z + 1);

      arma::uvec indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat += arma::conv_to<arma::mat>::from(
        arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1)
      );

      M_DP_out(count) = M_DP;
      ncl_out(count) = ncl;
      count++;
    }

    if (print == 1.0) {
      if (nit + 1 <= NBURN) {
        if (10 * (nit + 1) / (1.0 * NBURN) ==
            round(10 * (nit + 1) / (NBURN))) {
          Rcpp::Rcout << "Burn-in " << (100 * (nit + 1) / (NBURN))
                      << "% completed \n";
        }
      } else {
        if ((10 * (nit + 1 - NBURN) / (1.0 * NTHIN * NOUTSAMPLE)) ==
            round(10 * (nit + 1 - NBURN) / (NTHIN * NOUTSAMPLE))) {
          Rcpp::Rcout << "MCMC "
                      << (100 * (nit + 1 - NBURN) / (NTHIN * NOUTSAMPLE))
                      << "% completed \n";
        }
      }
    }
  }

  sim_mat /= static_cast<double>(NOUTSAMPLE);

  Rcpp::List ret = Rcpp::List::create(
    Rcpp::_["mu"] = mu_C_out,
    Rcpp::_["mu_C"] = mu_C_out,
    Rcpp::_["z"] = z_out,
    Rcpp::_["sim_mat"] = sim_mat,
    Rcpp::_["M_DP"] = M_DP_out,
    Rcpp::_["ncl"] = ncl_out
  );
  return ret;
}
