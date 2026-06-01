
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "others.h"
using namespace Rcpp;

// -----------------------------------------------------------------------------
// DDPM_Cont.cpp
// Slice-sampler DDPM for continuous endpoints from arm-level summary statistics.
// This follows the retrospective slice-sampler structure used in DDPM_Bin.cpp,
// replacing the beta-binomial component by a normal summary-likelihood component.
//
// Input convention:
//   ybar = c(historical control means..., current control mean)
//   sd   = c(historical control SDs...,    current control SD)
//   n    = c(historical control ns...,     current control n)
//
// The last element is treated as the current control.
// Historical controls use weights wHC; the current control uses weights wCC.
// Atoms are shared, and dependence between the two random probability measures
// is induced through shared stick-breaking increments with probability phi.
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List DDPM_Cont(
    arma::vec ybar,
    arma::vec sd,
    arma::vec n,
    double hyper_gamma_shape = 1,
    double hyper_gamma_scale = 1,
    double proposal_phi_sd = 0.1,
    double mu_G0 = 0,
    double tau2_G0 = 10000,
    double phi_gamma1 = 2,
    double phi_gamma2 = 2,
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

  arma::uvec idHC(J - 1, arma::fill::zeros);
  for (unsigned int i = 0; i < J - 1; i++) idHC(i) = i;
  unsigned int idCC = J - 1;

  double M_DP = 1.0;
  double phi = 0.5;

  // Initialization: one occupied cluster per source.
  arma::uvec z(J, arma::fill::zeros);
  for (unsigned int i = 0; i < J; i++) z(i) = i;

  unsigned int K = J;
  arma::vec vHC(K, arma::fill::zeros), vCC(K, arma::fill::zeros);
  arma::vec wHC(K, arma::fill::zeros), wCC(K, arma::fill::zeros);
  arma::uvec shared(K, arma::fill::zeros);

  double remHC = 1.0, remCC = 1.0;
  for (unsigned int c = 0; c < K; c++) {
    shared(c) = (R::runif(0.0, 1.0) < phi) ? 1 : 0;

    if (shared(c) == 1) {
      double vv = R::rbeta(1.0, M_DP);
      if (vv >= 0.9999) vv = 0.9999;
      if (vv <= 0.0001) vv = 0.0001;
      vHC(c) = vv;
      vCC(c) = vv;
    } else {
      vHC(c) = R::rbeta(1.0, M_DP);
      vCC(c) = R::rbeta(1.0, M_DP);
      if (vHC(c) >= 0.9999) vHC(c) = 0.9999;
      if (vHC(c) <= 0.0001) vHC(c) = 0.0001;
      if (vCC(c) >= 0.9999) vCC(c) = 0.9999;
      if (vCC(c) <= 0.0001) vCC(c) = 0.0001;
    }

    wHC(c) = remHC * vHC(c);
    remHC *= (1.0 - vHC(c));
    wCC(c) = remCC * vCC(c);
    remCC *= (1.0 - vCC(c));
  }

  arma::vec mu_star(K, arma::fill::zeros);
  for (unsigned int c = 0; c < K; c++) {
    mu_star(c) = R::rnorm(mu_G0, std::sqrt(tau2_G0));
  }

  arma::vec mu_C_out(NOUTSAMPLE, arma::fill::zeros);
  arma::mat z_out(NOUTSAMPLE, J, arma::fill::zeros);
  arma::mat sim_mat(J, J, arma::fill::zeros);
  arma::vec M_DP_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec phi_out(NOUTSAMPLE, arma::fill::zeros);
  arma::vec ncl_out(NOUTSAMPLE, arma::fill::zeros);

  unsigned int count = 0;

  for (unsigned int nit = 0; nit < NBURN + NTHIN * NOUTSAMPLE; nit++) {
    if (nit % 100 == 0) Rcpp::checkUserInterrupt();

    unsigned int maxcl = z.max();

    // Keep only instantiated clusters up to max occupied label before updating.
    vHC = vHC.head(maxcl + 1);
    vCC = vCC.head(maxcl + 1);
    wHC = wHC.head(maxcl + 1);
    wCC = wCC.head(maxcl + 1);
    mu_star = mu_star.head(maxcl + 1);
    shared = shared.head(maxcl + 1);

    // 1. Update dependent stick-breaking increments.
    //    This is the same slice-sampler block as DDPM_Bin, but expressed with
    //    an explicit shared-stick indicator for numerical clarity.
    for (unsigned int c = 0; c < maxcl + 1; c++) {
      unsigned int nH_c = 0, tailH = 0;
      for (arma::uword ii = 0; ii < idHC.n_elem; ii++) {
        unsigned int j = idHC(ii);
        if (z(j) == c) nH_c++;
        if (z(j) > c) tailH++;
      }

      unsigned int nC_c = (z(idCC) == c) ? 1 : 0;
      unsigned int tailC = (z(idCC) > c) ? 1 : 0;

      double log_shared =
        std::log(phi) +
        log_beta_fn_ddpm_cont(1.0 + nH_c + nC_c,
                              M_DP + tailH + tailC) -
        log_beta_fn_ddpm_cont(1.0, M_DP);

      double log_ind =
        std::log(1.0 - phi) +
        log_beta_fn_ddpm_cont(1.0 + nH_c, M_DP + tailH) -
        log_beta_fn_ddpm_cont(1.0, M_DP) +
        log_beta_fn_ddpm_cont(1.0 + nC_c, M_DP + tailC) -
        log_beta_fn_ddpm_cont(1.0, M_DP);

      double mx = std::max(log_shared, log_ind);
      double p_shared = std::exp(log_shared - mx) /
        (std::exp(log_shared - mx) + std::exp(log_ind - mx));

      shared(c) = (R::runif(0.0, 1.0) < p_shared) ? 1 : 0;

      if (shared(c) == 1) {
        double vv = R::rbeta(1.0 + nH_c + nC_c,
                             M_DP + tailH + tailC);
        if (vv >= 0.9999) vv = 0.9999;
        if (vv <= 0.0001) vv = 0.0001;
        vHC(c) = vv;
        vCC(c) = vv;
      } else {
        vHC(c) = R::rbeta(1.0 + nH_c, M_DP + tailH);
        vCC(c) = R::rbeta(1.0 + nC_c, M_DP + tailC);
        if (vHC(c) >= 0.9999) vHC(c) = 0.9999;
        if (vHC(c) <= 0.0001) vHC(c) = 0.0001;
        if (vCC(c) >= 0.9999) vCC(c) = 0.9999;
        if (vCC(c) <= 0.0001) vCC(c) = 0.0001;
      }
    }

    recompute_weights_ddpm_cont(wHC, vHC);
    recompute_weights_ddpm_cont(wCC, vCC);

    // 2. Update phi.
    //    The argument proposal_phi_sd is kept for API compatibility with
    //    DDPM_Bin.cpp.  Here phi is updated directly from its conjugate
    //    beta full conditional using the explicit shared-stick indicators.
    unsigned int n_shared = arma::accu(shared);
    phi = R::rbeta(phi_gamma1 + static_cast<double>(n_shared),
                   phi_gamma2 + static_cast<double>(shared.n_elem - n_shared));
    if (phi >= 0.999999) phi = 0.999999;
    if (phi <= 0.000001) phi = 0.000001;

    // 3. Update component means.
    update_component_means_ddpm_cont(mu_star, z, ybar, var_y, mu_G0, tau2_G0);

    // 4. Update slice variables.
    arma::vec uHC(idHC.n_elem, arma::fill::zeros);
    for (arma::uword ii = 0; ii < idHC.n_elem; ii++) {
      unsigned int j = idHC(ii);
      uHC(ii) = R::runif(0.0, wHC(z(j)));
    }
    arma::vec uCC(1, arma::fill::zeros);
    uCC(0) = R::runif(0.0, wCC(z(idCC)));

    double uHC_star = uHC.min();
    double uCC_star = uCC.min();

    // 5. Retrospectively instantiate new clusters until both remaining sticks
    //    are below their slice thresholds.
    double tailHC = 1.0 - arma::accu(wHC);
    double tailCC = 1.0 - arma::accu(wCC);
    if (tailHC < 0.0) tailHC = 0.0;
    if (tailCC < 0.0) tailCC = 0.0;

    while ((tailHC > uHC_star) || (tailCC > uCC_star)) {
      unsigned int sh = (R::runif(0.0, 1.0) < phi) ? 1 : 0;
      double vH_new, vC_new;

      if (sh == 1) {
        double vv = R::rbeta(1.0, M_DP);
        if (vv >= 0.9999) vv = 0.9999;
        if (vv <= 0.0001) vv = 0.0001;
        vH_new = vv;
        vC_new = vv;
      } else {
        vH_new = R::rbeta(1.0, M_DP);
        vC_new = R::rbeta(1.0, M_DP);
        if (vH_new >= 0.9999) vH_new = 0.9999;
        if (vH_new <= 0.0001) vH_new = 0.0001;
        if (vC_new >= 0.9999) vC_new = 0.9999;
        if (vC_new <= 0.0001) vC_new = 0.0001;
      }

      double wH_new = tailHC * vH_new;
      tailHC *= (1.0 - vH_new);
      double wC_new = tailCC * vC_new;
      tailCC *= (1.0 - vC_new);

      vHC = arma::join_cols(vHC, arma::vec(1).fill(vH_new));
      vCC = arma::join_cols(vCC, arma::vec(1).fill(vC_new));
      wHC = arma::join_cols(wHC, arma::vec(1).fill(wH_new));
      wCC = arma::join_cols(wCC, arma::vec(1).fill(wC_new));
      shared = arma::join_cols(shared, arma::uvec(1).fill(sh));
      mu_star = arma::join_cols(
        mu_star,
        arma::vec(1).fill(R::rnorm(mu_G0, std::sqrt(tau2_G0)))
      );
    }

    // 6. Update historical allocations.
    for (arma::uword ii = 0; ii < idHC.n_elem; ii++) {
      unsigned int j = idHC(ii);
      arma::uvec candi = arma::find(wHC > uHC(ii));
      arma::vec logp(candi.n_elem, arma::fill::zeros);

      for (arma::uword h = 0; h < candi.n_elem; h++) {
        unsigned int c = candi(h);
        logp(h) = R::dnorm(ybar(j), mu_star(c), std::sqrt(var_y(j)), 1);
      }

      logp -= logp.max();
      arma::vec p = arma::exp(logp);
      p /= arma::accu(p);

      int sampled = sample_index_from_prob_ddpm_cont(p);
      z(j) = candi(sampled);
    }

    // 7. Update current-control allocation.
    {
      arma::uvec candi = arma::find(wCC > uCC(0));
      arma::vec logp(candi.n_elem, arma::fill::zeros);

      for (arma::uword h = 0; h < candi.n_elem; h++) {
        unsigned int c = candi(h);
        logp(h) = R::dnorm(ybar(idCC), mu_star(c), std::sqrt(var_y(idCC)), 1);
      }

      logp -= logp.max();
      arma::vec p = arma::exp(logp);
      p /= arma::accu(p);

      int sampled = sample_index_from_prob_ddpm_cont(p);
      z(idCC) = candi(sampled);
    }

    // 8. Update DP precision.
    unsigned int ncl = n_occupied_ddpm_cont(z);
    M_DP = sample_dp_precision_ddpm_cont(
      M_DP, J, ncl, hyper_gamma_shape, hyper_gamma_scale
    );

    // 9. Save.
    unsigned int diff = nit - NBURN + 1;
    if ((nit + 1 > NBURN) && ((diff / NTHIN) * NTHIN == diff)) {
      mu_C_out(count) = mu_star(z(idCC));
      z_out.row(count) = arma::conv_to<arma::rowvec>::from(z + 1);

      arma::uvec indices = arma::regspace<arma::uvec>(0, J - 1);
      sim_mat += arma::conv_to<arma::mat>::from(
        arma::repmat(z(indices), 1, J) == arma::repmat(z(indices).t(), J, 1)
      );

      M_DP_out(count) = M_DP;
      phi_out(count) = phi;
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
    Rcpp::_["phi"] = phi_out,
    Rcpp::_["ncl"] = ncl_out
  );
  return ret;
}
