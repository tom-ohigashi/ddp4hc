#' DDPM method with Linear regression model
#'
#' @param y_C Response variable for Current trial
#' @param x_C covariate matrix for Current trial
#' @param g_C Group identifier Current trial
#' @param y_h Response variable for historical control
#' @param x_h covariate matrix for historical control
#' @param study_id_h study indicator for historical control
#' @param hyper_gamma_shape shape parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param hyper_gamma_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param mu_beta_G0 mean parameter for regression coefficient's prior, with a default value of 0
#' @param tau_beta_G0 variance parameter for regression coefficient's prior, with a default value of 10000
#' @param alpha_sigma_G0 shape parameter of Gamma prior for error variance with a default value of 1
#' @param beta_sigma_G0 scale parameter of Gamma prior for error variance with a default value of 1
#' @param proposal_phi_sd standard deviation of truncated normal distribution as a proposal for phi, with a default value of 0.1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DDPM_Linear <- function(y_C, x_C, g_C, y_h, x_h, study_id_h, hyper_gamma_shape = 1, hyper_gamma_scale = 1, mu_beta_G0 = 0, tau_beta_G0 = 10000, alpha_sigma_G0 = 1, beta_sigma_G0 = 1, proposal_phi_sd = 0.1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  n_C <- length(y_C)
  n_h <- length(y_h)
  .Call('_ddp4hc_DDPM_Linear', y_C, x_C, g_C, n_C, y_h, x_h, study_id_h, n_h, hyper_gamma_shape, hyper_gamma_scale, mu_beta_G0, tau_beta_G0, alpha_sigma_G0, beta_sigma_G0, proposal_phi_sd, NBURN, NTHIN, NOUTSAMPLE, print)
}


#' DDPM method with Linear regression model and Cauchy prior for concentration parameter for DP
#'
#' @param y_C Response variable for Current trial
#' @param x_C covariate matrix for Current trial
#' @param g_C Group identifier Current trial
#' @param y_h Response variable for historical control
#' @param x_h covariate matrix for historical control
#' @param study_id_h study indicator for historical control
#' @param hcauchy_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param mu_beta_G0 mean parameter for regression coefficient's prior, with a default value of 0
#' @param tau_beta_G0 variance parameter for regression coefficient's prior, with a default value of 10000
#' @param alpha_sigma_G0 shape parameter of Gamma prior for error variance with a default value of 1
#' @param beta_sigma_G0 scale parameter of Gamma prior for error variance with a default value of 1
#' @param proposal_phi_sd standard deviation of truncated normal distribution as a proposal for phi, with a default value of 0.1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DDPM_Linear_Ca <- function(y_C, x_C, g_C, y_h, x_h, study_id_h, hcauchy_scale = 1, mu_beta_G0 = 0, tau_beta_G0 = 10000, alpha_sigma_G0 = 1, beta_sigma_G0 = 1, proposal_phi_sd = 0.1, proposal_M_DP_sd = 1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  n_C <- length(y_C)
  n_h <- length(y_h)
  .Call('_ddp4hc_DDPM_Linear_Ca', y_C, x_C, g_C, n_C, y_h, x_h, study_id_h, n_h, hcauchy_scale, mu_beta_G0, tau_beta_G0, alpha_sigma_G0, beta_sigma_G0, proposal_phi_sd, proposal_M_DP_sd, NBURN, NTHIN, NOUTSAMPLE, print)
}


#' DPM method with Linear regression model
#'
#' @param y_C Response variable for Current trial
#' @param x_C covariate matrix for Current trial
#' @param g_C Group identifier Current trial
#' @param y_h Response variable for historical control
#' @param x_h covariate matrix for historical control
#' @param study_id_h study indicator for historical control
#' @param hyper_gamma_shape shape parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param hyper_gamma_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param mu_beta_G0 mean parameter for regression coefficient's prior, with a default value of 0
#' @param tau_beta_G0 variance parameter for regression coefficient's prior, with a default value of 10000
#' @param alpha_sigma_G0 shape parameter of Gamma prior for error variance with a default value of 1
#' @param beta_sigma_G0 scale parameter of Gamma prior for error variance with a default value of 1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DPM_Linear <- function(y_C, x_C, g_C, y_h, x_h, study_id_h, hyper_gamma_shape = 1, hyper_gamma_scale = 1, mu_beta_G0 = 0, tau_beta_G0 = 10000, alpha_sigma_G0 = 1, beta_sigma_G0 = 1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  n_C <- length(y_C)
  n_h <- length(y_h)
  .Call('_ddp4hc_DPM_Linear', y_C, x_C, g_C, n_C, y_h, x_h, study_id_h, n_h, hyper_gamma_shape, hyper_gamma_scale, mu_beta_G0, tau_beta_G0, alpha_sigma_G0, beta_sigma_G0, NBURN, NTHIN, NOUTSAMPLE, print)
}


#' DPM method with Linear regression model and Cauchy prior for concentration parameter for DP
#'
#' @param y_C Response variable for Current trial
#' @param x_C covariate matrix for Current trial
#' @param g_C Group identifier Current trial
#' @param y_h Response variable for historical control
#' @param x_h covariate matrix for historical control
#' @param study_id_h study indicator for historical control
#' @param hyper_gamma_shape shape parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param hyper_gamma_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param mu_beta_G0 mean parameter for regression coefficient's prior, with a default value of 0
#' @param tau_beta_G0 variance parameter for regression coefficient's prior, with a default value of 10000
#' @param alpha_sigma_G0 shape parameter of Gamma prior for error variance with a default value of 1
#' @param beta_sigma_G0 scale parameter of Gamma prior for error variance with a default value of 1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DPM_Linear_Ca <- function(y_C, x_C, g_C, y_h, x_h, study_id_h, hcauchy_scale = 1, mu_beta_G0 = 0, tau_beta_G0 = 10000, alpha_sigma_G0 = 1, beta_sigma_G0 = 1, proposal_M_DP_sd = 1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  n_C <- length(y_C)
  n_h <- length(y_h)
  .Call('_ddp4hc_DPM_Linear_Ca', y_C, x_C, g_C, n_C, y_h, x_h, study_id_h, n_h, hcauchy_scale, mu_beta_G0, tau_beta_G0, alpha_sigma_G0, beta_sigma_G0, proposal_M_DP_sd, NBURN, NTHIN, NOUTSAMPLE, print)
}


#'  Linear regression model
#'
#' @param y Response variable
#' @param x Predictor matrix
#' @param g Group identifier
#' @param mu_beta_G0 mean parameter for regression coefficient's prior, with a default value of 0
#' @param tau_beta_G0 variance parameter for regression coefficient's prior, with a default value of 10000
#' @param alpha_sigma_G0 shape parameter of Gamma prior for error variance with a default value of 1
#' @param beta_sigma_G0 scale parameter of Gamma prior for error variance with a default value of 1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
Simple_Linear <- function(y, x, g, mu_beta_G0 = 0, tau_beta_G0 = 10000, alpha_sigma_G0 = 1, beta_sigma_G0 = 1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  .Call('_ddp4hc_Simple_Linear', y, x, g, mu_beta_G0, tau_beta_G0, alpha_sigma_G0, beta_sigma_G0, NBURN, NTHIN, NOUTSAMPLE, print)
}


#' DDPM method with binomial model for aggregated study-level data
#'
#' @param y Response variable vector where the last element corresponds to current control data
#' @param n Number of participant vector where the last element corresponds to current control data
#' @param hyper_gamma_shape shape parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param hyper_gamma_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param proposal_phi_sd standard deviation of truncated normal distribution as a proposal for phi, with a default value of 0.1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DDPM_Bin <- function(y, n, hyper_gamma_shape = 1, hyper_gamma_scale = 1, proposal_phi_sd = 0.1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  .Call('_ddp4hc_DDPM_Bin', y, n, hyper_gamma_shape, hyper_gamma_scale, proposal_phi_sd, NBURN, NTHIN, NOUTSAMPLE, print)
}


#' DDPM method with binomial model for aggregated study-level data and Cauchy prior for concentration parameter for DP
#'
#' @param y Response variable vector where the last element corresponds to current control data
#' @param n Number of participant vector where the last element corresponds to current control data
#' @param hcauchy_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param proposal_M_DP_sd proposal for M_DP, with a default value of 1
#' @param proposal_phi_sd standard deviation of truncated normal distribution as a proposal for phi, with a default value of 0.1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DDPM_Bin_Ca <- function(y, n, hcauchy_scale = 1, proposal_M_DP_sd = 1, proposal_phi_sd = 0.1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  .Call('_ddp4hc_DDPM_Bin_Ca', y, n, hcauchy_scale, proposal_M_DP_sd, proposal_phi_sd, NBURN, NTHIN, NOUTSAMPLE, print)
}


#' DPM method with binomial model for aggregated study-level data
#'
#' @param y Response variable vector where the last element corresponds to current control data
#' @param n Number of participant vector where the last element corresponds to current control data
#' @param hyper_gamma_shape shape parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param hyper_gamma_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DPM_Bin <- function(y, n, hyper_gamma_shape = 1, hyper_gamma_scale = 1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  .Call('_ddp4hc_DPM_Bin', y, n, hyper_gamma_shape, hyper_gamma_scale, NBURN, NTHIN, NOUTSAMPLE, print)
}

#' DPM method with binomial model for aggregated study-level data and Cauchy prior for concentration parameter for DP
#'
#' @param y Response variable vector where the last element corresponds to current control data
#' @param n Number of participant vector where the last element corresponds to current control data
#' @param hcauchy_scale scale parameter for concentration parameter's prior in a DP, with a default value of 1
#' @param proposal_M_DP_sd proposal for M_DP, with a default value of 1
#' @param NBURN Number of iterations for the burn-in phase, with a default value of 4000
#' @param NTHIN Thinning rate applied to the chain, with a default value of 10
#' @param NOUTSAMPLE Mumber of iterations in the output chain, with a default value of 4000
#' @param print Specifies whether to display the MCMC output during execution with a default value of 0 (none)
#' 
#' @examples
#' # No examples provided yet.
#' 
#' @export
DPM_Bin_Ca <- function(y, n, hcauchy_scale = 1, proposal_M_DP_sd = 1, NBURN = 4000L, NTHIN = 10L, NOUTSAMPLE = 4000L, print = 0L) {
  .Call('_ddp4hc_DPM_Bin_Ca', y, n, hcauchy_scale, proposal_M_DP_sd, NBURN, NTHIN, NOUTSAMPLE, print)
}
