% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/models.R
\name{DDPM_Bin}
\alias{DDPM_Bin}
\title{DDPM method with binomial model for aggregated study-level data}
\usage{
DDPM_Bin(
  y,
  n,
  hyper_gamma_shape = 1,
  hyper_gamma_scale = 1,
  proposal_phi_sd = 0.1,
  NBURN = 4000L,
  NTHIN = 10L,
  NOUTSAMPLE = 4000L,
  print = 0L
)
}
\arguments{
\item{y}{Response variable vector where the last element corresponds to current control data}

\item{n}{Number of participant vector where the last element corresponds to current control data}

\item{hyper_gamma_shape}{shape parameter for concentration parameter's prior in a DP, with a default value of 1}

\item{hyper_gamma_scale}{scale parameter for concentration parameter's prior in a DP, with a default value of 1}

\item{proposal_phi_sd}{standard deviation of truncated normal distribution as a proposal for phi, with a default value of 0.1}

\item{NBURN}{Number of iterations for the burn-in phase, with a default value of 4000}

\item{NTHIN}{Thinning rate applied to the chain, with a default value of 10}

\item{NOUTSAMPLE}{Mumber of iterations in the output chain, with a default value of 4000}

\item{print}{Specifies whether to display the MCMC output during execution with a default value of 0 (none)}
}
\description{
DDPM method with binomial model for aggregated study-level data
}
\examples{
# No examples provided yet.

}
