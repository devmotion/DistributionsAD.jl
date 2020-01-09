module DistributionsAD

using PDMats, 
      ForwardDiff, 
      Zygote, 
      Tracker, 
      LinearAlgebra, 
      Distributions, 
      Random, 
      Combinatorics

using Tracker: TrackedReal
using LinearAlgebra: copytri!
using Distributions: AbstractMvLogNormal, 
                     ContinuousMultivariateDistribution

import StatsFuns: logsumexp, 
                  binomlogpdf, 
                  nbinomlogpdf, 
                  poislogpdf, 
                  nbetalogpdf
import Distributions: MvNormal, 
                      MvLogNormal, 
                      poissonbinomial_pdf_fft, 
                      logpdf, 
                      quantile, 
                      PoissonBinomial

export TuringScalMvNormal,
       TuringDiagMvNormal,
       TuringDenseMvNormal,
       TuringMvLogNormal,
       TuringPoissonBinomial,
       Multi,
       ArrayDist

include("common.jl")
include("univariate.jl")
include("multivariate.jl")
include("multi.jl")
include("array_dist.jl")

end
