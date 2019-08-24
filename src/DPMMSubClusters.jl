__precompile__()

module DPMMSubClusters


using Distributed
using DistributedArrays
using StatsBase
using Distributions
using SpecialFunctions
using CatViews
using LinearAlgebra
using JLD2
using Clustering

import Random:seed!


include("ds.jl")

#Priors:
include("priors/niw.jl")
include("priors/multinomial_prior.jl")

#Distributions:
include("distributions/mv_gaussian.jl")
include("distributions/multinomial_dist.jl")

include("utils.jl")
include("shared_actions.jl")
include("local_clusters_actions.jl")
include("global_params.jl")
include("dp-parallel-sampling.jl")
include("data_generators.jl")

if length(procs()) == 1
    new_procs = addprocs(1)
end


export generate_gaussian_data, generate_mnmm_data, dp_parallel_sampling, dp_parallel, run_model_from_checkpoint, save_model, calculate_posterior, fit, get_labels_histogram


end # module
