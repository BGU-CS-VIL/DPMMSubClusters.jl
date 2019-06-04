__precompile__()

module DPMMSubClusters


using Distributed
using DistributedArrays
using StatsBase
using Distributions
using SpecialFunctions
using CatViews
using LinearAlgebra
using Random
using JLD2

include("ds.jl")
include("distributions/niw.jl")
include("distributions/mv_gaussian.jl")
include("distributions/multinomial_dist.jl")
include("distributions/multinomial_prior.jl")
include("utils.jl")
include("shared_actions.jl")
include("local_clusters_actions.jl")
include("global_params.jl")
include("dp-parallel-sampling.jl")
include("data_generators.jl")

export generate_gaussian_data, generate_mnmm_data, dp_parallel_sampling, dp_parallel, run_model_from_checkpoint, save_model, calculate_posterior


end # module
