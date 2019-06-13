using Test
using Distributed
using LinearAlgebra
addprocs(2)
@everywhere using Random
using DPMMSubClusters

include("multinomial_tests.jl")
include("niw_tests.jl")
include("unitests.jl")
include("module_tests.jl")
