using Test
using Distributed
addprocs(2)
@everywhere using Random
@everywhere using DPMMSubClusters

include("multinomial_tests.jl")
include("niw_tests.jl")
include("unitests.jl")
include("module_tests.jl")
