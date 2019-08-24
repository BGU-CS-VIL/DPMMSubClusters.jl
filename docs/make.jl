using Documenter, DPMMSubClusters

push!(LOAD_PATH,"../src/")
makedocs(
    sitename = "DPMMSubClusters.jl",
    doctest  = false,
    pages    = [
        "index.md",
        "getting_started.md",
        "usage.md",
        "priors.md",
        "data_generation.md",
        "perf.md"
    ]
)


deploydocs(
    repo = "github.com/dinarior/DPMMSubClusters.jl.git",
)
