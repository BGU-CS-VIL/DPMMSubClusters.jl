[![Build Status](https://api.travis-ci.com/dinarior/DPMMSubClusters.jl.svg?branch=master)](https://travis-ci.com/dinarior/DPMMSubClusters.jl)
[![Coverage Status](https://coveralls.io/repos/github/dinarior/DPMMSubClusters.jl/badge.svg?branch=master)](https://coveralls.io/github/dinarior/DPMMSubClusters.jl?branch=master)
[![codecov](https://codecov.io/gh/dinarior/DPMMSubClusters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dinarior/DPMMSubClusters.jl)


# DPMMSubClusters.jl
This repository is a *Julia* package which holds the code for our paper **Distributed MCMC Inference in Dirichlet Process Mixture Models Using Julia**, which was presented at CCGrid2019 Hyper Computing Maching Learning workshop (HPML) and is available [here](https://www.cs.bgu.ac.il/~dinari/papers/dpmm_hpml2019.pdf).<br>
<p align="center">
<img src="https://www.cs.bgu.ac.il/~dinari/images/clusters_low_slow.gif" alt="DPGMM SubClusters 2d example">
</p>

## Requirements
This package was developed and tested on *Julia 1.0.3*, prior versions will not work.
The following dependencies are required:
- CatViews
- Distributed
- DistributedArrays
- Distributions
- JLD2
- LinearAlgebra
- NPZ
- Random
- SpecialFunctions
- StatsBase


## Installation

Use Julia's package manager:
`(v1.0) pkg> add DPMMSubClusters`

## Usage

This package is aimed for distributed parallel computing, note that for it to work properly you must use it with atleast one worker process. More workers, distributed across different machines, are encouraged for increased performance.

The package currently contains priors for handling *Multinomial* or *Gaussian* mixture models.

While being very verstile in the setting and configuration, there are 2 modes which you can work with, either the *Basic*, which will use mostly predefined configuration, and will take the data as an argument, or *Advanced* use, which allows more configuration, loading data from file, and saving the model, or running from a saved checkpoint.

### Basic
In order to run in the basic mode, use the function:
```
labels, clusters, weights = fit(all_data::AbstractArray{Float64,2},local_hyper_params::distribution_hyper_params,α_param::Float64;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false)
```

Or, if opting for the default Gaussian weak prior:
```
labels, clusters, weights = fit(all_data::AbstractArray{Float64,2},α_param::Float64;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false)
```

`data` should be of the shape `DxN` , `hyper_params` are one of the available hyper parameters.
While saving the model with this mode is allowed, it is not encouraged. for that there exists the advanced mode.

Examples:
[2d Gaussian with plotting](https://nbviewer.jupyter.org/github/dinarior/DPMMSubClusters.jl/blob/master/examples/2d_gaussian/gaussian_2d.ipynb).
[Image Segmentation](https://nbviewer.jupyter.org/github/dinarior/DPMMSubClusters.jl/blob/master/examples/image_seg/dpgmm-superpixels.ipynb).


### Advanced
In this mode you are required to supply a params file, example for one is the file `global_params.jl`.
It includes all the configurable params. Running it is as simple as:
```
dp = dp_parallel(model_params::String; verbose = true, save_model = true)
```

The returned value `dp` is a data structure:
```
mutable struct dp_parallel_sampling
    model_hyperparams::model_hyper_params
    group::local_group
end
```
In which contains the `local_group`, another structure:
```
mutable struct local_group
    model_hyperparams::model_hyper_params
    points::AbstractArray{Float64,2}
    labels::AbstractArray{Int64,1}
    labels_subcluster::AbstractArray{Int64,1}
    local_clusters::Vector{local_cluster}
    weights::Vector{Float64}
end
```

Note that for data loading the package use `NPZ` , which utilize python *numpy* files. Thus the data files must be *pythonic*, and be of the shape `NxD`.

[Example of running from a params file, including saving and loading, with a multinomial prior](https://nbviewer.jupyter.org/github/dinarior/DPMMSubClusters.jl/blob/master/examples/save_load_model/save_load_example.ipynb).

## Additional Functions
Additional function exposed to the user include:

- `run_model_from_checkpoint(file_name)` : Used to restart a saved run, file_name must point to a valid checkpoint file created during a run of the model.  Note that the params files used for running the model initialy must still be available and in the same location, this is true for the data as well.
- `calculate_posterior(model)` : Calculate the posterior of a model, returned from `dp_parallel`.
- `generate_gaussian_data(N::Int64, D::Int64, K::Int64)`: Randomly generates gaussian data, `N` points, of dimension `D` from `K` clusters. return value is `points, labels, cluster_means, cluster_covariance`.
- `generate_mnmm_data(N::Int64, D::Int64, K::Int64, trials::Int64)`: Similar to above, just for multinomial data, the return value is `points, labels, clusters`

### Misc

For any questions: dinari@post.bgu.ac.il

Contributions, feature requests, suggestion etc.. are welcomed.

If you use this code for your work, please cite the following:

```
@inproceedings{Dinari:CCGrid:2019,
  title={Distributed {MCMC} Inference in {Dirichlet} Process Mixture Models Using {Julia}},
  author={Dinari, Or and Angel, Yu and Freifeld, Oren and Fisher III, John W},
  booktitle={International Symposium on Cluster, Cloud and Grid Computing (CCGRID) Workshop on High Performance Machine Learning Workshop},
  year={2019}
}
```
