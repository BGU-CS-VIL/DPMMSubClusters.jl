# dpmm_subclusters.jl
This repository is a *Julia* package which holds the code for our paper **Distributed MCMC Inference in Dirichlet Process Mixture Models Using Julia**, which was presented at CCGrid2019 Hyper Computing Maching Learning workshop (HPML).

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
`(v1.0) pkg> add https://github.com/dinarior/dpmm_subclusters.jl`

## Usage

This package is aimed for distributed parallel computing, note that for it to work properly you must use it with atleast one worker process. More workers are encouraged for increased performance.

The package currently contains priors for handling *Multinomial* or *Gaussian* mixture models.  Adding new priors is easy, and the process is described [here](#new_prior).

While being very verstile in the setting and configuration, there are 2 modes which you can work with, either the *Basic*, which will use mostly predefined configuration, and will take the data as an argument, or *Advanced* use, which allows more configuration, loading data from file, and saving the model, or running from a saved checkpoint.

### Basic
In order to run in the basic mode, use the function `dp_parallel(data,hyper_params, iterations, inital_cluster,seed)`

`data` should be of the shape `DxN` , `hyper_params` are one of the available hyper parameters.
The `initial_cluster` and `seed` are optional.

Example:
```
x,labels,clusters = generate_gaussian_data(10^5,2,6)

hyper_params = dpmm_subclusters.niw_hyperparams(1.0,
           zeros(2),
           4,
           Matrix{Float64}(I, 2, 2)*1)

dp = dp_parallel(x,hyper_params, 100, 1)
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
The `labels` hold the final assignments of the points, `local_clusters` contains the infered clusters parameters and `weights` contain the cluster weights.

Full example, including the plots, is supplied [here](https://github.com/dinarior/dpmm_subclusters.jl/blob/master/examples/gaussian_2d.jl).

### Advanced
In this mode you are required to supply a params file, example for one is the file `global_params.jl`.
It includes all the configurable params. Running it is as simple as:
`dp_parallel(params_file)`

Note that for data loading the package use `NPZ` , which utilize python *numpy* files. Thus the data files must be *pythonic*, and be of the shape `NxD`.


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
