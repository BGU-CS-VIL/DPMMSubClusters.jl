# Usage

The package offer two type of modes for running it, *Basic* and *Advanced*.

## Basic

This mode will run with mostly predefined settings, saving checkpoints is not recommended in this mode.
The model is run by using the `fit` method, with a minimal requirements of the `Data` and `α` concentration parameter. when `prior` is not supplied, it will automatically use a weak `NIW` prior.
```@meta
CurrentModule = DPMMSubClusters
```

```@docs
fit(all_data::AbstractArray{Float32,2},local_hyper_params::distribution_hyper_params,α_param::Float32;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false, burnout = 20, gt = nothing)
fit(all_data::AbstractArray{Float32,2},α_param::Float32;
        iters::Int64 = 100, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false,burnout = 20, gt = nothing)
```

## Advanced

This mode allows greater flexibility, and required a `Parameters` file (see below).
It is run by the function `dp_parallel`.
```@docs
dp_parallel(model_params::String; verbose = true, save_model = true,burnout = 5, gt = nothing)
```

In addition, you may restart a previously saved checkpoint:
```@docs
run_model_from_checkpoint(filename)
```

Note that that data is read from a `npy` file, and unlike the previous `fit` function, should be of `Samples X Dimensions`.

### Parameter File

For running the advanced mode you need to specify a parameters file, it is a `Julia` file, of the following struct:

```julia
#Data Loading specifics
data_path = "/path/to/data/"
data_prefix = "data_prefix"  #If the data file name is bob.npy, this should be 'bob'


#Model Parameters
iterations = 100
hard_clustering = false  #Soft or hard assignments
initial_clusters = 1
argmax_sample_stop = 0 #Change to hard assignment from soft at iterations - argmax_sample_stop
split_stop  = 0 #Stop split/merge moves at  iterations - split_stop

random_seed = nothing #When nothing, a random seed will be used.

max_split_iter = 20
burnout_period = 20

#Model hyperparams
α = 10.0 #Concetration Parameter
hyper_params = DPMMSubClusters.niw_hyperparams(1.0,
    zeros(Float32,2),
    5,
    Matrix{Float32}(I, 2, 2)*1.0)



#Saving specifics:
enable_saving = true
model_save_interval = 1000
save_path = "/path/to/save/dir/"
overwrite_prec = false
save_file_prefix = "checkpoint_"
```
