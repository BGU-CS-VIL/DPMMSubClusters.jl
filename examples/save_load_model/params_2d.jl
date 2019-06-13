random_seed = nothing

data_path = ""
data_prefix = "2d1ksample"


#Model Parameters
iterations = 99
hard_clustering = false
argmax_sample_stop = 10 #Change to hard assignment from soft at iterations - argmax_sample_stop
split_stop  = 10 #Stop split/merge moves at  iterations - split_stop

total_dim = 2

α = 100000.0


initial_clusters = 1


use_dict_for_global = false


hyper_params = niw_hyperparams(1.0,
    [0,0],
    5.0,
    Matrix{Float64}(I, 2, 2)*1.0)

#Saving specifics:
enable_saving = true
model_save_interval = 50
save_path = ""
overwrite_prec = false
save_file_prefix = "checkpoint_"
