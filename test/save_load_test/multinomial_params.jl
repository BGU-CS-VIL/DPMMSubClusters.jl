random_seed = nothing

data_path = "save_load_test/"
data_prefix = "mnm_data"


#Model Parameters
iterations = 39
hard_clustering = false
argmax_sample_stop = 10 #Change to hard assignment from soft at iterations - argmax_sample_stop
split_stop  = 10 #Stop split/merge moves at  iterations - split_stop

total_dim = 2

α = Float32(100000.0)


initial_clusters = 1


use_dict_for_global = false


hyper_params = multinomial_hyper(ones(Float32,100))

#Saving specifics:
enable_saving = true
model_save_interval = 20
save_path = "save_load_test/"
overwrite_prec = false
save_file_prefix = "checkpoint"
