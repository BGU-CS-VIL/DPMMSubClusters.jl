function create_data_for_test()
    data = zeros(2,1000)
    data[:,1:250] .= [-1,-1]
    data[:,251:500] .= [-1,1]
    data[:,501:750] .= [1,-1]
    data[:,751:1000] .= [1,1]
    return data
end

@testset "Testing Module (Determinstic)" begin
    data = create_data_for_test()
    labels,clusters,weights = fit(data,100.0, iters = 30, seed = 1234)
    @test all(data[:,1:250] .== data[:,1])
    @test all(data[:,251:500] .== data[:,251])
    @test all(data[:,501:750] .== data[:,501])
    @test all(data[:,751:1000] .== data[:,751])
    @test length(clusters) == 4
    @test all(weights .>= 0.2)
    labels_histogram = get_labels_histogram(labels)
    for (k,v) in labels_histogram
        @test v == 250
    end
end



@testset "Testing Module (Random mess)" begin
    @everywhere Random.seed!(12345)
    x,labels,clusters = generate_gaussian_data(10^5,3,30,3.0)

    hyper_params = DPMMSubClusters.niw_hyperparams(1.0,
               zeros(3),
               5,
               Matrix{Float64}(I, 3, 3)*1)

    dp = dp_parallel(x,hyper_params,1000000000000000000000.0, 50, 4)
    @test length(dp.group.local_clusters) == 3
end

@testset "Multinomial Module And save load" begin
    @everywhere Random.seed!(12345)
    x,labels,clusters = generate_mnmm_data(10^3,100,20,50)
    @test size(x,1) == 100
    @test size(x,2) == 10^3
    npzwrite("save_load_test/mnm_data.npy",x')

    dp = dp_parallel("save_load_test/multinomial_params.jl")
    @test length(dp.group.local_clusters) == 13
    dp = run_model_from_checkpoint("save_load_test/checkpoint_20.jld2")
    @test length(dp.group.local_clusters) == 12
end
