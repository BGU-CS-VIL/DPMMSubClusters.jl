function create_data_for_test()
    data = zeros(Float32,2,1000)
    data[:,1:250] .= [-1,-1]
    data[:,251:500] .= [-1,1]
    data[:,501:750] .= [1,-1]
    data[:,751:1000] .= [1,1]
    return data
end

# @testset "Testing Module (Determinstic)" begin
#     data = create_data_for_test()
#     labels,clusters,weights = fit(data,100.0, iters = 200, seed = 12345, burnout = 5)
#     @test all(data[:,1:250] .== data[:,1])
#     @test all(data[:,251:500] .== data[:,251])
#     @test all(data[:,501:750] .== data[:,501])
#     @test all(data[:,751:1000] .== data[:,751])
#     @test length(clusters) == 4
#     println(weights)
#     @test all(weights .>= 0.15)
#     labels_histogram = get_labels_histogram(labels)
#     for (k,v) in labels_histogram
#         @test v == 250
#     end
# end



# @testset "Testing Module (Random mess)" begin
#     @everywhere Random.seed!(12345)
#     x,labels,clusters = generate_gaussian_data(10^6,3,10,100.0)

#     hyper_params = DPMMSubClusters.niw_hyperparams(Float32(1.0),
#                zeros(Float32,3),
#                Float32(5),
#                Matrix{Float32}(I, 3, 3)*1)

#     dp = dp_parallel(x,hyper_params,Float32(1000000000000000000000.0), 100,1,nothing,true,false,15,labels)
#     println("Full: ",mutualinfo(Int.(labels),dp[1].group.labels,normed=true))
#     @test length(dp[1].group.local_clusters) > 1
# end


# @testset "Streaming Data" begin
#     @everywhere Random.seed!(12345)
#     x,labels,clusters = generate_gaussian_data(10^6,3,10,100.0)
#     x1 = x[:,1:5:end]
#     x2 = x[:,2:5:end]
#     x3 = x[:,3:5:end]
#     x4 = x[:,4:5:end]
#     x5 = x[:,5:5:end]
#     labels1 = labels[1:5:end]
#     labels2 = labels[2:5:end]
#     labels3 = labels[3:5:end]
#     labels4 = labels[4:5:end]
#     labels5 = labels[5:5:end]

#     hyper_params = DPMMSubClusters.niw_hyperparams(Float32(1.0),
#                zeros(Float32,3),
#                Float32(5),
#                Matrix{Float32}(I, 3, 3)*1)
#     dp = dp_parallel_streaming(x1,hyper_params,Float32(1000000000000000000000.0), 20,1,nothing,true,false,15,labels1)
#     labels = dp.group.labels
#     println("Part 1: ",mutualinfo(Int.(labels1),labels,normed=true))
#     run_model_streaming(dp,20,2,x2)
#     labels = dp.group.labels
#     println("Part 2: ",mutualinfo(Int.(labels2),labels,normed=true))
#     run_model_streaming(dp,20,4,x3)
#     labels = dp.group.labels
#     println("Part 3: ",mutualinfo(Int.(labels3),labels,normed=true))
#     run_model_streaming(dp,20,6,x4)
#     labels = dp.group.labels
#     println("Part 4: ",mutualinfo(Int.(labels4),labels,normed=true))
#     run_model_streaming(dp,20,8,x5)
#     labels = dp.group.labels
#     println("Part 5: ",mutualinfo(Int.(labels5),labels,normed=true))
#     @test length(dp.group.local_clusters) > 1
# end



@testset "Streaming Data 2" begin
    @everywhere Random.seed!(12345)
    x,labels,clusters = generate_gaussian_data(10^6,3,10,100.0)
    parts = 10000
    xs = [x[:,i:parts:end] for i=1:parts]
    labelss = [labels[i:parts:end] for i=1:parts]
    hyper_params = DPMMSubClusters.niw_hyperparams(Float32(1.0),
               zeros(Float32,3),
               Float32(5),
               Matrix{Float32}(I, 3, 3)*1)
    dp = dp_parallel_streaming(xs[1],hyper_params,Float32(1000000000000000000000.0), 1,1,nothing,true,false,15,labelss[1],0.0001)
    labels = dp.group.labels
    avg_nmi = mutualinfo(Int.(labelss[1]),labels,normed=true)
    for i=2:parts
        run_model_streaming(dp,1,i*0.5,xs[i])
        labels = dp.group.labels
        avg_nmi += mutualinfo(Int.(labelss[i]),labels,normed=true)
    end
    println("NMI: ",avg_nmi/parts)
    @test length(dp.group.local_clusters) > 1
end


# @testset "Multinomial Module And save load" begin
#     @everywhere Random.seed!(12345)
#     x,labels,clusters = generate_mnmm_data(10^3,100,20,50)
#     @test size(x,1) == 100
#     @test size(x,2) == 10^3
#     npzwrite("save_load_test/mnm_data.npy",x')

#     dp = dp_parallel("save_load_test/multinomial_params.jl")
#     @test length(dp[1].group.local_clusters) > 1
#     dp = run_model_from_checkpoint("save_load_test/checkpoint_20.jld2")
#     @test length(dp[1].group.local_clusters) > 1
# end
