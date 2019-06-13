function create_data_for_test()
    data = zeros(2,1000)
    data[:,1:250] .= [-1,-1]
    data[:,251:500] .= [-1,1]
    data[:,501:750] .= [1,-1]
    data[:,751:1000] .= [1,1]
    return data
end

@testset "Testing Module" begin
    data = create_data_for_test()
    labels,clusters,weights = fit(data,100.0, iters = 30, seed = 1234)
    @test all(data[:,1:250] .== data[:,1])
    @test all(data[:,251:500] .== data[:,251])
    @test all(data[:,501:750] .== data[:,501])
    @test all(data[:,751:1000] .== data[:,751])
    @test length(clusters) == 4
    @test all(weights .>= 0.2)
end
