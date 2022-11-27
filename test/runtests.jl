include("./src/WiSARDj.jl")
using Test

@testset "WiSARDj.jl" begin
    @test WiSARDj.WiSARDClassifier() == "Hello WiSARDClassifier!"
    # Write your tests here.
    using MLJ
    using RDatasets
    using .WiSARDj.MLJInterface: WiSARDClassifier
    iris = dataset("datasets", "iris"); # a DataFrame
    X = iris[:, 1:4];
    y = iris[:, 5];
    train, test = partition(eachindex(y), 0.7, shuffle=true);
    model = WiSARDClassifier(n_bits=8, n_tics=256, debug=true)
    fit!(model, X[train,:])
    yhat = predict(model, X[test,:]);
    misclassification_rate(yhat, y[test]);
end
