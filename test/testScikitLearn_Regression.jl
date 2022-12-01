
include("../src/WiSARDj.jl")
using .WiSARDj.SciLearnInterface: WiSARDRegressor

using CSV, DataFrames, MLJ, MLBase, MLJBase
using ScikitLearnBase
using Metrics

df = CSV.read("./datasets/boston.csv", DataFrames.DataFrame)

with_cv = true
model = WiSARDRegressor(n_bits=64, n_tics=1024, debug=true)

X = Matrix(DataFrames.select(df, Not([:medv])))
y = vec(Matrix(DataFrames.select(df, [:medv])))
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=1)
ScikitLearnBase.fit!(model, X[train,:], y[train,:])
ŷ = ScikitLearnBase.predict(model, X[test,:])
y_targets = vec(y[test,:])
MAE = Metrics.mae(vec(ŷ), y_targets)
MSE = Metrics.mse(vec(ŷ), y_targets)
R2 = Metrics.r2_score(vec(ŷ), y_targets)
println("Mean absolute error: $MAE")
println("Mean Squared Error: $MSE")
println("Coefficient of Determination: $R2")
using Plots
plot(hcat(y_targets, vec(ŷ)), label=["target" "preds"])