
include("../src/WiSARDj.jl")
using .WiSARDj.MLJInterface: WiSARDRegressor

using CSV, DataFrames, MLJ, MLBase, MLJBase
using Metrics

df = CSV.read("./datasets/boston.csv", DataFrames.DataFrame)

with_cv = true
model = WiSARDRegressor(n_bits=8, n_tics=256, debug=true)

X = Matrix(DataFrames.select(df, Not([:medv])))
y = vec(Matrix(DataFrames.select(df, [:medv])))
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=1)
MLJ.fit(model, Any, X[train,:], y[train,:])
ŷ = MLJ.predict(model, Any, X[test,:])
y_targets = y[test,:]
MAE = Metrics.mae(vec(ŷ), y_targets)
MSE = Metrics.mse(vec(ŷ), y_targets)
R2 = Metrics.r2_score(vec(ŷ), y_targets)
println("Mean absolute error: $MAE")
println("Mean Squared Error: $MSE")
println("Coefficient of Determination: $R2")