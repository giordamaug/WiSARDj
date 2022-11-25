
include("src/WiSARDj.jl")
using .WiSARDj: WiSARDClassifier, fit!, predict

using CSV, DataFrames, MLJ, MLBase, MLJBase

df = CSV.read("/Users/maurizio/WiSARDpy/datasets/iris.csv", DataFrames.DataFrame)


X = Matrix(DataFrames.select(df, Not([:species])))
y = vec(Matrix(DataFrames.select(df, [:species])))
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=1)

model = WiSARDClassifier()
WiSARDj.fit!(model, X[train,:], y[train,:])
ŷ = WiSARDj.predict(model, X[test,:])
accuracy = sum(ŷ .== y[test,:]) / length(y[test,:])
println("accuracy: $accuracy")
print(MLJBase.ConfusionMatrix()(ŷ, vec(y[test,:])))