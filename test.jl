
include("src/WiSARDj.jl")
using .WiSARDj: WiSARDClassifier, fit!, predict

using CSV, DataFrames, MLJ, MLBase

df = CSV.read("/Users/maurizio/WiSARDpy/datasets/iris.csv", DataFrames.DataFrame)


X = Matrix(select(df, Not([:species])))
y = vec(Matrix(select(df, [:species])))
train, test = partition(eachindex(y), 0.8, shuffle=true)

model = WiSARDClassifier()
WiSARDj.fit!(model, X[train,:], y[train,:])
y_pred = WiSARDj.predict(model, X[test,:])
accuracy = sum(y_pred .== y[test,:]) / length(y[test,:])
println("accuracy: $accuracy")
print(y_pred)
#show(stdout::IO, model)