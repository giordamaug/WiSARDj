
include("src/WiSARDj.jl")
using .WiSARDj: WiSARDClassifier, fit!, predict

using CSV, DataFrames, MLJ

df = CSV.read("/Users/maurizio/WiSARDpy/datasets/iris.csv", DataFrames.DataFrame)


X = Matrix(select(df, Not([:species])))
y = vec(Matrix(select(df, [:species])))
train, test = partition(eachindex(y), 0.8, shuffle=true)

model = WiSARDClassifier()
fit!(model, X[train,:], y[train,:])

show(stdout::IO, model)