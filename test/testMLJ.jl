
include("../src/WiSARDj.jl")
using .WiSARDj.MLJInterface: WiSARDClassifier

using CSV, DataFrames, MLJ, MLBase, MLJBase

df = CSV.read("/Users/maurizio/WiSARDpy/datasets/iris.csv", DataFrames.DataFrame)
#df = CSV.read("/Users/maurizio/WiSARDpy/datasets/biomat_clf.csv", DataFrames.DataFrame)

with_cv = true
model = WiSARDClassifier(n_bits=4, n_tics=16, mapping="linear", bleaching=true)

X = Matrix(DataFrames.select(df, Not([:species])))
y = vec(Matrix(DataFrames.select(df, [:species])))
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=1)
MLJ.fit(model, Any, X[train,:], y[train,:])
ŷ = MLJ.predict(model, Any, X[test,:])
y_targets = y[test,:]
accuracy = sum(ŷ .== y_targets) / length(y_targets)
println("accuracy: $accuracy")
MLJBase.ConfusionMatrix()(ŷ, coerce(vec(y_targets), OrderedFactor))  # I don't know why we need to coerce! 
