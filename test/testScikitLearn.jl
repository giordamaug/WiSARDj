
include("../src/WiSARDj.jl")
using .WiSARDj.SciLearnInterface: WiSARDClassifier, fit!, predict

using CSV, DataFrames, MLJ, MLBase, MLJBase
using ScikitLearn.CrossValidation: cross_val_predict
using ScikitLearnBase

#df = CSV.read("/Users/maurizio/WiSARDpy/datasets/biomat_clf.csv", DataFrames.DataFrame)
df = CSV.read("/Users/maurizio/WiSARDpy/datasets/iris.csv", DataFrames.DataFrame)

with_cv = false
model = WiSARDClassifier(n_bits=8, n_tics=256, bleaching=true, debug=true)

X = Matrix(DataFrames.select(df, Not([:label])))
y = vec(Matrix(DataFrames.select(df, [:label])))
if with_cv
    ŷ = cross_val_predict(model, X, y; cv=5)
    y_targets = y
else
    train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=1)
    fit!(model, X[train,:], y[train,:])
    ŷ = predict(model, X[test,:])
    y_targets = y[test,:]
end
accuracy = sum(ŷ .== y_targets) / length(y_targets)
println("accuracy: $accuracy")
MLJBase.ConfusionMatrix()(ŷ, coerce(vec(y_targets), OrderedFactor))  # I don't know why we need coerce!end
