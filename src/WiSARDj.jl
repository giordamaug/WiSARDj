module WiSARDj

# Write your package code here.
import ScikitLearnBase: BaseClassifier, BaseRegressor, predict, predict_proba,
                        fit!, get_classes, @declare_hyperparameters

################################################################################
# Classifier

"""
    WiSARDClassifier(;  n_bits::Int=8, 
                        n_tics::Int=256, 
                        random_state::Int=0, 
                        mapping::string='random', 
                        code::string='t', 
                        bleaching::Bool=True,
                        default_bleaching::Int=1,
                        confidence_bleaching::Float=0.01, 
                        debug::Bool=False   
                           )

Hyperparameters:

- `n_bits`: 
- `n_tics`: 
- `random_state`: 
- `mapping`: 
- `code`: 
- `bleaching`: 
- `default_bleaching`: 
- `confidence_bleaching`: 
- `debug`: 

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""

########## Types ##########

include("Ram.jl")

########## Models ##########

mutable struct WiSARDClassifier <: BaseClassifier
    n_bits::Int
    n_tics::Int
    random_state::Int
    mapping::String
    code::String
    bleaching::Bool
    default_bleaching::Int
    confidence_bleaching::Float64
    debug::Bool
    mypowers::Int
    retina_size::Int
    n_rams::Int
    WiSARDClassifier(;n_bits=8,n_tics=256,random_state=0,mapping="random",
                        code="t",bleaching=true,default_bleaching=1,confidence_bleaching=0.01,debug=false) =
        new(n_bits, n_tics, random_state, mapping, code, bleaching, default_bleaching, confidence_bleaching, debug)
end

get_classes(dt::WiSARDClassifier) = dt.classes
@declare_hyperparameters(WiSARDClassifier,
                         [:n_bits, :n_tics, :mapping, :code, :bleaching, :default_bleaching, :confidence_bleaching])

function fit!(dt::WiSARDClassifier, X, y)
    n_samples, n_features = size(X)
    dt.retina_size = dt.n_tics * n_features
    dt.n_rams = dt.retina_size % dt.n_bits == 0 ? ÷(dt.retina_size,dt.n_bits) : ÷(dt.retina_size,dt.n_bits + 1)
    for data in eachrow(X)
        println(data)
    end
end

function predict(dt::WiSARDClassifier, X)
end

#predict_proba(dt::WiSARDClassifier, X) =
#   apply_tree_proba(dt.root, X, dt.classes)

function show(io::IO, dt::WiSARDClassifier)
    println(io, "WiSARDClassifier")
    println(io, "n_bits:                $(dt.n_bits)")
    println(io, "n_tics:                $(dt.n_tics)")
    println(io, "n_rams:                $(dt.n_rams)")
    println(io, "n_retina:              $(dt.retina_size)")
end


# ScikitLearn API
export WiSARDCLassifier,
       # Should we export these functions? They have a conflict with
       # DataFrames/RDataset over fit!, and users can always
       # `using ScikitLearnBase`.
       predict, predict_proba, fit!, get_classes

end
