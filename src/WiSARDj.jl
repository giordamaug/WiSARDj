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

include("RAM.jl")
using RAMj: WRam, getEntry, updEntry
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
    mypowers::Array{Int128}
    retina_size::Int
    n_rams::Int
    wiznet::Array{Dict{Int128,Float64}}
    classes::Array{Any}
    n_classes::Int
    WiSARDClassifier(;n_bits=8,n_tics=256,random_state=0,mapping="random",
                        code="t",bleaching=true,default_bleaching=1,confidence_bleaching=0.01,debug=false) =
        new(n_bits, n_tics, random_state, mapping, code, bleaching, default_bleaching, confidence_bleaching, debug)
end

get_classes(dt::WiSARDClassifier) = dt.classes
@declare_hyperparameters(WiSARDClassifier,
                         [:n_bits, :n_tics, :mapping, :code, :bleaching, :default_bleaching, :confidence_bleaching])

# Binarize input (thermomer encoding) terand generates address tuple for Ram access
function _mk_tuple(dt::WiSARDClassifier, data)
    addresses = zeros(dt.n_rams)
    for i in 1:dt.n_rams
        for j in 1:dt.n_bits
            x = data[((i * dt.n_bits) + j) % dt.retina_size]
            index = ÷(x,dt.n_tics)
            value = ÷((data[index] -  dt.offstets[index]) * dt.n_tics, self.ranges[index])
			if x % dt.n_tics < value
                addresses[i] += mypowers[dt.n_tics -1 - j]
            end
		end
	end
    addresses
end

function _train(dt::WiSARDClassifier, X, y)
    """ Learning """
    addresses = _mk_tuple(dt,X)
    for j in 1:dt.n_bits
        dt.wiznet[y][i].updEntry(dt.wiznet[y][i], addresses[i], 1.0)
    end
end

function _test(dt::WiSARDClassifier, X)
    """ Testing """
    addresses = _mk_tuple(X)
    res = zeros(dt.n_classes, dt.n_rams)
    for y in 1:dt.n_classes
    	for i in 1:dt.n_rams
    		res[y,i] = getEntry(dt.wiznet[y][i], addresses[i]) > 0 ? 1.0 : 0.0  # make it better!
    	end
    end
    argmax(sum(x,dims=2))[1]
end


function fit!(dt::WiSARDClassifier, X, y)
    n_samples, n_features = size(X)
    dt.retina_size = dt.n_tics * n_features
    dt.n_rams = dt.retina_size % dt.n_bits == 0 ? ÷(dt.retina_size,dt.n_bits) : ÷(dt.retina_size,dt.n_bits + 1)
    dt.classes = unique(y)
    dt.n_classes = size(dt.classes, 1)
    dt.wiznet = [[WRam() for _ in 1:dt.n_rams] for _ in 1:dt.n_classes]
    dt.mypowers = fill(2,128).^[i for i in range(1,128)] # it canbe better!
    for data in eachrow(X)
    	_train(data, y[i])
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
