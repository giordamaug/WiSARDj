module WiSARDj

# Write your package code here.
import ScikitLearnBase: BaseClassifier, BaseRegressor, predict, predict_proba,
                        fit!, get_classes, @declare_hyperparameters
using Distributed
using ProgressMeter
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

include("RAMj.jl")
using .RAMj: WRam, getEntry, updEntry
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
    mypowers::Array{Int}
    retina_size::Int
    n_rams::Int
    wiznet #::Dict{Array{Dict{Int64,Float64}}}
    classes::Array{Any}
    n_classes::Int
    _mapping::Array{Int}
    ranges::Array{Float64}
    offsets::Array{Float64}
    WiSARDClassifier(;n_bits=8,n_tics=256,random_state=0,mapping="random",
                        code="t",bleaching=true,default_bleaching=1,confidence_bleaching=0.01,debug=false) =
        new(n_bits, n_tics, random_state, mapping, code, bleaching, default_bleaching, confidence_bleaching, debug)
end

get_classes(dt::WiSARDClassifier) = dt.classes
@declare_hyperparameters(WiSARDClassifier,
                         [:n_bits, :n_tics, :maptype, :code, :bleaching, :default_bleaching, :confidence_bleaching])

# Binarize input (thermomer encoding) terand generates address tuple for Ram access
function _mk_tuple(dt::WiSARDClassifier, data)
    addresses = zeros(Int, dt.n_rams)
    count = 0
    for i in 0:dt.n_rams-1
        for j in 0:dt.n_bits-1
            x = dt._mapping[(i * dt.n_bits + j) % (dt.retina_size)+1] - 1
            index = div(x,(dt.n_tics))
            value = div((data[index+1] -  dt.offsets[index+1]) * dt.n_tics, dt.ranges[index+1])
			if x % dt.n_tics < value
                addresses[i+1] += dt.mypowers[dt.n_bits - j]
            end
            count += 1
		end
	end
    return addresses
end

function _train(dt::WiSARDClassifier, data::Vector{Float64}, y)
    """ Learning """
    addresses = _mk_tuple(dt,data)
    for i in 1:dt.n_rams
        updEntry(dt.wiznet[y][i], addresses[i], 1.0)
    end
end

function _test(dt::WiSARDClassifier, data::Vector{Float64})
    """ Testing """
    addresses = _mk_tuple(dt,data)
    res = zeros(dt.n_classes, dt.n_rams)
    for y in 1:dt.n_classes
    	for i in 1:dt.n_rams
    		res[y,i] = getEntry(dt.wiznet[dt.classes[y]][i], addresses[i]) > 0 ? 1.0 : 0.0  # make it better!
    	end
    end
    return sum(res,dims=2) #argmax(sum(res,dims=2))[1]
end


function fit!(dt::WiSARDClassifier, X, y)
    n_samples, n_features = size(X)
    dt.retina_size = dt.n_tics * n_features
    dt.n_rams = dt.retina_size % dt.n_bits == 0 ? ÷(dt.retina_size,dt.n_bits) : ÷(dt.retina_size,dt.n_bits + 1)
    dt.classes = unique(y)
    dt.n_classes = size(dt.classes, 1)
    dt.wiznet = Dict()
    for c in dt.classes
    	dt.wiznet[c] = [WRam() for _ in 1:dt.n_rams]
    end
    dt.mypowers = fill(2,128).^[i-1 for i in 1:128]     # it can be better!
    dt._mapping = [i for i in 1:dt.retina_size]
    dt.offsets = findmin(X, dims=1)[1]
    dt.ranges = findmax(X, dims=1)[1] - dt.offsets
    dt.offsets = vec(dt.offsets)
    dt.ranges = vec(dt.ranges)
    dt.ranges[dt.ranges .== 0] .= 1
    @showprogress 1 "Training..."  for i in 1:n_samples
    	_train(dt, X[i, :], y[i])
    end
end

function predict(dt::WiSARDClassifier, X)
	n_samples, _ = size(X)

	y_pred = Vector{Any}(undef, n_samples)
	@showprogress 1 "Testing..."  for i in 1:n_samples
		response = _test(dt, X[i, :])
        y_pred[i] = dt.classes[argmax(response)]
    end
    return y_pred
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
