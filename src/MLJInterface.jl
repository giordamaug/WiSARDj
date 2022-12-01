module MLJInterface

# Write your package code here.
import MLJBase

using Distributed
using ProgressMeter: @showprogress
using Random
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
using .RAMj: WRam, RRam, getEntry, updEntry

########## Models ##########

mutable struct WiSARDClassifier <: MLJBase.Probabilistic
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
                        code="t",bleaching=false,default_bleaching=1,confidence_bleaching=0.01,debug=false) =
        new(n_bits, n_tics, random_state, mapping, code, bleaching, default_bleaching, confidence_bleaching, debug)
end

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

function _calc_confidence(results::Matrix{Float64})
    # get max value
    max_value = findmax(results)[1]
    if (max_value == 0)  # if max is null confidence will be 0
        return 0
    end
    # if there are two max values, confidence will be 0
    position = results[results.==max_value]
    if size(position, 1)>1
        return 0
    end 
    # get second max value
    #second_max = findmax(results[results. < max_value])[1]
    if size(results[results.< max_value])[1] > 0
        second_max = findmax(results[results.< max_value])[1]
    end
    # calculating new confidence value
    return 1 - second_max / max_value
end

function _train(dt::WiSARDClassifier, data::Vector{Float64}, y)
    """ Learning """
    addresses = _mk_tuple(dt,data)
    for i in 1:dt.n_rams
        updEntry(dt.wiznet[y][i], addresses[i], 1.0)
    end
end

function _test_nobleaching(dt::WiSARDClassifier, data::Vector{Float64})
    """ Testing """
    addresses = _mk_tuple(dt,data)
    res = zeros(dt.n_classes, dt.n_rams)
    for y in 1:dt.n_classes
    	for i in 1:dt.n_rams
    		res[y,i] = getEntry(dt.wiznet[dt.classes[y]][i], addresses[i]) > 0 ? 1.0 : 0.0  # make it better!
    	end
    end
    return sum(res,dims=2) 
end

function _response(dt::WiSARDClassifier, data::Vector{Float64})
    """ Testing """
    addresses = _mk_tuple(dt,data)
    res = zeros(dt.n_classes, dt.n_rams)
    for y in 1:dt.n_classes
    	for i in 1:dt.n_rams
    		res[y,i] = getEntry(dt.wiznet[dt.classes[y]][i], addresses[i])     # make it better!
    	end
    end
    return res 
end

function _test_bleaching(dt::WiSARDClassifier, data::Vector{Float64})
    """ Testing """
    b = dt.default_bleaching
    confidence = 0.0
    res_disc = _response(dt, data)
    result_partial = Any
    while confidence < dt.confidence_bleaching
        result_partial = sum(replace(x->x>=b ? 1.0 : 0.0, res_disc), dims=2)
        confidence = _calc_confidence(result_partial)
        b += 1
        if (sum(result_partial) == 0)
            result_partial = sum(replace(x->x>=1 ? 1.0 : 0.0, res_disc), dims=2)
            break
        end
    end
    result_sum = sum(result_partial, dims=1)[1]
    if result_sum==0.0
        result = sum(res_disc, dims=2)./ dt.nrams
    else
        result = sum(result_partial, dims=2)./ result_sum
    end
    return result
end

function MLJBase.fit(dt::WiSARDClassifier, verbosity, X, y)
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
    if dt.mapping  == "random"
        shuffle!(dt._mapping)
    end
    dt.offsets = findmin(X, dims=1)[1]
    dt.ranges = findmax(X, dims=1)[1] - dt.offsets
    dt.offsets = vec(dt.offsets)
    dt.ranges = vec(dt.ranges)
    dt.ranges[dt.ranges .== 0] .= 1
    @showprogress 1 "Testing..." for i in 1:n_samples
        _train(dt, X[i, :], y[i])
    end
end

function MLJBase.predict(dt::WiSARDClassifier, fitresult, X)
	n_samples, _ = size(X)

	y_pred = Vector{Any}(undef, n_samples)
    _test = dt.bleaching ? _test_bleaching : _test_nobleaching
	@showprogress 1 "Testing..."  for i in 1:n_samples
        y_pred[i] = dt.classes[argmax(_test(dt, X[i, :]))[1]]
    end
    return y_pred
end

function predict_proba(dt::WiSARDClassifier, X)  # there's no predict_proba in MLJ
	n_samples, _ = size(X)

	y_pred = Vector{Any}(undef, n_samples)
	@showprogress 1 "Testing..."  for i in 1:n_samples
        y_pred[i] = _test(dt, X[i, :])
    end
    return y_pred
end

function show(io::IO, dt::WiSARDClassifier)
        println(io, "WiSARDClassifier")
        println(io, "n_bits:                $(dt.n_bits)")
        println(io, "n_tics:                $(dt.n_tics)")
        println(io, "n_rams:                $(dt.n_rams)")
        println(io, "n_retina:              $(dt.retina_size)")
end
    
################################################################################
# Regressor

"""
    WiSARDRegressor(;  n_bits::Int=8, 
                        n_tics::Int=256, 
                        random_state::Int=0, 
                        mapping::string='random', 
                        code::string='t', 
                        debug::Bool=False   
                           )

Hyperparameters:

- `n_bits`: 
- `n_tics`: 
- `random_state`: 
- `mapping`: 
- `code`: 
- `debug`: 

Implements `fit!`, `predict`
"""

########## Types ##########

mutable struct WiSARDRegressor <: MLJBase.Probabilistic
    n_bits::Int
    n_tics::Int
    random_state::Int
    mapping::String
    code::String
    debug::Bool
    mypowers::Array{Int}
    retina_size::Int
    n_rams::Int
    _rams #::Array{Dict{Int64,Tuple{Float64, Float64}}}
    _mapping::Array{Int}
    ranges::Array{Float64}
    offsets::Array{Float64}
    WiSARDRegressor(;n_bits=8,n_tics=256,random_state=0,mapping="random",code="t",debug=false) =
        new(n_bits, n_tics, random_state, mapping, code, debug)
end

# Binarize input (thermomer encoding) terand generates address tuple for Ram access
function _mk_tuple(dt::WiSARDRegressor, data)
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

function _train(dt::WiSARDRegressor, data::Vector{Float64}, y)
    """ Learning """
    addresses = _mk_tuple(dt,data)
    for i in 1:dt.n_rams
        updEntry(dt._rams[i], addresses[i], y)
    end
end

function _test(dt::WiSARDRegressor, data::Vector{Float64})
    """ Testing """
    addresses = _mk_tuple(dt,data)
    res = reduce((x, y) -> x .+ y, [ getEntry(dt._rams[i], addresses[i]) for i in 1:dt.n_rams])
    result = res[1] != 0 ? res[2] / res[1] : 0.0
end


function MLJBase.fit(dt::WiSARDRegressor, verbosity, X, y)
    n_samples, n_features = size(X)
    dt.retina_size = dt.n_tics * n_features
    dt.n_rams = dt.retina_size % dt.n_bits == 0 ? ÷(dt.retina_size,dt.n_bits) : ÷(dt.retina_size,dt.n_bits + 1)
    dt._rams = [RRam() for _ in 1:dt.n_rams]
    dt.mypowers = fill(2,128).^[i-1 for i in 1:128]     # it can be better!
    dt._mapping = [i for i in 1:dt.retina_size]
    if dt.mapping  == "random"
        shuffle!(dt._mapping)
    end
    dt.offsets = findmin(X, dims=1)[1]
    dt.ranges = findmax(X, dims=1)[1] - dt.offsets
    dt.offsets = vec(dt.offsets)
    dt.ranges = vec(dt.ranges)
    dt.ranges[dt.ranges .== 0] .= 1
    @showprogress 1 "Training..." for i in 1:n_samples
        _train(dt, X[i, :], y[i])
    end
end

function MLJBase.predict(dt::WiSARDRegressor, fitresult, X)
    n_samples, _ = size(X)

    y_pred = Vector{Float64}(undef, n_samples)
    @showprogress 1 "Testing..."  for i in 1:n_samples
        y_pred[i] = _test(dt, X[i, :])
    end
    return y_pred
end


function show(io::IO, dt::WiSARDRegressor)
        println(io, "WiSARDRegressor")
        println(io, "n_bits:                $(dt.n_bits)")
        println(io, "n_tics:                $(dt.n_tics)")
        println(io, "n_rams:                $(dt.n_rams)")
        println(io, "n_retina:              $(dt.retina_size)")
end

################# MLJ API #################

export WiSARDCLassifier,
       # Should we export these functions? They have a conflict with
       # DataFrames/RDataset over fit!, and users can always
       # `using ScikitLearnBase`.
       predict, predict_proba, fit!, get_classes
export WiSARDREgressor,
       # Should we export these functions? They have a conflict with
       # DataFrames/RDataset over fit!, and users can always
       # `using ScikitLearnBase`.
       predict, fit!
end

