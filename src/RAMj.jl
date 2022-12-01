########## WiSARD Classificator RAM Types ##########
module RAMj

mutable struct WRam                 
    wentry::Dict{Int64,Float64}
    WRam() = new(Dict())
end

function getEntry(dt::WRam, key::Int64)
        if key in keys(dt.wentry)
            return dt.wentry[key]
        else
            return 0.0
        end
end
function updEntry(dt::WRam, key::Int64, value::Float64)
        if key in keys(dt.wentry)
            dt.wentry[key] += value
        else
            dt.wentry[key] = value
        end
end

########## WiSARD Regressor RAM Types ##########

mutable struct RRam
    wentry::Dict{Int64,Tuple{Float64, Float64}}
    RRam() = new(Dict())
end

function getEntry(dt::RRam, key::Int64)
        if key in keys(dt.wentry)
            return dt.wentry[key]
        else
            return (0.0, 0.0)
        end
end
function updEntry(dt::RRam, key::Int64, value::Float64)
        if key in keys(dt.wentry)
            dt.wentry[key] = (dt.wentry[key][1] + 1.0, dt.wentry[key][2] + value)
        else
            dt.wentry[key] = (1.0, value)
        end
end

########## RAM APIs ####################
export WRam, getEntry, updEntry
export RRam, getEntry, updEntry

end
