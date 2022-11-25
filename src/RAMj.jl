########## WiSARD Types ##########
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

export WRam, getEntry, updEntry

end
