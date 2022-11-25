########## WiSARD Types ##########

struct WRam                 
    wentry::Dict
    function getEntry(key::Int128)
        wentry[key]
    end
    function updEntry(key::Int128, value::Real)
        wentry[key] += 1.0
    end
end

export WRam
