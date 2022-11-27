include("src/WiSARDj.jl")
using .WiSARDj.MLJInterface: WiSARDClassifier

function mk_tuple(dt::WiSARDClassifier, data)
    addresses = zeros(Int, dt.n_rams)
    count = 0
    for i in 0:dt.n_rams-1
        for j in 0:dt.n_bits-1
            x = dt._mapping[(i * dt.n_bits + j) % (dt.retina_size)+1] - 1
            index = div(x,(dt.n_tics))
            value = div((data[index+1] -  dt.offsets[index+1]) * dt.n_tics, dt.ranges[index+1])
            println("cnt ", count, " data ", data[index+1], " off ", dt.offsets[index+1], " range ", dt.ranges[index+1], " value ", value," idx ", idx, " index ",index, " x ", x)
			if x % dt.n_tics < value
                addresses[i+1] += dt.mypowers[dt.n_bits - j]
            end
            count += 1
		end
	end
    return addresses
end

using CSV, DataFrames, MLJ, MLBase

df = CSV.read("/Users/maurizio/WiSARDpy/datasets/iris.csv", DataFrames.DataFrame)


X = Matrix(select(df, Not([:species])))
y = vec(Matrix(select(df, [:species])))
model = WiSARDClassifier(n_bits=4,n_tics=16)
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=1)
MLJ.fit(model, Any, X[train,:], y[train,:])
ŷ = MLJ.predict(model, Any, X[test,:])

#tpl = mk_tuple(model, X[1,:])
#print(tpl)
