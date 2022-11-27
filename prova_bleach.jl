function _calc_confidence(results::Matrix{Int})
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
    c = 1 - second_max / max_value
    return c
end

function _test_bleaching(res_disc::Vector{Float64})
    """ Testing """
    b = 1
    confidence = 0.0
    result_partial = Any
    display(res_disc)
    println("-") 
    display(result_partial)
    while confidence < 0.01
        result_partial = sum(x->x>=1, res_disc, dims=2)
        display(result_partial)
        confidence = _calc_confidence(result_partial)
        b += 1
        println("-") 
        display(result_partial)
        if (sum(result_partial) == 0)
            result_partial = sum(res_disc.>= 1, dims=2)
            println("-") 
            display(result_partial)
            break
        end
        break
    end
    result_sum = sum(result_partial, dims=1)[1] * 1.0
    println(result_sum)
    if result_sum==0.0
        result = sum(res_disc, dims=2)./float(dt.nrams)
    else
        result = sum(res_disc, dims=2)./result_sum
    end
    return argmax(result)[1]
end

response = [16. 45. 50. 50. 49. 37. 15. 46. 37. 50. 50. 50. 34. 50. 50. 50.; 3.  6. 39. 50. 37.  5.  1. 50.  0.  0. 25. 50.  0.  0. 35. 50.; 0.  1. 19. 42. 45. 13.  2. 50.  0.  0.  0. 34.  0.  0.  1. 27.]
println(_test_bleaching(response))