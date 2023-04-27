export get_outputs, set_inputs, process

function get_outputs(ind::CGPInd)
    # doesn't re-process, just gives outputs
    # outputs = Array{typeof(ind.buffer[1])}(undef, length(ind.outputs))
    outputs = Array{MType}(undef, length(ind.outputs))
    for i in eachindex(outputs)
        # println("testing")
        # println(typeof(ind.buffer[ind.outputs[i]]))
        # println(typeof(ind.outputs[i]))
        outputs[i] = ind.buffer[ind.outputs[i]]
    end
    outputs
end

function set_inputs(ind::CGPInd, inputs::AbstractArray)
    for i in eachindex(inputs)
        ind.buffer[i] = inputs[i]
    end
end

function process(ind::CGPInd)
    for i in eachindex(ind.nodes)
        n = ind.nodes[i]
        if n.active
            ind.buffer[i] = n.f(ind.buffer[n.x], ind.buffer[n.y], n.p)
        end
    end
    get_outputs(ind)
end

function process(ind::CGPInd, inputs::AbstractArray)
    set_inputs(ind, inputs)
    process(ind)
end
