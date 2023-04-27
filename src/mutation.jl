export uniform_mutate, goldman_mutate, profiling_mutate, is_valid

"create a child by randomly mutating genes"
function uniform_mutate(cfg::NamedTuple, ind::CGPInd; kwargs...)::CGPInd
    chromosome = copy(ind.chromosome)
    chance = rand(length(ind.chromosome))
    non_output = length(ind.chromosome) - length(ind.outputs)
    change = [chance[1:non_output] .<= cfg.m_rate;
              chance[(non_output+1):end] .<= cfg.out_m_rate]
    chromosome[change] = rand(sum(change))
    kwargs_dict = Dict(kwargs)
    if haskey(kwargs_dict, :init_function)
        return kwargs_dict[:init_function](cfg, chromosome)
    else
        return CGPInd(cfg, chromosome)
    end
end

"create a child that is structurally different from the parent"
function goldman_mutate(cfg::NamedTuple, ind::CGPInd, test_inputs::Any, evaluate_individual::Function; kwargs...)::CGPInd
    child = uniform_mutate(cfg, ind; kwargs...)
    while true
        if is_valid(cfg, child, ind, test_inputs, evaluate_individual)
            if any(ind.outputs != child.outputs)
                return child
            else
                for i in eachindex(ind.nodes)
                    if ind.nodes[i].active
                        if child.nodes[i].active
                            if (ind.nodes[i].f != child.nodes[i].f
                                || ind.nodes[i].x != child.nodes[i].x
                                || ind.nodes[i].y != child.nodes[i].y)
                                return child
                            end
                        else
                            return child
                        end
                    end
                end
            end
        end
        child = uniform_mutate(cfg, ind; kwargs...)
    end
    nothing
end

function goldman_mutate_no_constraints(cfg::NamedTuple, ind::CGPInd; kwargs...)::CGPInd
    child = uniform_mutate(cfg, ind; kwargs...)
    while true
        if any(ind.outputs != child.outputs)
            return child
        else
            for i in eachindex(ind.nodes)
                if ind.nodes[i].active
                    if child.nodes[i].active
                        if (ind.nodes[i].f != child.nodes[i].f
                            || ind.nodes[i].x != child.nodes[i].x
                            || ind.nodes[i].y != child.nodes[i].y)
                            return child
                        end
                    else
                        return child
                    end
                end
            end
        end
        child = uniform_mutate(cfg, ind; kwargs...)
    end
    nothing
end

"create a child that gives different outputs based on the provided inputs (a 2D matrix of (n_in, n_samples))"
function profiling_mutate(cfg::NamedTuple, ind::CGPInd, inputs::AbstractArray; kwargs...)::CGPInd
    child = uniform_mutate(cfg, ind; kwargs...)
    while true
        for i in 1:size(inputs, 2)
            out_ind = process(ind, inputs[:, i])
            out_child = process(child, inputs[:, i])
            if any(out_ind .!= out_child)
                return child
            end
        end
        child = uniform_mutate(cfg, ind; kwargs...)
    end
    nothing
end

function is_valid(cfg::NamedTuple, ind::CGPInd, parent::CGPInd, test_inputs::Any, evaluate_individual::Function)
    try
        # println("\n\n")
        #println("is_valid: $(objectid(parent))")
        if any(ind.outputs .<= ind.n_in) # no direct output-input connections
            return false 
        end
        
        # omega, P, phi, 2phi, Dir, Hsb, Tp, E, Sla, rivdis
        # test_inputs = [rand(10), rand(10), 170.0, 170.0*2, rand(10), rand(10), rand(10), rand(10), rand(10), rand(10)]
        reset!(ind)
        #println("before process()")
        outs = process(ind, test_inputs)

        if any(typeof.(outs) .!= Vector{Float64})  # vector outputs only
        # if any(typeof.(outs) .!= Float64)  # scalar outputs only
        #    println("vector outputs only")
            return false 
        end

        if length(outs[1]) != length(test_inputs[1])
        #    println("length(outs[1]) != length(test_inputs[1])")
            return false
        end

        reset!(parent)
        outs0 = process(parent, test_inputs)

        # display(parent.fitness)
        #ch_fit = evaluate_individual(ind, 1, 2, 3, 4, 5)
        #ind.fitness .= ch_fit
        # display(ch_fit)        
        # display("parent_outs: $outs0")
        # display("child_outs: $outs")
        
        #if sum(isequal.(outs0, outs)) != 0  # no constant outputs
        if outs0[1] == outs[1]
        #    println("outs0 == outs")
            return false
        end

        #if parent.fitness == ind.fitness
        #    #display("parent.fitness == ind.fitness\n")
        #    return false
        #end

        return true

    catch
        #println("catch")
        return false
    end
end
