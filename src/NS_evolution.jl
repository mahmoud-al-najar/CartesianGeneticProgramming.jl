####################
## Novelty Search ##
####################
export NSEvolution, NSGeneration, NSPopulate, NSEvaluate, save_gen, calc_novelty, novelty_ordering, remove_duplicates, remove_negative_fitness
import Cambrian.populate, Cambrian.evaluate,  Cambrian.selection, Cambrian.generation, Cambrian.save_gen, Cambrian.log_gen

mutable struct NSEvolution{T<:CGPInd} <: Cambrian.AbstractEvolution
    config::NamedTuple
    logger::CambrianLogger
    population::Array{T}
    fitness::Function
    type::DataType
    inds_archive::Array{CGPInd}
    novelty_archive::Array{Float64}
    k::Int64  # k closest neighbors in novelty calculation
    Nq::Int64  # Nq samples
    gen::Int
end

# using Plots
using StatsBase
using LinearAlgebra
using Distances
using Formatting
using NaNStatistics

populate(e::NSEvolution) = NSPopulate(e)
evaluate(e::NSEvolution) = NSEvaluate(e)
generation(e::NSEvolution) = NSGeneration(e)

function NSEvolution(cfg::NamedTuple, fitness::Function;
                      logfile=string("logs/", cfg.id, ".csv"), kwargs...)
                      
    logger = CambrianLogger(logfile)
    kwargs_dict = Dict(kwargs)
    gen = 0
    if haskey(kwargs_dict, :init_function)
        population = Cambrian.initialize(CGPInd, cfg, init_function=kwargs_dict[:init_function])
        inds_archive=Array{CGPInd}(undef, 0)
    elseif haskey(kwargs_dict, :population_dir)
        path = kwargs_dict[:population_dir]
        gen = parse(Int, split(path, "/")[end][1:end-4])
        dna_archive = JSON.parsefile(path)["archive"]
        inds_archive = Array{CGPInd,1}(undef, 0)
        println("length(dna_archive): ", length(dna_archive))
        
        for i in eachindex(dna_archive)
            chromo = convert(Array{Float64,1}, dna_archive[i]["chromosome"])
            ind = CGPInd(cfg, chromo)
            fits = dna_archive[i]["fitness"]
            if !(nothing in fits)
                ind.fitness .= dna_archive[i]["fitness"]
                push!(inds_archive, ind)
            end
        end
        population = Array{CGPInd}(undef, 0)
    else
        population = Cambrian.initialize(CGPInd, cfg)
        inds_archive=Array{CGPInd}(undef, 0)
    end
    
    novelty_archive=Array{Float64}(undef, 0)

    NSEvolution(cfg, logger, population, fitness, CGPInd, inds_archive, novelty_archive, cfg.k, cfg.Nq, gen)
end

function NSEvaluate(e::NSEvolution)
    for i in eachindex(e.population)
        e.population[i].fitness[:] .= e.fitness(e.population[i])[:]
    end
end

#function calc_novelty(ind::CGPInd, archive::Array{CGPInd}, k::Int)
#    distances = Array{Float64}(undef, length(archive))
#    for i in eachindex(archive)
#        dist = euclidean(ind.fitness, archive[i].fitness)
#        if abs(dist) != Inf
#            distances[i] = dist
#        else
#            distances[i] = 0.0
#        end
#    end
#    sorted_indices = sortperm(distances)[1:min(length(distances), k)]
#    return nanmean(distances[sorted_indices])
#end

#function novelty_ordering(pop::Array{CGPInd}, reference_inds::Array{CGPInd}, k::Int)
#    novelty = Array{Float64}(undef, length(pop))
#    # for i in eachindex(pop)
#    Threads.@threads for i in eachindex(pop)
#        ind1 = pop[i]
#        novelty[i] = calc_novelty(ind1, reference_inds, k)
#    end
#    sorted_novelty_indices = reverse(sortperm(novelty))
#    return pop[sorted_novelty_indices], novelty[sorted_novelty_indices]
#end

function calc_novelty(ind::CGPInd, archive::Array{CGPInd}, k::Int)
    ### score-correlation-based novelty 
    correlations = Array{Float64}(undef, length(archive))
    for i in eachindex(archive)
        correlations[i] = nancor(ind.fitness, archive[i].fitness)
    end
    # reverse to get most correlated
    sorted_indices = reverse(sortperm(correlations))[1:min(length(correlations), k)]
    return nanmean(correlations[sorted_indices])
end

function novelty_ordering(pop::Array{CGPInd}, reference_inds::Array{CGPInd}, k::Int)
    ### ordering score-correlation-based novelty metric
    mean_correlations = Array{Float64}(undef, length(pop))
    Threads.@threads for i in eachindex(pop)
        mean_correlations[i] = calc_novelty(pop[i], reference_inds, k)
    end
    # don't reverse, ascending correlation ordering
    sorted_novelty_indices = sortperm(mean_correlations)
    return pop[sorted_novelty_indices], mean_correlations[sorted_novelty_indices]
end

function uniqueidx(x::AbstractArray{T}) where T
    uniqueset = Set{T}()
    ex = eachindex(x)
    idxs = Vector{eltype(ex)}()
    for i in ex
        xi = x[i]
        if !(xi in uniqueset)
            push!(idxs, i)
            push!(uniqueset, xi)
        end
    end
    idxs
end

function remove_duplicates(inds::Array{CGPInd})
    fits = map(x->x.fitness, inds)
    unique_indices = uniqueidx(fits)
    return inds[unique_indices]
end

function remove_negative_fitness(inds::Array{CGPInd})
    # fits[map(x -> !(0 in (x .>= 0)), eachslice(fits, dims=1)), :]
    idx = map(x -> !(0 in (x.fitness .>= 0)), inds)
    return inds[idx], idx
end

function NSPopulate(e::NSEvolution)
    e.inds_archive, idx = remove_negative_fitness(e.inds_archive)
    e.inds_archive = remove_duplicates([e.inds_archive..., e.population...])
    e.inds_archive, e.novelty_archive = novelty_ordering(e.inds_archive, e.inds_archive, e.config.k)
    
    e.inds_archive = e.inds_archive[1:min(e.config.n_population * 10, length(e.inds_archive))]
    e.novelty_archive = e.novelty_archive[1:min(e.config.n_population * 10, length(e.novelty_archive))]

    i = 1
    i_parent = 1
    offspring = Array{e.type}(undef, e.config.n_offsprings)

    offspring_per_parent = e.config.max_offspring_per_parent
    if length(e.inds_archive)*offspring_per_parent < e.config.n_offsprings
        offspring_per_parent = ceil(e.config.n_offsprings / length(e.inds_archive))
    end

    while i <= e.config.n_offsprings
        parent = e.inds_archive[i_parent]
        println("Parent novelty: ", e.novelty_archive[i_parent])
        for _ in 1:offspring_per_parent
            if i <= e.config.n_offsprings
                child = copy(parent)
                if rand() < e.config.p_mutation
                    child = mutate(parent)
                end
                offspring[i] = child
                i += 1
            else
                break
            end
        end
        i_parent += 1
    end
    
    e.population = offspring
end

function NSGeneration(e::NSEvolution)
    # if length(e.inds_archive) > 1
    #     novelty_sorted_pop, novelty_archive = novelty_ordering(e.population, e.inds_archive, e.config.k)
    # else
    #     novelty_sorted_pop, novelty_archive = novelty_ordering(e.population, e.population, e.config.k)
    # end
    
    # novelty_sorted_pop = novelty_sorted_pop[1:e.Nq]
    # push!(e.inds_archive, novelty_sorted_pop...)

    # novelty_archive = novelty_archive[1:e.Nq]
    # push!(e.novelty_archive, novelty_archive...)
    
    println("NSGeneration ==> ", length(e.inds_archive))
end

"overrides Cambrian.e::AbstractEvolution, records inds_archive in single dna file"
function save_gen(e::NSEvolution)
    path = joinpath(e.config.output_dir, "gens", String(e.config.id))
    mkpath(path)
    fname = joinpath(path, Formatting.format("{1:04d}.dna", e.gen))
    f = open(fname, "w+")
    write(f, "{\"archive\": [")
    for i in eachindex(e.inds_archive)        
        write(f, string(e.inds_archive[i]))
        if i < length(e.inds_archive)
            write(f, ",")
        end
    end
    write(f, "]}")
    close(f)
end

function log_gen(e::NSEvolution)
    for d in 1:e.config.d_fitness
        maxs = map(i->i.fitness[d], e.population)
        with_logger(e.logger) do
            @info Formatting.format("{1:04d},{2:e},{3:e},{4:e}#,{5:e}",
                e.gen, maximum(maxs), mean(maxs), std(maxs), length(e.inds_archive))
        end
    end
    flush(e.logger.stream)
end

