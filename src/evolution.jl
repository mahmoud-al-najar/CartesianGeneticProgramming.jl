export CGPEvolution

import Cambrian.populate, Cambrian.evaluate, Cambrian.max_selection

# include("../shorefor/create_shorefor_ind.jl")

mutable struct CGPEvolution{T} <: Cambrian.AbstractEvolution
    config::NamedTuple
    logger::CambrianLogger
    population::Array{T}
    fitness::Function
    gen::Int
end

function ga_populate(e::CGPEvolution)
    new_pop = Array{CGPInd}(undef, 0)
    if e.config.n_elite > 0
        sort!(e.population)
        # append!(new_pop,
                # e.population[(length(e.population)-e.config.n_elite+2):end]) ## +2 instead of +1 // shorefor always added
        append!(new_pop,e.population[(length(e.population)-e.config.n_elite+1):end])
        # shorefor = get_shorefor(e.config)
        # shorefor = get_fullmodel_shorefor_dxdt(e.config)
        # push!(new_pop, shorefor)
    end

    for i in (e.config.n_elite+1):e.config.n_population
        p1 = selection(e.population)
        child = deepcopy(p1)
        if e.config.p_crossover > 0 && rand() < e.config.p_crossover
            parents = vcat(p1, [selection(e.population) for i in 2:e.config.n_parents])
            child = crossover(parents...)
        end
        if e.config.p_mutation > 0 && rand() < e.config.p_mutation
            child = mutate(child)
        end
        push!(new_pop, child)
    end
    e.population = new_pop
end

populate(e::CGPEvolution) = ga_populate(e)
evaluate(e::CGPEvolution) = Cambrian.fitness_evaluate(e, e.fitness)
selection(pop::Array{CGPInd,1}) = tournament_selection(pop, 3)

function CGPEvolution(cfg::NamedTuple, fitness::Function;
                      logfile=string("logs/", cfg.id, ".csv"), kwargs...)
    logger = CambrianLogger(logfile)
    kwargs_dict = Dict(kwargs)
    gen = 0
    if haskey(kwargs_dict, :init_function)
        population = Cambrian.initialize(CGPInd, cfg, init_function=kwargs_dict[:init_function])
        
    elseif haskey(kwargs_dict, :population_dir)
        path = kwargs_dict[:population_dir]
        population = Array{CGPInd,1}(undef, cfg.n_population)
        gen = parse(Int, split(path, "/")[end])
        i = 1
        for path in readdir(path; join=true)
            dna = JSON.parsefile(path)
            chromo = convert(Array{Float64,1}, dna["chromosome"])
            ind = CGPInd(cfg, chromo)
            population[i] = ind
            i += 1
        end
    else
        population = Cambrian.initialize(CGPInd, cfg)
    end
    CGPEvolution(cfg, logger, population, fitness, gen)
end
