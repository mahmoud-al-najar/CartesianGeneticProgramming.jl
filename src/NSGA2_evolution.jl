####
#### ORIGINAL IMPLEMENTATION: https://github.com/SuReLI/NSGA-II.jl
####
export NSGA2Evolution,fastNonDominatedSort!,dominates,crowdingDistanceAssignement!,NSGA2Generation,NSGA2Populate,NSGA2Evaluate, get_fits_and_ranks, plot_fits_and_ranks

import Cambrian.populate, Cambrian.evaluate,  Cambrian.selection, Cambrian.generation, Cambrian.log_gen

mutable struct NSGA2Evolution{T<:Individual} <: Cambrian.AbstractEvolution
    config::NamedTuple
    logger::CambrianLogger
    population::Array{T}
    fitness::Function
    type::DataType
    rank::Dict{UInt64,Int64}
    distance::Dict{UInt64,Float64}
    gen::Int
    offsprings::Dict{UInt64,Bool}
    ties::Int
end

# using Plots
using Logging

populate(e::NSGA2Evolution) = NSGA2Populate(e)
evaluate(e::NSGA2Evolution) = NSGA2Evaluate(e)
generation(e::NSGA2Evolution) = NSGA2Generation(e)

function NSGA2Evolution(cfg::NamedTuple, fitness::Function;
                      logfile=string("logs/", cfg.id, ".csv"), kwargs...)
                      
    logger = CambrianLogger(logfile)
    kwargs_dict = Dict(kwargs)
    gen = 0
    ties = 0
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
            ind_fit = dna["fitness"]

            ind.fitness .= ind_fit
            population[i] = ind
            i += 1
        end
    else
        population = Cambrian.initialize(CGPInd, cfg)
    end

    rank=Dict{UInt64,Int64}()
    distance=Dict{UInt64,Float64}()
    offsprings=Dict{UInt64,Bool}()
    for x in population
        offsprings[objectid(x)]=true
    end

    # config::NamedTuple
    # logger::CambrianLogger
    # population::Array{T}
    # fitness::Function
    # type::DataType
    # rank::Dict{UInt64,Int64}
    # distance::Dict{UInt64,Float64}
    # gen::Int
    # offsprings::Dict{UInt64,Bool}
    # ties::Int
    NSGA2Evolution(cfg, logger, population, fitness,CGPInd ,rank,distance, gen, offsprings, ties)
end

function NSGA2Evaluate(e::NSGA2Evolution)
    Threads.@threads for i in eachindex(e.population)
        if e.offsprings[objectid(e.population[i])]
            # TODO: no need to re-evaluate full pop
            e.population[i].fitness[:] = e.fitness(e.population[i])[:]
            e.offsprings[objectid(e.population[i])]=false
        end
    end
end


# function NSGA2Evaluate(e::NSGA2Evolution)
#     index_WAVES = rand(1:100)
#     index_RIVER = rand(1:102)
#     index_SLA = rand(1:171)

#     #i1 = (e.gen-1)*5+1
#     #i2 = i1 + 1
#     #i3 = i1 + 2
#     #i4 = i1 + 3
#     #i5 = i1 + 4 

#     Threads.@threads for i in eachindex(e.population)
#     #for i in eachindex(e.population)
#         # TODO: no need to re-evaluate full pop
#         e.population[i].fitness[:] = e.fitness(e.population[i], index_WAVES, index_RIVER, index_SLA)[:]
#         #e.population[i].fitness[:] = e.fitness(e.population[i], i1, i2, i3, i4, i5)[:]
#         e.offsprings[objectid(e.population[i])]=false
#     end
# end

function NSGA2Populate(e::NSGA2Evolution)
    T=typeof(e.population[1])
    Qt=Array{T}(undef,0)
    for ind in e.population
        push!(Qt,ind)
    end
    i=0
    ties = 0
    while i < e.config.n_offsprings
        parents =  [Cambrian.tournament_selection(e.population, 5) for i in 1:2]
        if sum(parents[1].fitness .< parents[2].fitness) == sum(parents[2].fitness .< parents[1].fitness)
            ties += 1
        end
        if  rand() < e.config.p_crossover
            child1,child2 = crossover(parents...)
        else
            child1,child2= copy(parents[1]),copy(parents[2])
        end
        if rand() < e.config.p_mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
        end
        push!(Qt,child1)
        e.offsprings[objectid(child1)]=true
        i+=1
        if i < e.config.n_offsprings
            push!(Qt,child2)
            e.offsprings[objectid(child2)]=true
            i+=1
        end
    end
    e.rank=Dict(objectid(x)=>0 for x in Qt)
    e.distance=Dict(objectid(x)=>0. for x in Qt)
    e.population=Qt
    e.ties = ties
end

function dominates(e::NSGA2Evolution,ind1::T,ind2::T) where {T <: Individual}
    dom=true
    for i in 1:e.config.d_fitness
        if ind1.fitness[i]<ind2.fitness[i]
            return false
        elseif ind1.fitness[i]>ind2.fitness[i]
            dom=true
        end
    end
    return dom
end

function fastNonDominatedSort!(e::NSGA2Evolution)
    T=typeof(e.population[1])
    Fi=Array{T}(undef,0)

    n=Dict(objectid(x)=>0 for x in e.population)
    S=Dict(objectid(x)=>Array{T}(undef,0) for x in e.population)

    for ind1 in e.population
        for ind2 in e.population
            if dominates(e,ind1,ind2)
                push!(S[objectid(ind1)],ind2)
            elseif dominates(e,ind2,ind1)
                n[objectid(ind1)]+=1
            end
        end
        if n[objectid(ind1)]==0
            e.rank[objectid(ind1)]=1
            push!(Fi,ind1)
        end
    end

    i=1

    while isempty(Fi)==false
        Q=Array{T}(undef,0)
        for ind1 in Fi
            currentS=S[objectid(ind1)]
            for ind2 in currentS
                n[objectid(ind2)]-=1
                if n[objectid(ind2)]==0
                    e.rank[objectid(ind2)]=i+1
                    push!(Q,ind2)
                end
            end
        end
        i=i+1
        Fi=Q
    end
end

function crowdingDistanceAssignement!(e::NSGA2Evolution,I::Array{T}) where {T <: Individual}
    for x in I
        e.distance[objectid(x)]=0
    end
    l=length(I)
    for i in 1:e.config.d_fitness
        sort!(I,by=x->x.fitness[i])
        if I[1].fitness[i]!=I[end].fitness[i]
            e.distance[objectid(I[1])]=Inf
            e.distance[objectid(I[end])]=Inf
            quot=I[end].fitness[i]-I[1].fitness[i]
            for j in 2:l-1
                e.distance[objectid(I[j])] = e.distance[objectid(I[j])] + (I[j+1].fitness[i]-I[j-1].fitness[i]) / quot
            end
        end
    end
end

function get_fits_and_ranks(e::NSGA2Evolution)
    fits_and_ranks = Array{Float64}(undef, e.config.n_population, e.config.d_fitness + 1)  # last index: rank
    for i in eachindex(e.population)
        ind = e.population[i]
        for j in collect(1:e.config.d_fitness)
            fits_and_ranks[i, j] = ind.fitness[j]
        end
        fits_and_ranks[i, end] = e.rank[objectid(ind)]
    end
    fits_and_ranks
end

function plot_fits_and_ranks(e::NSGA2Evolution)
    fits_and_ranks = get_fits_and_ranks(e)
    display(fits_and_ranks)
    title = "Gen:$(e.gen)"
    title *= ", |R1|: $(count(x -> e.rank[objectid(x)] == 1, e.population))"
    title *= ", |R2|: $(count(x -> e.rank[objectid(x)] == 2, e.population))"
    title *= ", |R3|: $(count(x -> e.rank[objectid(x)] == 3, e.population))"
    println(title)
    p1 = scatter(fits_and_ranks[:, 1], fits_and_ranks[:, 2], group=fits_and_ranks[:, 3], title=title, legend=:outerbottomright)
    p1
end

function NSGA2Generation(e::NSGA2Evolution)
    if e.gen>1
        T=typeof(e.population[1])
        fastNonDominatedSort!(e)
        Pt1=Array{T}(undef,0)
        i=1
        sort!(e.population,by= x -> e.rank[objectid(x)])
        rank=1
        indIni=1
        indNext=findlast(x -> e.rank[objectid(x)] == rank , e.population)
        while indNext < e.config.n_population
            # println("indNext: $indNext, while indNext < e.config.n_population")
            Pt1=[Pt1...,e.population[indIni:indNext]...]
            rank+=1
            indIni=indNext+1
            indNext=findlast(x -> e.rank[objectid(x)] == rank, e.population)
        end
        if isempty(Pt1)
            # println("indNext: $indNext, if isempty(Pt1)")
            I=e.population[1:indNext]
            crowdingDistanceAssignement!(e,I)
            sort!(I, by= x->e.distance[objectid(x)],rev=true)
            Pt1=I[1:e.config.n_population]
        else
            # println("else")
            I=e.population[indIni:indNext]
            crowdingDistanceAssignement!(e,I)
            sort!(I, by= x->e.distance[objectid(x)],rev=true)
            Pt1=[Pt1...,I[1:e.config.n_population-length(Pt1)]...]
        end
        e.population=Pt1
        e.offsprings=Dict(objectid(x)=>false for x in e.population)
        # if e.gen % e.config.log_gen == 0
        #     display(plot(plot_fits_and_ranks(e)))
        #     # readline()
        # end
    end
end

function log_gen(e::NSGA2Evolution)
    for d in 1:e.config.d_fitness
        maxs = map(i->i.fitness[d], e.population)
        with_logger(e.logger) do
            @info Formatting.format("{1:04d},{2:e},{3:e},{4:e}#,{5:e},{6:e},{7:e}",
                e.gen, maximum(maxs), mean(maxs), std(maxs), count(x->x==1,values(e.rank)), length(values(e.rank)), e.ties)
        end
    end
    flush(e.logger.stream)
end

"save the population in gens/"
function save_gen(e::NSGA2Evolution)
    path = joinpath("gens", String(e.config.id))
    path = joinpath(path, Formatting.format("{1:04d}", e.gen))
    mkpath(path)
    sort!(e.population)
    for i in eachindex(e.population)
        fname = joinpath(path, Formatting.format("{1:04d}.dna", i))
        f = open(fname, "w+")
        write(f, string(e.population[i]))
        close(f)
    end
end
