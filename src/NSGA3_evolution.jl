####
#### ORIGINAL IMPLEMENTATION: https://github.com/SuReLI/NSGA-II.jl
####
export NSGA3Evolution, fastNonDominatedSort!, dominates, NSGA3Generation, NSGA3Populate, NSGA3Evaluate
import Cambrian.populate, Cambrian.evaluate,  Cambrian.selection, Cambrian.generation, Cambrian.log_gen
# using LinearAlgebra

mutable struct NSGA3Evolution{T<:Individual} <: Cambrian.AbstractEvolution
    config::NamedTuple
    logger::CambrianLogger
    population::Array{T}
    fitness::Function
    type::DataType
    rank::Dict{UInt64,Int64}
    distance::Dict{UInt64,Float64}
    gen::Int
    offsprings::Dict{UInt64,Bool}
    reference_points::Array{Vector{Float64},1}
end

populate(e::NSGA3Evolution) = NSGA3Populate(e)
evaluate(e::NSGA3Evolution) = NSGA3Evaluate(e)
generation(e::NSGA3Evolution) = NSGA3Generation(e)

"""
    fitsmat(population)
returns fitness matrix of size (popsize, d_fitness)
"""
function fitsmat(population::Array{CGPInd})
    fits = Array{Float64}(undef, length(population), length(population[1].fitness))
    for i in 1:length(population)
        fits[i, :] .= population[i].fitness
    end
    fits
end

"""
    fitsmat(population)
returns population fitness array
"""
function fitsarray(population::Array{CGPInd})
    [ind -> ind.fitness, population]
end

"""
    nadir(population)
Computes the nadir point from an array of CGPInd's.
nadir: minimum fitness per dimension in a maximizing problem
"""
function nadir(population::Array{CGPInd})
    nadir(fitsmat(population))
end

nadir(fits::Array{Float64,2}) = minimum(fits, dims=1)

"""
    ideal(population)
Computes the ideal point from an array of CGPInd's.
ideal: maximum fitness per dimension in a maximizing problem
"""
function ideal(population::Array{CGPInd})
    ideal(fitsmat(population))
end

ideal(fits::Array{Float64,2}) = maximum(fits, dims=1)

function NSGA3Evolution(cfg::NamedTuple, fitness::Function;
                      logfile=string("logs/", cfg.id, ".csv"), kwargs...)
                      
    logger = CambrianLogger(logfile)
    kwargs_dict = Dict(kwargs)
    gen = 0
    if haskey(kwargs_dict, :init_function)
        println("init_function")
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
        println("rand init")
        population = Cambrian.initialize(CGPInd, cfg)
    end

    rank=Dict{UInt64,Int64}()
    distance=Dict{UInt64,Float64}()
    offsprings=Dict{UInt64,Bool}()
    for x in population
        offsprings[objectid(x)]=true
    end

    reference_points = gen_weights(cfg.d_fitness, cfg.partitions)
    NSGA3Evolution(cfg, logger, population, fitness,CGPInd ,rank,distance, gen, offsprings, reference_points)
end

function gen_weights(a, b)
    nobj = a;
    H    = b;
    a    = zeros(nobj);
    d    = H;
    w    = [];
    produce_weight!(a, 1, d, H, nobj, w)
    return Array.(w)
end

function  produce_weight!(a, i, d, H, nobj, w)
    for k=0:d
        if i<nobj
            a[i] = k;
            d2   = d - k;
            produce_weight!(a, i+1, d2, H, nobj, w);
        else
            a[i] = d;
            push!(w, a/H)
            break;
        end
    end
end

function NSGA3Evaluate(e::NSGA3Evolution)
    i_pcts = rand(e.config.d_fitness)

    Threads.@threads for i in eachindex(e.population)
    # for i in eachindex(e.population)
        # TODO: no need to re-evaluate full pop
        e.population[i].fitness[:] = e.fitness(e.population[i], i_pcts)[:]
    end
end

function NSGA3Populate(e::NSGA3Evolution)
    T=typeof(e.population[1])
    Qt=Array{T}(undef,0)
    for ind in e.population
        push!(Qt,ind)
    end
    i=0
    while i < e.config.n_offsprings
        parents =  [tournament_selection(e.population, 5) for i in 1:2]
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
end

function dominates(e::NSGA3Evolution,ind1::T,ind2::T) where {T <: Individual}
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

function fastNonDominatedSort!(e::NSGA3Evolution)
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

function hyperplane_normalization(population) 
    M = length(population[1].fitness)

    ideal_point = ideal(population)[:]
    nadir_point = fill(-Inf, length(ideal_point))

    Fx = fitsmat(population) .- ideal_point'

    # identify extreme points
    extreme_points = zeros(Int, M)
    w = LinearAlgebra.I + fill(1e-6, M, M)

    for i in 1:M
        extreme_points[i] = argmin(nadir(Fx' ./ w[i,:])[:])
    end

    # check if intercepts can be obtained
    S = Fx[extreme_points,:]
    if LinearAlgebra.det(S) ≈ 0 # check if soluble matrix
        nadir_point = nadir(population)[:]
    else
        hyperplane = S \ ones(M)
        intercepts = 1 ./ hyperplane # intercepts
        nadir_point = ideal_point + intercepts
    end


    ideal_point, nadir_point
end


function normalize(population) 
    ideal_point, nadir_point = hyperplane_normalization(population)

    b = nadir_point - ideal_point

    # prevent division by zero
    mask = b .< eps()
    b[mask] .= eps()

    return [ (sol.fitness - ideal_point) ./ b for sol in population ]
end

distance_point_to_rect(s, w) = @fastmath norm(s - (dot(w,s) / dot(w, w))*w  )

function associate!(nich, nich_freq, distance, F, reference_points, l) 
    N = length(F)

    # find closest nich to corresponding solution
    for i = 1:N
        Threads.@threads for j = 1:lastindex(reference_points)
            d = distance_point_to_rect(F[i], reference_points[j])

            distance[i] < d && continue

            distance[i] = d
            nich[i] = j
        end

        # not associate last  front
        if i < l
            nich_freq[nich[i]] += 1
        end
        
    end
end

pick_random(itr, item) = rand(findall(i -> i == item, itr))
find_item(itr, item)  = findall(i -> i == item, itr)

function niching!(e, l)
    ### population: including the last rank for niching, popsize could be > N
    ### reference points: e.reference_points
    ### N: popsize
    ### l: start index of last rank in population
    population = e.population
    reference_points = e.reference_points
    N = e.config.n_population
    if length(population) == N
        return nothing
    end
    
    F = normalize(population)
    k = l

    # allocate memory
    nich = zeros(Int, length(population))
    nich_freq = zeros(Int, length(reference_points))
    available_niches = ones(Bool, length(reference_points))
    distance = fill(Inf, length(population))

    # associate to niches
    associate!(nich, nich_freq, distance, F, reference_points, l)

    # keep last front
    last_front_id = k:length(population)
    last_front = population[last_front_id]
    deleteat!(population, last_front_id)

    # last front niches information
    niches_last_front = nich[last_front_id]
    distance_last_front = distance[last_front_id]
    
    # save survivors
    i = 1
    while k <= N
        mini = minimum(view(nich_freq, available_niches))
        # nich to be assigned
        j_hat = pick_random(nich_freq, mini)

        # candidate solutions 
        I_j_hat = find_item( niches_last_front, j_hat )
        if isempty(I_j_hat)
            available_niches[j_hat] = false
            continue
        end

        if mini == 0
            ds = view(distance_last_front, I_j_hat)
            s = I_j_hat[argmin(ds)]
            push!(population, last_front[s])
        else
            s = rand(I_j_hat)
            push!(population, last_front[s])
        end

        nich_freq[j_hat] += 1
        deleteat!(last_front, s)
        deleteat!(niches_last_front, s)
        deleteat!(distance_last_front, s)

        k += 1
    end
    e.population = population
    nothing
end

function NSGA3Generation(e::NSGA3Evolution)
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
            Pt1=[Pt1...,e.population[indIni:indNext]...]
            rank+=1
            indIni=indNext+1
            indNext=findlast(x -> e.rank[objectid(x)] == rank, e.population)
        end

        deleteat!(e.population, indNext:length(e.population))
        l = findfirst(ind -> e.rank[objectid(ind)]==rank, e.population)
        niching!(e, l)
        e.offsprings=Dict(objectid(x)=>false for x in e.population)
    end
end

function log_gen(e::NSGA3Evolution)
    for d in 1:e.config.d_fitness
        maxs = map(i->i.fitness[d], e.population)
        with_logger(e.logger) do
            @info Formatting.format("{1:04d},{2:e},{3:e},{4:e}#,{5:e}",
                e.gen, maximum(maxs), mean(maxs), std(maxs), count(x->x==1,values(e.rank)))
        end
    end
    flush(e.logger.stream)
end
