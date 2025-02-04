export Node, CGPInd, get_genes, set_genes!, reset!, forward_connections, get_output_trace, get_chromosome, get_maxs
import Base.copy, Base.String, Base.show, Base.summary
import Cambrian.print

MType = Union{Nothing, Float64, Array{Float64}}
SorX = Union{Symbol, Expr}

"default function for nodes, will cause error if used as a function node"
function f_null(args...)::Nothing
    nothing
end

struct Node
    x::Int16
    y::Int16
    f::Function
    p::Array{Float64}
    active::Bool
end

struct CGPInd <: Cambrian.Individual
    n_in::Int16
    n_out::Int16
    n_parameters::Int16
    chromosome::Array{Float64}
    genes::Array{Float64}
    outputs::Array{Int16}
    nodes::Array{Node}
    buffer::AbstractArray
    fitness::Array{Float64}
end

function recur_active(active::BitArray, n_in::Integer, ind::Int16,
                      xs::Array{Int16}, ys::Array{Int16}, fs::Array{Int16},
                      two_arity::BitArray)::BitArray
    if ind > 0 && ~active[ind]
        active[ind] = true
        active = recur_active(active, n_in, xs[ind], xs, ys, fs, two_arity)
        if two_arity[fs[ind]]
            active = recur_active(active, n_in, ys[ind], xs, ys, fs, two_arity)
        end
    end
    active
end

function find_active(cfg::NamedTuple, genes::Array{Float64},
                     outputs::Array{Int16})::BitArray
    R = cfg.rows
    C = cfg.columns
    n = Int16(cfg.n_in)
    active = falses(R, C)
    xs = Int16.(genes[:, :, 1] .- n)
    ys = Int16.(genes[:, :, 2] .- n)
    fs = Int16.(genes[:, :, 3])
    for i in eachindex(outputs)
        active = recur_active(active, cfg.n_in, outputs[i] - n, xs, ys, fs,
                              cfg.two_arity)
    end
    active
end

function CGPInd(cfg::NamedTuple, chromosome::Array{Float64},
                genes::Array{Float64}, outputs::Array{Int16}; kwargs...)::CGPInd
    R = cfg.rows
    C = cfg.columns
    nodes = Array{Node}(undef, R * C + cfg.n_in)
    p = Float64[]
    for i in 1:cfg.n_in
        nodes[i] = Node(0, 0, f_null, p, false)
    end
    i = cfg.n_in
    active = find_active(cfg, genes, outputs)
    for y in 1:C
        for x in 1:R
            i += 1
            if cfg.n_parameters > 0
                p = genes[x, y, 4:end]
            end
            nodes[i] = Node(Int16(genes[x, y, 1]), Int16(genes[x, y, 2]),
                            cfg.functions[Int16(genes[x, y, 3])], p,
                            active[x, y])
        end
    end
    kwargs_dict = Dict(kwargs)
    # Use given input buffer or default to Array{Float64, 1} type
    if haskey(kwargs_dict, :buffer)
        buffer = kwargs_dict[:buffer]
    else
        buffer = Array{MType}(nothing, R * C + cfg.n_in)
        buffer .= 0.0
    end
    fitness = -Inf .* ones(cfg.d_fitness)
    CGPInd(cfg.n_in, cfg.n_out, cfg.n_parameters, chromosome, genes, outputs,
           nodes, buffer, fitness)
end

function get_maxs(cfg)
    R = cfg.rows
    C = cfg.columns
    P = cfg.n_parameters
    maxs = collect(1:R:R*C) .- 1
    maxs = round.((R*C .- maxs) .* cfg.recur .+ maxs)
    maxs = min.(R*C + cfg.n_in, maxs .+ cfg.n_in)
    maxs = repeat(maxs, 1, R)'
    maxs
end

function CGPInd(cfg::NamedTuple, chromosome::Array{Float64}; kwargs...)::CGPInd
    R = cfg.rows
    C = cfg.columns
    P = cfg.n_parameters
    # chromosome: node genes, output genes
    genes = reshape(chromosome[1:(R*C*(3+P))], R, C, 3+P)
    # TODO: recurrency is ugly and slow
    maxs = get_maxs(cfg)
    genes[:, :, 1] .*= maxs
    genes[:, :, 2] .*= maxs
    genes[:, :, 3] .*= length(cfg.functions)
    genes[:, :, 4:3+P] .*= cfg.param_max
    genes[:, :, 1:3] = Int16.(ceil.(genes[:, :, 1:3])) # ceil all genes except parameters that stay in [0,1]
    outputs = Int16.(ceil.(chromosome[(R*C*(3+P)+1):end] .* (R * C + cfg.n_in)))
    CGPInd(cfg, chromosome, genes, outputs; kwargs...)
end

function get_chromosome(cfg::NamedTuple, ind::CGPInd)
    genes, outputs = deepcopy(ind.genes), deepcopy(ind.outputs)
    R = cfg.rows
    C = cfg.columns
    P = cfg.n_parameters
    maxs = get_maxs(cfg)
    # genes to chromo
    x_chromo = ((genes[:, :, 1] .- 0.5) ./ maxs)[:]
    y_chromo = ((genes[:, :, 2] .- 0.5) ./ maxs)[:]
    f_chromo = ((genes[:, :, 3] .- 0.5) ./ length(cfg.functions))[:]
    p_chromo = (genes[:, :, 4:3+P] ./cfg.param_max)[:]
    # outputs
    o_chromo = outputs ./ (R * C + cfg.n_in)
    # order
    chromosome = 0 * ones(cfg.rows * cfg.columns * (3 + cfg.n_parameters) + cfg.n_out)
    chromosome[1:C] .= x_chromo
    chromosome[C+1:2C] .= y_chromo
    chromosome[2C+1:3C] .= f_chromo
    chromosome[3C+1:R*C*(3+P)] = p_chromo
    chromosome[(R*C*(3+P)+1):end] .= o_chromo
    chromosome
end

function CGPInd(cfg::NamedTuple; kwargs...)::CGPInd
    function chromo_to_genes(cfg::NamedTuple, chromosome; return_output_genes=false)
        R = cfg.rows
        C = cfg.columns
        P = cfg.n_parameters
        genes = reshape(chromosome[1:(R*C*(3+P))], R, C, 3+P)
        
        maxs = get_maxs(cfg)
    
        genes[:, :, 1] .*= maxs
        genes[:, :, 2] .*= maxs
        genes[:, :, 3] .*= length(cfg.functions)
        genes[:, :, 4:3+P] .*= cfg.param_max
        genes[:, :, 1:3] = Int16.(ceil.(genes[:, :, 1:3])) # ceil all genes except parameters that stay in [0,1]
        outputs = Int16.(ceil.(chromosome[(R*C*(3+P)+1):end] .* (R * C + cfg.n_in)))
        if return_output_genes 
            return genes, outputs
        else
            return genes
        end
    end
    
    function genes_to_chromo(cfg::NamedTuple, genes, outputs)
        R = cfg.rows
        C = cfg.columns
        P = cfg.n_parameters
    
        maxs = get_maxs(cfg)
    
        # genes to chromo
        x_chromo = (genes[:, :, 1] ./ maxs)[:]
        y_chromo = (genes[:, :, 2] ./ maxs)[:]
        f_chromo = (genes[:, :, 3] ./ length(cfg.functions))[:]
        p_chromo = (genes[:, :, 4:3+P] ./cfg.param_max)[:]
        # outputs
        o_chromo = outputs ./ (R * C + cfg.n_in)
        # order
        chromosome = 0 * ones(cfg.rows * cfg.columns * (3 + cfg.n_parameters) + cfg.n_out)
        chromosome[1:C] .= x_chromo
        chromosome[C+1:2C] .= y_chromo
        chromosome[2C+1:3C] .= f_chromo
        chromosome[3C+1:R*C*(3+P)] = p_chromo
        chromosome[(R*C*(3+P)+1):end] .= o_chromo
        chromosome
    end
    chromosome = rand(cfg.rows * cfg.columns * (3 + cfg.n_parameters) + cfg.n_out)
    chr, outs = chromo_to_genes(cfg, chromosome; return_output_genes=true)
    chromosome = genes_to_chromo(cfg, chr, outs)
    CGPInd(cfg, chromosome; kwargs...)
end

function CGPInd(cfg::NamedTuple, ind::String)::CGPInd
    dict = JSON.parse(ind)
    CGPInd(cfg, Array{Float64}(dict["chromosome"]))
end

function copy(n::Node)
    Node(n.x, n.y, n.f, n.p, n.active)
end

function copy(ind::CGPInd)
    buffer = Array{MType}(nothing, length(ind.buffer))
    buffer .= 0.0
    nodes = Array{Node}(undef, length(ind.nodes))
    for i in eachindex(ind.nodes)
        nodes[i] = copy(ind.nodes[i])
    end
    CGPInd(ind.n_in, ind.n_out, ind.n_parameters, copy(ind.chromosome), copy(ind.genes),
           copy(ind.outputs), nodes, buffer, copy(ind.fitness))
end

function String(n::Node)
    JSON.json(Dict(:x=>n.x, :y=>n.y, :f=>string(n.f),  :p=>string(n.p),
              :active=>n.active))
end

function show(io::IO, n::Node)
    print(io, String(n))
end

function String(ind::CGPInd)
    JSON.json(Dict("chromosome"=>ind.chromosome, "fitness"=>ind.fitness,
                   "xs"=>ind.genes[:, :, 1][:], "ys"=>ind.genes[:, :, 2][:],
                   "fs"=>ind.genes[:, :, 3][:]))
end

function show(io::IO, ind::CGPInd)
    print(io, String(ind))
end

function get_active_nodes(ind::CGPInd)
    ind.nodes[[n.active for n in ind.nodes]]
end

function summary(io::IO, ind::CGPInd)
    print(io, string("CGPInd(", get_active_nodes(ind), ", ",
                     findall([n.active for n in ind.nodes]), ", ",
                     ind.outputs, " ,",
                     ind.fitness, ")"))
end

function interpret(i::CGPInd)
    x::AbstractArray->process(i, x)
end

function reset!(c::CGPInd)
    c.buffer .= 0.0
end

"""
    get_gene_indexes(ind::CGPInd, node_id::Integer)
Given an individual and the index of one of its nodes, return the indexes of the
chromosome used to encode this particular node.
"""
function get_gene_indexes(c::CGPInd, node_id::Integer)
    index_start = node_id - c.n_in
    step = length(c.nodes) - c.n_in
    index_end = (2 + c.n_parameters) * (length(c.nodes) - c.n_in) + node_id - c.n_in
    return index_start:step:index_end
end

"""
    get_genes(ind::CGPInd, node_id::Integer)::Array{Float64}
Given an individual and the index of one of its nodes, return the chromosome
used to encode this particular node.
Example:
    get_genes(ind, 42)
"""
function get_genes(c::CGPInd, node_id::Integer)::Array{Float64}
    if node_id > c.n_in
        return c.chromosome[get_gene_indexes(c, node_id)]
    else
        return zeros(3 + c.n_parameters)
    end
end

"""
    get_genes(c::CGPInd, nodes::Array{<:Integer})::Array{Float64}
Given an individual and an array of indexes of one of its nodes, return the
chromosomes used to encode these particular nodes.
Example:
    get_genes(ind, [7, 42])
"""
function get_genes(c::CGPInd, nodes::Array{<:Integer})::Array{Float64}
    if length(nodes) > 0
        return reduce(vcat, map(x->get_genes(c, x), nodes))
    else
        return Array{Float64}(0)
    end
end

"set the genes of node_id to genes"
function set_genes!(c::CGPInd, node_id::Integer, genes::Array{Float64})
    if node_id > c.n_in
        @assert length(genes) == 3 + c.n_parameters
        c.chromosome[get_gene_indexes(c, node_id)] = genes
    end
end

"a list for each node i of a list of all the nodes which have the node i as an input, and i"
function forward_connections(c::CGPInd)
    connections = [[i] for i in 1:length(c.nodes)]
    for ci in eachindex(c.nodes)
        conns = [c.nodes[ci].x, c.nodes[ci].y]
        for conn in conns
            if conn > 0
                push!(connections[conn], ci)
            end
        end
    end
    map(unique, connections)
end

"recursively walk back through inputs"
function recur_output_trace(c::CGPInd, ind::Int16, visited::Array{Int16})
    if ~(ind in visited)
        push!(visited, ind)
        if ind > c.n_in
            recur_output_trace(c, c.nodes[ind].x, visited)
            if CGPFunctions.arity[string(c.nodes[ind].f)] == 2
                recur_output_trace(c, c.nodes[ind].y, visited)
            end
        end
    end
    visited
end

"return a list of node indices which determine the output"
function get_output_trace(c::CGPInd, output_ind::Integer)
    recur_output_trace(c, c.outputs[output_ind], Array{Int16}(undef, 0))
end

"return a list of node indices which determine the output for multiple outputs"
function get_output_trace(c::CGPInd, outputs::Array{<:Integer})
    if length(outputs) > 0
        return unique(reduce(vcat, map(x->get_output_trace(c, x), outputs)))
    else
        return Array{Int16}(undef, 0)
    end
end

"return a list of node indices which determine the output for all outputs"
function get_output_trace(c::CGPInd)
    get_output_trace(c, collect(1:c.n_out))
end