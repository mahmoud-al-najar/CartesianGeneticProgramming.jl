export single_point_crossover,
    random_node_crossover,
    aligned_node_crossover,
    proportional_crossover,
    output_graph_crossover,
    subgraph_crossover,
    discrete_phenotypic_crossover,
    crossover

"single point crossover, genes from p1 up to a random point, then genes from p2"
function single_point_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd)
    cpoint = rand(2:(min(length(c1.chromosome), length(c2.chromosome))-2))
    if rand(Bool)
        ngenes = deepcopy([c1.chromosome[1:cpoint]; c2.chromosome[(cpoint+1):end]])
        return CGPInd(cfg, ngenes)
    else
        ngenes = deepcopy([c2.chromosome[1:cpoint]; c1.chromosome[(cpoint+1):end]])
        return CGPInd(cfg, ngenes)
    end
end

"returns random input genes from c1 and c2 equally, not a full crossover operator"
function random_inputs(c1::CGPInd, c2::CGPInd)
    child_inputs = zeros(c1.n_in)
    parent = rand(Bool, c1.n_in)
    child_inputs[parent] = c1.chromosome[1:c1.n_in][parent]
    child_inputs[.~parent] = c2.chromosome[1:c1.n_in][.~parent]
    child_inputs
end

"returns random output genes from c1 and c2 equally, not a full crossover operator"
function random_outputs(c1::CGPInd, c2::CGPInd)
    child_outputs = zeros(c1.n_out)
    parent = rand(Bool, c1.n_out)
    s = length(c1.chromosome) - c1.n_out + 1
    child_outputs[parent] = c1.chromosome[s:end][parent]
    child_outputs[.~parent] = c2.chromosome[s:end][.~parent]
    child_outputs
end

"take random nodes from each parent equally, up to the size of the smaller parent"
function random_node_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd)
    l = min(length(c1.chromosome), length(c2.chromosome)) - max(c1.n_out, c2.n_out)
    p = rand(Bool, Int64(round(l / 3)))
    p_nodes = repeat(p, inner=3)
    append!(p_nodes, rand(Bool, c1.n_out))
    genes = rand(length(c1.chromosome))
    genes[p_nodes] = c1.chromosome[p_nodes]
    genes[.~p_nodes] = c2.chromosome[.~p_nodes]
    CGPInd(cfg, genes)
end

"align nodes based on position, then take from each parent equally, only works on PCGP"
function aligned_node_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd)
    p1_pos = get_positions(c1)
    p2_pos = get_positions(c2)
    min_nodes = min(length(c1.nodes)-c1.n_in, length(c2.nodes)-c2.n_in)
    p1_inds = c1.n_in + collect(1:min_nodes)
    p1_nodes = Array{Int64}(undef, 0)
    p2_nodes = Array{Int64}(undef, 0)
    for node in c1.n_in+(1:min_nodes)
        if rand() < 0.5
            i = indmin(abs.(p1_pos[p1_inds] - p2_pos[node]))
            append!(p1_nodes, [p1_inds[i]])
            p1_inds = deleteat!(p1_inds, i)
        else
            append!(p2_nodes, [node])
        end
    end
    genes = [random_inputs(c1, c2); random_outputs(c1, c2); get_genes(
        c1, p1_nodes); get_genes(c2, p2_nodes)]
    CGPInd(cfg, genes)
end

"choose a point between the genes from the two parents for a child individual"
function proportional_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd)
    genes = Array{Float64}(undef, 0)
    if length(c1.chromosome) == length(c2.chromosome)
        r = rand(length(c1.chromosome))
        genes = ((1 .- r) .* c1.chromosome) .+ (r .* c2.chromosome)
    elseif length(c1.chromosome) > length(c2.chromosome)
        r = rand(length(c2.chromosome))
        genes = ((1 .- r) .* c1.chromosome[1:length(c2.chromosome)]) .+ (r .* c2.chromosome)
        genes = [genes; c1.chromosome[(length(c2.chromosome)+1):end]]
    else
        r = rand(length(c1.chromosome))
        genes = ((1 .- r) .* c1.chromosome) .+ (r .* c2.chromosome[1:length(c1.chromosome)])
        genes = [genes; c2.chromosome[(length(c1.chromosome)+1):end]]
    end
    CGPInd(cfg, genes)
end

"crossover with a list of nodes"
function node_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd,
                        c1_nodes::Array{Int16}, c2_nodes::Array{Int16},
                        output_genes::AbstractArray = Array{Float64}(undef, 0))
    s = length(c1.chromosome) - c1.n_out
    if length(output_genes) == 0
        for o in 1:c1.n_out
            g = rand() < 0.5 ? c1.chromosome[s+o] : c2.chromosome[s+o]
            push!(output_genes, g)
        end
    end
    genes = Array{Float64}(undef, 0)
    for i in (c1.n_in+1):length(c1.nodes)
        if i in c1_nodes
            if i in c2_nodes
                g = rand() < 0.5 ? get_genes(c1, i) : get_genes(c2, i)
                append!(genes, g)
            else
                append!(genes, get_genes(c1, i))
            end
        else
            if i in c2_nodes
                append!(genes, get_genes(c2, i))
            else
                append!(genes, rand(3 + cfg.n_parameters))
            end
        end
    end
    append!(genes, output_genes)
    CGPInd(cfg, genes)
end

"split outputs equally between parents, then construct a child from their corresponding input graphs"
function output_graph_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd)
    p1_nodes = Array{Int16}(undef, 0)
    p2_nodes = Array{Int16}(undef, 0)
    output_genes = Array{Float64}(undef, 0)
    s = length(c1.chromosome) - c1.n_out
    for output in 1:c1.n_out
        if rand() < 0.5
            append!(p1_nodes, get_output_trace(c1, output))
            push!(output_genes, c1.chromosome[s+output])
        else
            append!(p2_nodes, get_output_trace(c2, output))
            push!(output_genes, c2.chromosome[s+output])
        end
    end
    c1_nodes = sort(unique(p1_nodes))
    c2_nodes = sort(unique(p2_nodes))
    node_crossover(cfg, c1, c2, c1_nodes, c2_nodes, output_genes)
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

"Take subgraphs from both parents equally, adding all nodes of the chosen subgraphs to the child"
function subgraph_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd)
    fc1 = forward_connections(c1)
    fc2 = forward_connections(c2)
    c1_nodes = []; c2_nodes = []
    for i in (c1.n_in+1):min(length(c1.nodes), length(c2.nodes))
        if rand(Bool)
            append!(c1_nodes, fc1[i])
        else
            append!(c2_nodes, fc2[i])
        end
    end
    if length(c1.nodes) < length(c2.nodes)
        for i in length(c1.nodes):length(c2.nodes)
            if rand(Bool)
                append!(c2_nodes, fc2[i])
            end
        end
    elseif length(c2.nodes) < length(c1.nodes)
        for i in length(c2.nodes):length(c1.nodes)
            if rand(Bool)
                append!(c1_nodes, fc1[i])
            end
        end
    end
    c1_nodes = Array{Int16}(unique(intersect(collect((c1.n_in+1):length(c1.nodes)), c1_nodes)))
    c2_nodes = Array{Int16}(unique(intersect(collect((c2.n_in+1):length(c2.nodes)), c2_nodes)))
    node_crossover(cfg, c1, c2, c1_nodes, c2_nodes)
end

function discrete_phenotypic_crossover(cfg::NamedTuple, c1::CGPInd, c2::CGPInd)
    G1 = deepcopy(c1.genes)
    G2 = deepcopy(c2.genes)
    N1 = get_output_trace(c1) .- c1.n_in; deleteat!(N1, N1 .<= 0)
    N2 = get_output_trace(c2) .- c2.n_in; deleteat!(N2, N2 .<= 0)
    
    l1 = length(N1)
    l2 = length(N2)
    lmin = min(l1, l2)
    lmax = max(l1, l2)

    i = 1
    while i <= lmin
        if rand() > 0.5
            if i == lmin-1 && l1 != l2
                r = rand(0:lmax-i)
                if l1 < l2
                    n1 = N1[i]
                    n2 = N2[i+r]
                else
                    n1 = N1[i+r]
                    n2 = N2[i]
                end
            else
                n1 = N1[i]
                n2 = N2[i]
            end
            f1 = G1[1, n1, 3]
            f2 = G2[1, n2, 3]
            G1[1, n1, 3] = f2
            G2[1, n2, 3] = f1
        end
        i += 1
    end
    ch1 = CGPInd(cfg, genes_to_chromo(cfg, G1, c1.outputs))
    ch2 = CGPInd(cfg, genes_to_chromo(cfg, G2, c2.outputs))
    return ch1, ch2 
end
