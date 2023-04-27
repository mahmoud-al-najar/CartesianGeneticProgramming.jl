export MType
export SorX
export CGPFunctions

module CGPFunctions
using Statistics
using Images
using StatsBase
using NumericalIntegration
using SeisNoise

global arity = Dict()

SorX = Union{Symbol, Expr}
MType = Union{Nothing, Float64, Array{Float64}}

function fgen(name::Symbol, ar::Int, s1::SorX, s2::SorX, s3::SorX, s4::SorX;
              safe::Bool=false)
    if ar == 1
        @eval function $name(x::Float64, y::MType, p::Array{Float64,1}=Float64[])::MType
            $s1
        end
        if safe
            @eval function $name(x::Array{Float64}, y::MType, p::Array{Float64,1}=Float64[])::MType
                try
                    return $s4
                catch
                    return x
                end
            end
        else
            @eval function $name(x::Array{Float64}, y::MType, p::Array{Float64,1}=Float64[])::MType
                $s4
            end
        end
    else
        @eval function $name(x::Float64, y::Float64, p::Array{Float64,1}=Float64[])::MType
            $s1
        end
        @eval function $name(x::Float64, y::Array{Float64}, p::Array{Float64,1}=Float64[])::MType
            $s2
        end
        if safe
            @eval function $name(x::Array{Float64}, y::Float64, p::Array{Float64,1}=Float64[])::MType
                try
                    return $s3
                catch
                    return x
                end
            end
            @eval function $name(x::Array{Float64}, y::Array{Float64}, p::Array{Float64,1}=Float64[])::MType
                try
                    return $s4
                catch
                    return x
                end
            end
        else
            @eval function $name(x::Array{Float64}, y::Float64, p::Array{Float64,1}=Float64[])::MType
                $s3
            end
            @eval function $name(x::Array{Float64}, y::Array{Float64}, p::Array{Float64,1}=Float64[])::MType
                $s4
            end
        end
    end
    arity[String(name)] = ar
end

function fgen(name::Symbol, ar::Int, s1::SorX, s2::SorX, s3::SorX;
              safe::Bool=false)
    fgen(name, ar, s1, s2, s2, s3; safe=safe)
end

function fgen(name::Symbol, ar::Int, s1::SorX, s2::SorX; safe::Bool=false)
    fgen(name, ar, s1, s1, s2, s2; safe=safe)
end

function fgen(name::Symbol, ar::Int, s1::SorX)
    fgen(name, ar, s1, s1, s1, s1)
end

function recur_active!(active::BitArray, ind::Int16, xs::Array{Int16},
                       ys::Array{Int16}, fs::Array{Int16},
                       two_arity::BitArray)::Nothing
    if ind > 0 && ~active[ind]
        active[ind] = true
        recur_active!(active, xs[ind], xs, ys, fs, two_arity)
        if two_arity[fs[ind]]
            recur_active!(active, ys[ind], xs, ys, fs, two_arity)
        end
    end
end

function find_active(cfg::Dict, genes::Array{Int16},
                     outputs::Array{Int16})::BitArray
    R = cfg["rows"]
    C = cfg["columns"]
    active = falses(R, C)
    xs = genes[:, :, 1] .- Int16(cfg["n_in"])
    ys = genes[:, :, 2] .- Int16(cfg["n_in"])
    fs = genes[:, :, 3]
    for i in eachindex(outputs)
        recur_active!(active, outputs[i] - Int16(cfg["n_in"]), xs, ys, fs,
                      cfg["two_arity"])
    end
    active
end


function eqsize(x::AbstractArray, y::AbstractArray)
    if ndims(x) != ndims(y)
        maxdim = max(ndims(x), ndims(y))
        x = repeat(x, ones(Int, maxdim)...)
        y = repeat(y, ones(Int, maxdim)...)
    end
    newx, newy = Images.paddedviews(0, x, y)
    (copy(newx), copy(newy))
end

function moving_sum(x::AbstractArray)
    if length(x) > 1
        x = deepcopy(x)
        for i in collect(1:length(x)-1)
            x[i] = x[i] + x[i+1]
        end
        x[1:end-1]
    else
        x
    end
end

# function scaled(x::Float64)
#     if isnan(x)
#         return 0.0
#     end
#     min(max(x, -1.0), 1.0)
# end

# function scaled(x::Array{Float64})
#     x[isnan.(x)] .= 0.0
#     min.(max.(x, -1.0), 1.0)
# end

function inf_test(x)
    abs(x) == Inf ? 0.0 : x
end

function normalized(x::Array{Float64})
    div = maximum(x) - minimum(x)
    if div > 0
        return (x .- minimum(x)) ./ div
    end
    # scaled(x)
    x
end

# List processing
fgen(:f_head, 1, :(x), :(is_empty(x)[1]))
fgen(:f_last, 1, :(x), :(x[end]))
fgen(:f_tail, 1, :(x), :(length(x) > 1 ? x[end-1:end] : x))
fgen(:f_diff, 1, :(x), :(length(x) > 1 ? diff(x, dims=1) : zeros(1)))
fgen(:f_avg_diff, 1, :(x),
     :(length(x) > 1 ? Statistics.mean(diff(x, dims=1)) : zeros(1)))
fgen(:f_reverse, 1, :(x), :(reverse(x[:])))
fgen(:f_push_back, 2, :([x; y]), :([x; y[:]]), :([x[:]; y]), :([x[:]; y[:]]))
fgen(:f_push_front, 2, :([y; x]), :([y[:]; x]), :([y; x[:]]), :([y[:]; x[:]]))
fgen(:f_set, 2, :(x), :(x * ones(size(y))), :(y * ones(size(x))),
     :(Statistics.mean(x) * ones(size(y))))
fgen(:f_sum, 1, :(x), :(sum(x)))
fgen(:f_vectorize, 1, :(x), :(x[:]))

# Mathematical
fgen(:f_add, 2, :(x + y), :(x .+ y), :(.+(eqsize(x, y)...)))
fgen(:f_abs_subtract, 2, :(abs(x - y)), :(abs.(x .- y)), :(abs.(.-(eqsize(x, y)...))))
fgen(:f_subtract, 2, :(x - y), :(x .- y), :(.-(eqsize(x, y)...)))
fgen(:f_mult, 2, :(x * y), :(x .* y), :(.*(eqsize(x, y)...)))
fgen(:f_div, 2, :(x / y), :(x ./ y), :(./(eqsize(x, y)...)))
fgen(:f_abs, 1, :(abs(x)), :(abs.(x)))
fgen(:f_sqrt, 1, :(sqrt(abs(x))), :(sqrt.(abs.(x))))
fgen(:f_pow, 2, :(abs(x) ^ abs(y)), :(abs(x) .^ abs.(y)), :(abs.(x) .^ abs(y)),
     :(.^(eqsize(abs.(x), abs.(y))...)))
fgen(:f_exp, 1, :((exp(x) - 1.0) / (exp(1.0) - 1.0)),
     :((exp.(x) .- 1.0) / (exp(1.0) - 1.0)))

fgen(:f_sin, 1, :(sin(inf_test(x))), :(sin.(inf_test.(x))))
fgen(:f_cos, 1, :(cos(inf_test(x))), :(cos.(inf_test.(x))))
fgen(:f_tanh, 1, :(tanh(inf_test(x))), :(tanh.(inf_test.(x))))

fgen(:f_sqrt_xy, 2, :(sqrt(x^2 + y^2) / sqrt(2.0)),
     :(sqrt.(x^2 .+ y.^2) ./ sqrt(2.0)),
     :(sqrt.(x.^2 .+ y^2) ./ sqrt(2.0)),
     :(sqrt.(.+(eqsize(x.^2, y.^2)...)) ./ sqrt(2.0)))
fgen(:f_lt, 2, :(Float64(x < y)), :(Float64.(x .< y)),
    :(Float64.(.<(eqsize(x, y)...))))
fgen(:f_gt, 2, :(Float64(x > y)), :(Float64.(x .> y)),
    :(Float64.(.>(eqsize(x, y)...))))

# Statistical
fgen(:f_stddev, 1, :(zeros(1)[1]), :(Statistics.std(x[:])); safe=true)
fgen(:f_skew, 1, :(x), :(StatsBase.skewness(x[:])); safe=true)
fgen(:f_kurtosis, 1, :(x), :(StatsBase.kurtosis(x[:])); safe=true)
fgen(:f_mean, 1, :(x), :(Statistics.mean(x)); safe=true)
fgen(:f_median, 1, :(x), :(Statistics.median(x)); safe=true)
fgen(:f_range, 1, :(x), :(maximum(x)-minimum(x)-1.0); safe=true)
fgen(:f_round, 1, :(round(x)), :(round.(x)); safe=true)
fgen(:f_ceil, 1, :(ceil(x)), :(ceil.(x)); safe=true)
fgen(:f_floor, 1, :(floor(x)), :(floor.(x)); safe=true)
fgen(:f_maximum, 1, :(x), :(maximum(is_empty(x))); safe=true)
fgen(:f_max, 2, :(max(x, y)), :(max.(x, y)), :(max.(eqsize(x, y)...)); safe=true)
fgen(:f_minimum, 1, :(x), :(minimum(is_empty(x))); safe=true)
fgen(:f_min, 2, :(min(x, y)), :(min.(x, y)), :(min.(eqsize(x, y)...)); safe=true)

# Logical
fgen(:f_and, 2, :(Float64((&)(Int(round(x)), Int(round(y))))),
     :(Float64.((&).(Int(round(x)), Int.(round.(y))))),
     :(Float64.((&).(Int.(round.(x)), Int(round(y))))),
     :(Float64.((&).(eqsize(Int.(round.(x)), Int.(round.(y)))...))))
fgen(:f_or, 2, :(Float64((|)(Int(round(x)), Int(round(y))))),
     :(Float64.((|).(Int(round(x)), Int.(round.(y))))),
     :(Float64.((|).(Int.(round.(x)), Int(round(y))))),
     :(Float64.((|).(eqsize(Int.(round.(x)), Int.(round.(y)))...))))
fgen(:f_xor, 2, :(Float64(xor(Int(abs(round(x))), Int(abs(round(y)))))),
     :(Float64.(xor.(Int(abs(round(x))), Int.(abs.(round.(y)))))),
     :(Float64.(xor.(Int.(abs.(round.(x))), Int(abs(round(y)))))),
     :(Float64.(xor.(eqsize(Int.(abs.(round.(x))), Int.(abs.(round.(y))))...))))
fgen(:f_not, 1, :(1 - abs(round(x))), :(1 .- abs.(round.(x))))

# Misc
fgen(:f_vecfromdouble, 1, :([x]), :(x))
fgen(:f_nop, 1, :(x))
fgen(:f_zeros, 1, :(zeros(1)[1]), :(zeros(size(x))))
fgen(:f_ones, 1, :(ones(1)[1]), :(ones(size(x))))
fgen(:f_normalize, 1, :(x), :(normalized(is_empty(x))))

function i_range(x)
    if typeof(x) == Float64
        if x > 0 && x != Inf && x < 10000 
            return collect(1.0:x)
        else
            return [1.0]
        end
    else
        if length(x) > 0 && all(abs.(x) .!= Inf)
            return collect(1.0:length(x))
        else
            return [1.0]
        end
    end
end

function is_empty(arr)
    length(arr) > 0 ? arr : [0.0]
end

function m_integrate(arr)
    length(arr) > 2 ? NumericalIntegration.integrate(collect(1:length(arr)), arr) : arr
end

function m_conv(v, u)
    if typeof(v) != Vector{Float64} || typeof(u) != Vector{Float64}
        return v
    end
    kernel = copy(length(u) < length(v) ? u : v)
    vector = copy(length(u) > length(v) ? u : v)

    vmean = mean(vector)
    vector .-= vmean
    out = vector .* 0
    l = length(kernel)
    for i in l+1:length(vector)
        wv = vector[Int(i-l)+1:Int(i)] .* kernel # (kernel ./ sum(kernel))
        out[Int(i)] = sum(wv)
    end

    return out .+ vmean
end

function m_detrend(x)
    if length(x) > 2
        return SeisNoise.detrend(x)
    else
        return x
    end
end

# MN
fgen(:f_irange, 1, :(i_range(x)), :(i_range(x)))
fgen(:f_lteq, 2, :(Float64(x <= y)), :(Float64.(x .<= y)),
    :(Float64.(.<=(eqsize(x, y)...))))
fgen(:f_gteq, 2, :(Float64(x >= y)), :(Float64.(x .>= y)),
    :(Float64.(.>=(eqsize(x, y)...))))
fgen(:f_negate, 1, :(-x), :(-x))
fgen(:f_tpow, 1, :(10.0^x), :(10.0 .^x))
fgen(:f_moving_sum, 1, :(x), :(moving_sum(x)))
fgen(:f_pop, 1, :(x), :(is_empty(x)[end]))
fgen(:f_integrate, 1, :(x), :(m_integrate(x)))
fgen(:f_detrend, 1, :(x), :(m_detrend(x)); safe=true)
fgen(:f_conv, 2, :(x), :(m_conv(x, y)))
end
