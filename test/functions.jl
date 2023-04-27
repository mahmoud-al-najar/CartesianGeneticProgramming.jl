using Test
using CartesianGeneticProgramming

const global D = 3
const global smax = 10
const global snum = 5

function test_inputs(f::Function, inps::AbstractArray)
    out = copy(f(inps...))
    # println("f: $(string(f)), out: $out, f(inps...): $(f(inps...))")
    @test typeof(out) <: Union{Nothing, Float64, Array{Float64}} # CartesianGeneticProgramming.MType

    @test all(out == f(inps...)) # functions are idempotent
#    @test all(out .>= -1.0)
#    @test all(out .<= 1.0)
end

function test_constant_function(f::Function)
    for constant in -1:1
        c = Float64(constant)
        test_inputs(f, [c, c])
        for d in 1:D
            for s in Int64.(round.(range(1, smax, length=snum)))
                test_inputs(f, [c, c .* ones(repeat([s], inner=d)...)])
                test_inputs(f, [c .* ones(repeat([s], inner=d)...), c])
                for dy in 1:D
                    for sy in Int64.(round.(range(1, smax, length=snum)))
                        test_inputs(f, [c .* ones(repeat([s], inner=d)...),
                                        c .* ones(repeat([sy], inner=dy)...)])
                    end
                end
            end
        end
    end
end

function test_function(f::Function)
    test_inputs(f, [2 * rand() - 1, 2 * rand() - 1])
    for d in 1:D
        for s in Int64.(round.(range(1, smax, length=snum)))
            test_inputs(f, [2 * rand() - 1,
                            2 .* rand(repeat([s], inner=d)...) .- 1])
            test_inputs(f, [2 .* rand(repeat([s], inner=d)...) .- 1,
                            2 * rand()-1])
            for dy in 1:D
                for sy in Int64.(round.(range(1, smax, length=snum)))
                    test_inputs(f, [2 .* rand(repeat([s], inner=d)...) .- 1,
                                    2 .* rand(repeat([sy], inner=dy)...) .- 1])
                end
            end
        end
    end
end

function test_functions(functions::Array{Function})
    for f in functions
        test_function(f)
    end
end

# @testset "List processing functions" begin
#     functions = [
#         CGPFunctions.f_head,
#         CGPFunctions.f_last,
#         CGPFunctions.f_tail,
#         CGPFunctions.f_diff,
#         CGPFunctions.f_avg_diff,
#         CGPFunctions.f_reverse,
#         CGPFunctions.f_push_back,
#         CGPFunctions.f_push_front,
#         CGPFunctions.f_set,
#         CGPFunctions.f_sum,
#         CGPFunctions.f_vectorize
#     ]
#     test_functions(functions)
# end

@testset "Mathematical functions" begin
    functions = [
        CGPFunctions.f_add,
        CGPFunctions.f_subtract,
        # CGPFunctions.f_mult,
        # CGPFunctions.f_div,
        # CGPFunctions.f_abs,
        # CGPFunctions.f_sqrt,
        # CGPFunctions.f_pow,
        # CGPFunctions.f_exp,

    #    CGPFunctions.f_sin,
    #    CGPFunctions.f_cos,
    #    CGPFunctions.f_tanh,
        # CGPFunctions.f_sqrt_xy,
        # CGPFunctions.f_lt,
        # CGPFunctions.f_gt
    ]
    test_functions(functions)
end

# @testset "Statistical functions" begin
#     functions = [
#         CGPFunctions.f_stddev,
#         CGPFunctions.f_skew,
#         CGPFunctions.f_kurtosis,
#         CGPFunctions.f_mean,
#         CGPFunctions.f_median,
#         CGPFunctions.f_range,
#         CGPFunctions.f_round,
#         CGPFunctions.f_ceil,
#         CGPFunctions.f_floor,
#         CGPFunctions.f_maximum,
#         CGPFunctions.f_max,
#         CGPFunctions.f_minimum,
#         CGPFunctions.f_min
#     ]
#     test_functions(functions)
# end

# @testset "Logical functions" begin
#     functions = [
#         CGPFunctions.f_and,
#         CGPFunctions.f_or,
#         CGPFunctions.f_xor,
#         CGPFunctions.f_not
#     ]
#     test_functions(functions)
# end

# @testset "Miscellaneous functions" begin
#     functions = [
#         CGPFunctions.f_vecfromdouble,
#         CGPFunctions.f_nop,
#         CGPFunctions.f_zeros,
#         CGPFunctions.f_ones,
#         CGPFunctions.f_normalize
#     ]
#     test_functions(functions)
# end
