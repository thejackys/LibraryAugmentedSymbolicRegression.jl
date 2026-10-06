# Run this file in a separate Julia process. The stub cannot send requests.
ENV["START_LLAMASERVER"] = "false"

module LaSRLLMSuccessTests

using Test
using Random: MersenneTwister, seed!
using Logging: NullLogger, with_logger
using LibraryAugmentedSymbolicRegression: LibraryAugmentedSymbolicRegression, LaSROptions
using DynamicExpressions:
    AbstractExpressionNode, Expression, Node, count_nodes, eval_tree_array,
    get_contents, get_metadata
using JSON: json

const LF = LibraryAugmentedSymbolicRegression.LLMFunctionsModule
const PT = parentmodule(LF.CustomOpenAISchema)
const response = Ref("")
const requests = Any[]

function LF.aigenerate(
    ::LF.CustomOpenAISchema, prompt::Vector{<:PT.AbstractMessage}; kwargs...
)
    push!(requests, (; kwargs...))
    return (; content=response[])
end

const stub_method = which(
    LF.aigenerate, (LF.CustomOpenAISchema, Vector{PT.AbstractMessage})
)

function check_values(tree, X, expected, options)
    values, complete = eval_tree_array(tree, X, options.operators)
    @test complete
    @test values ≈ expected
end

function run_tests()
    seed!(11)
    @testset "LaSR successful LLM responses, no network" begin
        for T in (Float32, Float64)
            @testset "$T node and wrapper boundaries" begin
                options = LaSROptions(;
                    binary_operators=[+, -, *],
                    unary_operators=[sin],
                    variable_names=Dict(1 => "x0", 2 => "x1"),
                    prompts_dir=joinpath(pkgdir(LibraryAugmentedSymbolicRegression), "prompts") * "/",
                    idea_database=AbstractString[],
                    use_llm=true,
                    use_concepts=false,
                    api_key="stub-only",
                    model="stub-only",
                    api_kwargs=Dict("url" => "http://127.0.0.1:1/v1", "max_tokens" => 32),
                    http_kwargs=Dict("retries" => 0),
                    num_generated_equations=2,
                    verbose=false,
                )
                a, b = Node{T}(; feature=1), Node{T}(; feature=2)
                ex1 = Expression(a; operators=options.operators, variable_names=["x0", "x1"], label="first")
                ex2 = Expression(b; operators=options.operators, variable_names=["x0", "x1"], label="second")
                X = T[0.2 0.4 0.6; 0.3 0.7 0.9]

                for (text, expected) in (
                    ("x0 + x1", vec(X[1, :] + X[2, :])),
                    ("sin(x0)", sin.(vec(X[1, :]))),
                    ("x0", vec(X[1, :])),
                    ("2.5", fill(T(2.5), 3)),
                    ("y = x1", vec(X[2, :])),
                    ("x0 + 1", vec(X[1, :]) .+ one(T)),
                )
                    @testset "$text" begin
                        response[] = json([text])
                        before = length(requests)
                        mutation = LF.llm_mutate_tree(a, options)
                        random_tree = LF.llm_randomize_tree(a, 4, options, 2, MersenneTwister(11))
                        child1, child2 = LF.llm_crossover_trees(a, b, options)
                        for tree in (mutation, random_tree, child1)
                            @test tree isa AbstractExpressionNode{T}
                            check_values(tree, X, expected, options)
                        end
                        @test child2 === b

                        wrapped = (
                            LF.llm_mutate_tree(ex1, options),
                            LF.llm_randomize_tree(ex1, 4, options, 2, MersenneTwister(11)),
                        )
                        crossed = LF.llm_crossover_trees(ex1, ex2, options)
                        for ex in (wrapped..., crossed[1])
                            @test typeof(ex) === typeof(ex1)
                            @test get_metadata(ex) === get_metadata(ex1)
                            check_values(get_contents(ex), X, expected, options)
                        end
                        @test typeof(crossed[2]) === typeof(ex2)
                        @test get_metadata(crossed[2]) === get_metadata(ex2)
                        @test get_contents(crossed[2]) === b
                        @test length(requests) == before + 6
                    end
                end

                @testset "Two crossover candidates" begin
                    response[] = json(["x0 + x1", "x0 + x1"])
                    expected = vec(X[1, :] + X[2, :])
                    for tree in LF.llm_crossover_trees(a, b, options)
                        @test tree isa AbstractExpressionNode{T}
                        check_values(tree, X, expected, options)
                    end
                    children = LF.llm_crossover_trees(ex1, ex2, options)
                    for (ex, parent) in zip(children, (ex1, ex2))
                        @test typeof(ex) === typeof(parent)
                        @test get_metadata(ex) === get_metadata(parent)
                        check_values(get_contents(ex), X, expected, options)
                    end
                end

                @testset "Invalid data and parser sentinel" begin
                    marker = "NSYM_LLM_SUCCESS_EXECUTED"
                    previous = pop!(ENV, marker, nothing)
                    try
                        for payload in (
                            "", "not JSON", "[]", "null", "[\"x0\", 2]",
                            "{\"bad\": 2}", "```json\n[\"x0\",\n```",
                            json(["sin("]), json(["unknown(x0)"]),
                            json(["sin(", "unknown(x0)"]),
                            json(["1.0"]), json(["y = 1.0", "1.0"]),
                            "begin ENV[\"$marker\"] = \"true\"; [\"x0\"] end",
                        )
                            response[] = payload
                            before = length(requests)
                            with_logger(NullLogger()) do
                                @test LF.llm_mutate_tree(a, options) === a
                                first, second = LF.llm_crossover_trees(a, b, options)
                                @test first === a
                                @test second === b
                                fallback = LF._gen_llm_random_tree(3, options, 2, T)
                                @test fallback isa AbstractExpressionNode{T}
                                @test count_nodes(fallback) == 3
                                mutation = LF.llm_mutate_tree(ex1, options)
                                @test typeof(mutation) === typeof(ex1)
                                @test get_metadata(mutation) === get_metadata(ex1)
                                @test get_contents(mutation) === a
                                for (ex, parent) in zip(LF.llm_crossover_trees(ex1, ex2, options), (ex1, ex2))
                                    @test typeof(ex) === typeof(parent)
                                    @test get_metadata(ex) === get_metadata(parent)
                                    @test get_contents(ex) === get_contents(parent)
                                end
                                randomized = LF.llm_randomize_tree(ex1, 4, options, 2, MersenneTwister(11))
                                @test typeof(randomized) === typeof(ex1)
                                @test get_metadata(randomized) === get_metadata(ex1)
                                @test get_contents(randomized) isa AbstractExpressionNode{T}
                            end
                            @test length(requests) == before + 6
                            @test !haskey(ENV, marker)
                        end
                    finally
                        previous === nothing ? delete!(ENV, marker) : (ENV[marker] = previous)
                    end
                end
            end
        end
        @test !isempty(requests)
        @test all(kws -> kws.api_key == "stub-only" && kws.model == "stub-only", requests)
        @test all(kws -> kws.api_kwargs.url == "http://127.0.0.1:1/v1", requests)
    end
end

try
    run_tests()
finally
    Base.delete_method(stub_method)
end

end # module
