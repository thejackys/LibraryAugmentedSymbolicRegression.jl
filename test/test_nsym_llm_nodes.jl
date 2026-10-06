module NSYMLLMNodeTests

using Test
using Random: MersenneTwister, seed!
using Logging: NullLogger, with_logger
using DynamicExpressions:
    Node, GraphNode, Expression, get_contents, get_metadata, eval_tree_array
using LibraryAugmentedSymbolicRegression:
    LaSROptions, parse_expr, llm_randomize_tree, llm_mutate_tree, llm_crossover_trees
using LibraryAugmentedSymbolicRegression.LLMFunctionsModule:
    CustomOpenAISchema, SystemMessage, AIMessage
import LibraryAugmentedSymbolicRegression.LLMFunctionsModule: aigenerate

const ChatMessage = supertype(SystemMessage{String})
const response = Ref("")
const calls = Ref(0)

# This more-specific method replaces only the request boundary. Prompt creation,
# JSON parsing, expression parsing, and the exported operations remain active.
function aigenerate(::CustomOpenAISchema, ::Vector{<:ChatMessage}; kwargs...)
    @test kwargs[:model] == "nsym-success-stub"
    calls[] += 1
    return AIMessage(; content=response[], status=200, run_id=1)
end

function check_values(tree, expected, X, options)
    values, complete = eval_tree_array(tree, X, options.operators)
    @test complete
    @test values ≈ expected
end

function check_operations(first, second, expected, X, options)
    first_node, second_node = get_contents(first), get_contents(second)
    before = calls[]
    for (left, right) in ((first_node, second_node), (first, second))
        randomized = llm_randomize_tree(left, 8, options, 2, MersenneTwister(11))
        mutated = llm_mutate_tree(left, options)
        crossed, unchanged = llm_crossover_trees(left, right, options)
        for result in (randomized, mutated, crossed)
            @test result isa typeof(left)
            check_values(result, expected, X, options)
            if result isa Expression
                @test get_metadata(result) === get_metadata(left)
            end
        end
        @test unchanged isa typeof(right)
        if unchanged isa Expression
            @test get_contents(unchanged) === second_node
            @test get_metadata(unchanged) === get_metadata(right)
        else
            @test unchanged === second_node
        end
    end
    @test calls[] == before + 6
end

function run_tests()
    @testset "LLM operations return nodes from successful responses" begin
        for T in (Float32, Float64), node_type in (Node, GraphNode)
            @testset "$T / $node_type" begin
                options = LaSROptions(;
                    binary_operators=[+, -, *],
                    unary_operators=[sin],
                    node_type=node_type,
                    variable_names=Dict(1 => "x1", 2 => "x2"),
                    prompts_dir=joinpath(@__DIR__, "..", "prompts") * "/",
                    use_llm=true,
                    use_concepts=false,
                    use_concept_evolution=false,
                    idea_database=AbstractString[],
                    num_generated_equations=2,
                    api_key="stub-only",
                    model="nsym-success-stub",
                    api_kwargs=Dict("url" => "http://127.0.0.1:1/v1", "max_tokens" => 32),
                    http_kwargs=Dict("retries" => 0),
                    verbose=false,
                )
                X = T[-0.8 -0.1 0.2 0.9; 0.3 -0.5 0.7 -0.2]
                first = parse_expr(T, "x1 + x2", options)
                second = parse_expr(T, "x1 - x2", options)
                first_node, second_node = get_contents(first), get_contents(second)
                @test first_node isa node_type{T}
                @test second_node isa node_type{T}

                for (equation, expected) in (
                    ("sin(x1)", sin.(X[1, :])),
                    ("x1 + x2*x2", X[1, :] .+ X[2, :] .^ 2),
                    ("x1", X[1, :]),
                    ("2", fill(T(2), size(X, 2))),
                    ("y = x1", X[1, :]),
                    ("y = 2", fill(T(2), size(X, 2))),
                )
                    response[] = "[\"$equation\"]"
                    check_operations(first, second, expected, X, options)
                end

                @testset "Multiple crossover candidates" begin
                    response[] = "[\"sin(x1)\", \"x1*x2\"]"
                    expected = (sin.(X[1, :]), X[1, :] .* X[2, :])
                    before = calls[]
                    seed!(17)
                    for (left, right) in ((first_node, second_node), (first, second))
                        children = llm_crossover_trees(left, right, options)
                        for (child, parent) in zip(children, (left, right))
                            @test child isa typeof(parent)
                            values, complete = eval_tree_array(child, X, options.operators)
                            @test complete
                            @test any(target -> isapprox(values, target), expected)
                            if child isa Expression
                                @test get_metadata(child) === get_metadata(parent)
                            end
                        end
                    end
                    @test calls[] == before + 2
                end

                @testset "Invalid parse keeps the original nodes" begin
                    # These are valid JSON responses with invalid expressions.
                    for payload in (
                        "[\"unknown(x1)\"]",
                        "[\"unknown(x1)\", \"unknown(x2)\"]",
                        "[\"sin(\"]",
                        "[\"1\"]",
                        "[\"y = 1\"]",
                        "[]",
                    )
                        response[] = payload
                        before = calls[]
                        with_logger(NullLogger()) do
                            for (left, right) in ((first_node, second_node), (first, second))
                                randomized = llm_randomize_tree(left, 8, options, 2, MersenneTwister(11))
                                mutated = llm_mutate_tree(left, options)
                                crossed = llm_crossover_trees(left, right, options)
                                for (result, parent) in (
                                    (randomized, left), (mutated, left),
                                    (crossed[1], left), (crossed[2], right),
                                )
                                    @test result isa typeof(parent)
                                    if result isa Expression
                                        @test get_contents(result) === get_contents(parent)
                                        @test get_metadata(result) === get_metadata(parent)
                                    else
                                        @test result === parent
                                    end
                                end
                            end
                        end
                        @test calls[] == before + 6
                    end
                end

                check_values(first, X[1, :] .+ X[2, :], X, options)
                check_values(second, X[1, :] .- X[2, :], X, options)
            end
        end
    end
end

stub_method = which(aigenerate, (CustomOpenAISchema, Vector{ChatMessage}))
try
    run_tests()
finally
    Base.delete_method(stub_method)
end

end
