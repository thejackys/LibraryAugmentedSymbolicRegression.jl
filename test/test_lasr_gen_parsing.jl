# LaSR uses an LLM to generate expressions. This test case ensures that the parser can correctly format the LLM outputs.
# NSYM_SAFE_JSON_ONLY: Local nsym security regression coverage for untrusted LLM text.
println("Testing LaSR llm output parser")
using LibraryAugmentedSymbolicRegression: LaSROptions, parse_msg_content

include("test_params.jl")
options = LaSROptions(;
    default_params..., binary_operators=[+, *, ^, -], unary_operators=[sin, cos, exp]
)

include("static/sample_outputs.jl")

for (i, (llm_output, parsed_output)) in
    enumerate(zip(sample_llm_outputs, sample_parsed_outputs))
    @test parse_msg_content(llm_output, options) == parsed_output
end
println("Passed.")

for (i, (llm_output, parsed_output)) in
    enumerate(zip(expert_sample_llm_outputs, expert_sample_parsed_outputs))
    @test parse_msg_content(llm_output, options) == parsed_output
end
println("Passed.")

@test parse_msg_content("[\"x1 + x2\", \"x1 * x2\"]", options) ==
    ["x1 + x2", "x1 * x2"]
dict_output = parse_msg_content("""```json
{"first":"x1 + x2","second":"x1 * x2"}
```""", options)
@test Set(dict_output) == Set(["x1 + x2", "x1 * x2"])

execution_marker = "NSYM_LASR_PARSE_EXECUTED"
delete!(ENV, execution_marker)
malicious_output =
    "begin ENV[\"$execution_marker\"] = \"true\"; [\"x1 + x2\"] end"
@test parse_msg_content(malicious_output, options) == String[]
@test !haskey(ENV, execution_marker)

@test parse_msg_content("", options) == String[]
@test parse_msg_content("```json\n[\"x1\",\n```", options) == String[]
