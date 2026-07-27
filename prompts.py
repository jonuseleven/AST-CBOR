"""
Prompt templates for all extraction and generation modules.

All prompts are embedded as Python string constants to eliminate
external file dependencies. Each template uses Python's str.format()
with named placeholders.
"""

# =============================================================================
# Extraction Module Prompts
# =============================================================================

NORMALIZER_PROMPT = """Analyze and summarize the optimization problem:

1. Summarize the problem.
2. List entities involved.
3. Describe entity attributes (use array for repetitive ones with actual data, describe rows/columns).
4. Identify quantities sought.
5. List constraints (use array for repetitive ones with actual data, describe rows/columns).
6. State optimization objective.

Output in plain text only:
Summary of original problem: [summary]
Entities involved: [list]
Entity attributes: [desc/array with data]
Quantities sought: [what]
Constraints: [list/array with data]
Optimization objective: [obj]

CRITICAL — PRESERVE INEQUALITY SEMANTICS:
When summarizing constraints, you MUST preserve the EXACT inequality direction from the original problem.
- "needs X", "requires X", "must have at least X", "demand for X" → write as "≥ X" (NOT "equals X" or "= X")
- "at most X", "cannot exceed X", "no more than X" → write as "≤ X"
- "exactly X", "must equal X", "precisely X" → write as "= X"
- The word "need" or "require" in English means a minimum threshold (≥), NOT exact equality (=).
- Do NOT rephrase a demand-satisfaction constraint ("needs 74 tons") as equality ("must equal 74 tons").
- If the original says a region "has 42 tons but needs 74 tons", the constraint is: final_amount ≥ 74 (NOT final_amount = 74).
- If the original says "requires at least 83 grams", the constraint is: total ≥ 83 (NOT total = 83).

CRITICAL — MATHEMATICAL PRESERVATION:
- Copy ALL numerical values EXACTLY as they appear. Do not round or approximate.
- Preserve the original constraint direction: if the problem says "at least", use ≥; if "at most", use ≤.
- Do NOT combine multiple distinct constraints into one summary constraint.
- List each constraint separately, even if they are of similar type.

{example_block}
Problem:
{problem_description}"""


ENTITY_EXTRACTOR_PROMPT = """You are an expert in combinatorial optimization.

List ONLY the physical objects or decision-variable entities involved.
Exclude: abstract concepts (profit, budget, time, percentage, constraints, objectives),
         processes (transport, allocation, generation),
         or any numeric/percentage values.

When multiple similar items exist (e.g., multiple warehouses, machines, routes), group them as a single entity type if they have similar properties and functions.

Output format: Answer: entity1, entity2, entity3
Provide EXACTLY ONE line starting with "Answer:" followed by a comma-separated list.
DO NOT repeat the same line multiple times.
DO NOT include any other text, explanations or formatting.

{example_block}Now answer the following problem with the same format.
Problem:
{target_problem}"""


ATTRIBUTE_EXTRACTOR_PROMPT = """You are an expert in solving combinatorial optimization problems.
Based on the description and identified entities, list ONLY the ESSENTIAL attributes of those entities that are critical for formulating the optimization problem.
DO NOT include any explanations or reasoning.
Start your response with "Answer:" followed by entity-attribute mappings in this format: "entity1: attr1 (explanation1, value_range1), attr2 (explanation2, value_range2); entity2: attr3 (explanation3, value_range3)"
ONLY include attributes that are explicitly mentioned or strongly implied in the problem description.
DO NOT include generic attributes that are not specifically relevant to this problem.
Keep your response concise and focused.

For each attribute you identify, you must also provide:
1. A brief explanation of what the attribute represents
2. Possible values or value ranges if mentioned in the problem (use "Unknown" if not specified)
3. If the value has a unit (like thousands, millions, etc.), include it in the value range description

CRITICAL — CONTRADICTORY INFORMATION RESOLUTION:
- If the problem gives TWO different numerical values for the same quantity (e.g., "each unit gives 4 impact points" AND "effectiveness = 2*advertising + 3*promotion"), use ONLY the value that appears in the constraint/objective context, NOT the standalone "per unit" description
- If a formula and a "per unit" description conflict, the formula in the constraint/objective context takes priority
- If an attribute value is given in the constraint context as a coefficient, prefer that over a standalone descriptive number
- NEVER invent two different values for the same attribute — if there are two numbers, pick the most constraint-relevant one

CRITICAL — UNIT AWARENESS:
- If a variable measures "dollars spent" and the cost is also "dollars", the objective coefficient is 1 (not a separate per-dollar multiplier)
- If a variable measures "number of units" and each unit costs $C, the objective coefficient is C
- Identify whether each variable is a COUNT (integer, measured in items/people/acres) or a MONETARY amount (continuous, measured in dollars)

{example_block}
Now answer the following problem with the same format.

Problem:
{target_problem}
Entities:
{target_entities}"""


OBJECTIVE_EXTRACTOR_PROMPT = """You are an expert in combinatorial optimization.
Based on the description, identified entities and their attributes, identify ONLY ONE optimization objective.
Also explain how this objective is calculated based on the given attributes.

Output format:
Answer: single_objective_with_optimization_direction
Calculation: explanation of how the objective is computed from the attributes

CRITICAL RULES:
1. Output EXACTLY ONE objective - never multiple objectives
2. Do NOT include any constraints in the objective
3. Do NOT repeat the same objective in different wording
4. Choose the most important/clear objective from the problem
5. If unsure, select the primary objective mentioned in the problem
6. The objective MUST include the optimization direction (maximize_ or minimize_) as a prefix
7. Your response MUST contain EXACTLY one line starting with "Answer:" and one line starting with "Calculation:"
8. DO NOT include any explanations or thoughts beyond the required format

UNIT AND COEFFICIENT RULES:
- If the variable represents DOLLARS SPENT, the objective coefficient is 1 (cost = dollars spent)
- If the variable represents NUMBER OF UNITS and each costs $C, the coefficient is C
- "Each dollar invested yields $5000" → if the variable is "dollars invested", coefficient = 1 (not 5000)
- "Cost per acre is $5 for X" → if X is measured in acres, coefficient = 5
- Check: does the Calculation use the attributes with their correct units?

{example_block}Now answer the following problem with the same format.

Problem:
{target_problem}
Entities:
{target_entities}
Attributes:
{target_attributes}"""


VARIABLE_EXTRACTOR_PROMPT = """You are an expert in combinatorial optimization.
Based on the description, identified entities and their attributes, identify the decision variables that need to be determined.
Also explain how these variables are calculated based on the given attributes.

Output format:
Answer: variable_name (type), variable_name (type), ...
Calculation: explanation of how the decision variables are computed from the attributes

CRITICAL RULES:
1. Output ONLY the decision variables that need to be solved for
2. Do NOT include the optimization objective
3. Do NOT include any constraints
4. Provide a clear calculation method showing how the decision variables relate to the attributes
5. Do NOT include optimization direction prefixes like maximize_ or minimize_
6. Your response MUST contain EXACTLY one line starting with "Answer:" and one line starting with "Calculation:"
7. DO NOT include any explanations or thoughts beyond the required format

VARIABLE TYPE RULES (CRITICAL — type annotation is REQUIRED):
- EVERY variable in the Answer line MUST have a type annotation: "name (type)"
- Valid types: continuous, integer, binary
- "integer": counts of items, people, animals, vehicles, trips, units produced (things you count)
- "binary": yes/no decisions, choose/don't choose, whether to invest, whether to visit
- "continuous": amounts measured in weight, dollars, time, area, distance, percentages, servings
- For TSP/routing: edge variables are binary, subtour rank variables are continuous
- For flow problems: flow amounts per edge are continuous
- For scheduling: if people count matters and the problem says "whole numbers" → integer; otherwise continuous
- For transportation: shipment quantities are continuous (tons, units can be split)

COMPLETENESS RULES:
- Include ALL variables needed to compute the objective and all constraints
- For routing (TSP/VRP): list BOTH edge binary variables AND subtour elimination variables
- For multi-commodity flow: list flow variables for EACH commodity on EACH edge
- Do NOT collapse a full set of variables into one abstract name like "route_order" or "tour_sequence"
- Name each variable specifically: "shipment_W1_to_W2" not "shipment_quantity"

{example_block}Now answer the following problem with the same format.

Problem:
{target_problem}
Entities:
{target_entities}
Attributes:
{target_attributes}"""


CONSTRAINT_EXTRACTOR_PROMPT = """You are an expert in combinatorial optimization. Based on the problem description and the identified entities and attributes, list ALL constraints that limit the solution space. For each constraint, provide both a natural language explanation and a mathematical representation using the defined attributes.

Output format:
Answer: constraint1, constraint2, constraint3, ...
Explanation: constraint1: natural language explanation of constraint1
Calculation: constraint1: mathematical representation using defined attributes with correct inequality/equality signs
Explanation: constraint2: natural language explanation of constraint2
Calculation: constraint2: mathematical representation using defined attributes with correct inequality/equality signs
...

CRITICAL RULES — DISAMBIGUATION:
1. "at least twice as much X as Y" → X >= 2*Y (NOT Y >= 2*X)
2. "X exceeds Y by at most K" → X - Y <= K (NOT X <= Y + K)
3. "twice the number of A and B" in context of a sum → 2*(A + B), NOT 2*A + B
4. "the sum of three times X and Y" → 3*X + Y, NOT 3*(X + Y)
5. "cannot exceed X by more than K" → Y - X <= K
6. When in doubt about inequality direction, state the constraint in natural language first, then convert

CRITICAL RULES — COMPLETENESS:
1. List ALL constraints from the problem - do not skip any
2. Include non-negativity constraints for ALL variables
3. If variables are INTEGER as stated in the problem, add an explicit integer constraint
4. If the problem implies a minimum quantity (e.g., "invest in both properties" → each >= 1), include it
5. Each constraint should appear in the Answer list
6. Each constraint in the Answer list MUST have both Explanation and Calculation entries
7. Use the exact same constraint names in Explanation and Calculation lines as listed in the Answer
8. Mathematical representations should use the actual attribute names, not generic variables like x, y
9. Constraints should be atomic (one condition per constraint), not compound
10. Do NOT include the objective function among the constraints
11. Do NOT include redundant or tautological constraints
12. Make sure your constraints are complete phrases without truncation

CRITICAL — CONCRETE VALUES (NOT abstract parameters):
- NEVER use abstract parameter names like "capacity(i,j)", "demand[i]", "cost(i,j)" in Calculation lines
- ALWAYS write the actual numerical values from the problem data
- For a capacity matrix, write: "f(0,1) ≤ 6, f(0,2) ≤ 1, f(0,3) ≤ 14, ..." with actual numbers
- For a demand list, write: "x1 ≥ 74, x2 ≥ 476, ..." with actual numbers
- If there are too many values to list individually, write them compactly: "∀i,j: f(i,j) ≤ [matrix of 9×9 values: row0=[0,6,1,14,...], row1=[2,0,5,...], ...]"
- The Calculation line MUST contain enough information for someone to write code WITHOUT looking back at the original problem

CRITICAL RULES — INEQUALITY DIRECTION:
- For demand satisfaction constraints: use >= (at least satisfy demand)
- For resource limitation constraints: use <= (not exceeding available resources)
- For balance constraints: use == (precise balance)
- For capacity constraints: use <= (not exceeding capacity)
- For minimum requirement constraints: use >= (at least meeting minimum)
- For constraints like "need", "require", "must have": use >= (at least meet the requirement), NOT ==

{example_block}
Now answer the following problem with the same format.

Problem:
{target_problem}
Entities:
{target_entities}
Attributes:
{target_attributes}"""


# =============================================================================
# Gurobi Generation Prompts
# =============================================================================

GUROBI_MATH_MODEL_PROMPT = """You are an expert in mathematical optimization modeling.

You are given a JSON object that describes a linear / combinatorial optimization problem.
The JSON already contains entities, attributes, objectives, decision variables, and constraints.

================ JSON INPUT ================
{json_input}
===========================================

================ DETAILED ATTRIBUTES ================
{detailed_attributes}
===========================================

Your task:
Extract a precise mathematical optimization model.

============ DISAMBIGUATION RULES (apply BEFORE modeling) ============

1. IDENTIFY AMBIGUOUS PHRASES:
   - "twice the number of A and B" means 2*(A+B), NOT 2*A + B
   - "A exceeds B by more than K" means A - B >= K, NOT A > B + K
   - "at least twice as much X as Y" means X >= 2*Y, NOT Y >= 2*X
   - "cannot exceed A by more than K" means X - A <= K, NOT X <= A + K
   - When multiple numerical descriptions exist for the same quantity, use the one that appears in the constraints/attributes section of the JSON

2. RESOLVE CONTRADICTORY INFORMATION:
   - If the problem gives two different formulas for the same quantity, prefer the one stated in the objectives or constraints section
   - If "each unit costs $C" appears, the cost coefficient is C, not some other value
   - If a variable is described in dollars AND has a "per unit" cost, the variable represents NUMBER OF UNITS, and the cost coefficient applies per unit

3. UNIT AND SCALE CHECK:
   - If a variable is measured in dollars/units/people, do NOT multiply it by another dollar/unit/people coefficient unless explicitly stated
   - "Each dollar invested yields $5000" → the coefficient is 1, not 5000 (investing $1 returns $5000 in value, but the COST is $1 per dollar invested)
   - "Cost per acre is $5 for X and $10 for Y" → objective coefficient for X is 5, for Y is 10
   - If the answer is expected in dollars, the objective must also be in dollars

4. CONSTRAINT COMPLETENESS CHECK:
   - Does the problem imply a minimum purchase/usage? (e.g., "invest in two properties" → at least 1 of each)
   - Are there non-negativity constraints for ALL variables?
   - If variables represent counts/quantities, they MUST be >= 0
   - If the problem mentions "whole numbers" or "integers", mark variables as INTEGER
   - If a constraint mentions "at least" or "minimum", it's a >= constraint
   - If a constraint mentions "at most" or "cannot exceed", it's a <= constraint

5. PARAMETER VERIFICATION:
   - Count parameters: if the JSON lists N costs/distances/values, ALL N must appear in the model
   - For transportation problems: verify supply[i] and demand[j] for ALL i,j
   - For TSP/routing: verify the cost/distance matrix is complete
   - If a parameter appears in the JSON but NOT in your model, you missed a constraint

6. INDEXING CONSISTENCY:
   - If the problem uses 1-based indexing (Month 1, Month 2, ...), your model MUST also use 1-based indexing
   - If you define variables with 0-based indexing in code, add a NOTE: "Indices: 0-based (month 0 = January)"
   - Be explicit about the index range: "for t = 1,...,6" or "for i = 0,...,8"
   - NEVER mix 0-based and 1-based indices in the same constraint group

7. NONLINEAR CONSTRAINT DETECTION:
   - If you see a constraint like "X * Y == 0" where X and Y are both variables, this is NONLINEAR
   - Do NOT write nonlinear constraints directly. Instead:
     * Introduce a binary variable z
     * Replace "X * Y == 0" with: "X <= M * z" and "Y <= M * (1 - z)" where M is a large constant
   - If you see "either X or Y must be zero", use the binary variable approach above

8. SANITY CHECK (do this mentally, do NOT include in output):
   - Can the objective reach a finite value given the constraints?
   - If the optimal value would be zero or infinity, you are missing constraints
   - Does the model have at least one variable? At least one constraint?
   - Are all coefficients non-negative when the problem implies they should be?
   - Check: are ALL numerical values from the problem description present in the model?

============ OUTPUT FORMAT ============

Output MUST contain exactly the following three sections
(in this exact order and with these exact headers):

Decision Variables:
- List each decision variable using symbolic names (e.g., x_A, x_B)
- Clearly state domain (continuous / integer / binary)
- Include non-negativity or bounds if applicable

Objective:
- Write a single objective function
- Use "Minimize:" or "Maximize:"

Constraints:
- Write each constraint as a standard algebraic inequality or equality
- One constraint per line

FORMAT EXAMPLE:
Decision Variables:
- x_A: number of units of product A to produce (integer, non-negative)
- x_B: number of units of product B to produce (integer, non-negative)

Objective:
Minimize: 5*x_A + 3*x_B

Constraints:
- 2*x_A + x_B >= 100
- x_A + 3*x_B <= 80
- x_A >= 10

STRICT RULES:
- Do NOT generate any code
- Do NOT explain reasoning
- Do NOT repeat the JSON
- Do NOT add any extra text
- Include ALL numerical values from the JSON
- Follow the disambiguation rules above for any ambiguous phrasing"""


GUROBI_CORE_PROMPT = """You are a professional Python developer specialized in Gurobi optimization.

Below is a mathematical optimization model:

================ MATHEMATICAL MODEL ================
{math_model}
===================================================

Your task:
Convert this model into Gurobi Python code. ONLY output the following elements:
1. Variable definitions using model.addVar or model.addVars
2. Objective definition using model.setObjective
3. Constraint definitions using model.addConstr or model.addConstrs

CRITICAL RULES:
- ONLY output valid Python code
- DO NOT output any instructions, explanations, or rule lists
- DO NOT repeat the mathematical model
- DO NOT include import statements
- DO NOT include model creation
- DO NOT call optimize()
- DO NOT print results
- DO NOT add comments or explanations
- DO NOT include any text that is not Python code
- DO NOT include any headers or footers
- DO NOT include any metadata
- Use <= for "less than or equal to" constraints
- Use >= for "greater than or equal to" constraints
- Gurobi does NOT support < or > operators - convert these to <= or >= respectively
- For decision variables representing counts (e.g., number of pills, vehicles, people, items, catalysts, units, etc.), use vtype=grb.GRB.INTEGER
- For decision variables that can be continuous, use vtype=grb.GRB.CONTINUOUS (default)
- For binary variables, use vtype=grb.GRB.BINARY
- Always explicitly specify the variable type when defining variables using the vtype parameter
- COUNT parameters: verify that every numerical value from the math model appears exactly once in the code
- If the math model has N constraints, output exactly N addConstr/addConstrs calls
- Do NOT drop, skip, or simplify any constraint
- Preserve ALL coefficient values exactly as they appear in the math model
- Use actual numerical values from the mathematical model in the code - do NOT use placeholders or 0s
- If a constraint coefficient is 0, include it anyway (do not skip terms)

ANTI-PLACEHOLDER — CRITICAL:
- NEVER write: "# capacity data needed" or "# replace with actual values" or any comment about missing data
- NEVER write: "capacity = [...]  # fill in" or any placeholder list
- If the math model gives specific numbers, use THOSE EXACT NUMBERS in the code
- If you cannot find a numerical value in the math model, the constraint is probably not needed — skip it entirely
- Every addConstr call MUST contain concrete numerical coefficients from the math model
- Do NOT create a constraint that references an undefined variable or parameter
- Check: does every number in the math model appear somewhere in your code? If not, you missed a constraint or coefficient

FOR IMPROVED CODE STRUCTURE:
- When multiple variables or constraints follow a pattern, use dictionaries, loops, or list comprehensions instead of defining each individually
- Use model.addVars for creating multiple similar variables at once
- Use model.addConstrs for creating multiple similar constraints at once
- Define ALL variables before using them in constraints
- If using network flow variables, create a structure that contains all valid arcs/connections and define variables only for those

Begin your response with the Python code immediately, without any text before or after."""


GUROBI_FULL_PROMPT = """You are an optimization engineer.

Below is a Gurobi modeling code snippet:

================ GUROBI CORE CODE ================
{core_code}
=================================================

Your task:
Produce a complete, executable Python script that solves the optimization problem.

The script MUST:
- Import gurobipy as grb
- Create a Gurobi model
- Set model.Params.Seed = 0 for reproducible solver behavior
- Include the provided modeling code unchanged
- Call model.optimize()
- Print exactly one machine-readable solver status line after optimize():
  - "Solver status: OPTIMAL" when model.Status == grb.GRB.OPTIMAL
  - "Solver status: INFEASIBLE" when model.Status == grb.GRB.INFEASIBLE
  - "Solver status: UNBOUNDED" when model.Status == grb.GRB.UNBOUNDED
  - "Solver status: INF_OR_UNBD" when model.Status == grb.GRB.INF_OR_UNBD
  - "Solver status: OTHER" for every other status
- Only in the OPTIMAL branch, print the optimal objective value and variables:
    for var in model.getVars():
        print(f"{{var.VarName}}: {{var.X}}")

CRITICAL RULES:
- Do NOT add example data
- Do NOT add explanations
- Do NOT define functions or classes
- Output ONLY a single Python script
- Make sure to include the loop for printing all variables: for var in model.getVars(): print(f"{{var.VarName}}: {{var.X}}")
- Ensure the indentation of the print statement inside the for loop is correct
- Do NOT add any duplicate code or repeated sections
- Make sure to print the optimal objective value as well, using print(f"Optimal objective value: {{model.ObjVal}}") or similar
- Never access model.ObjVal or var.X outside the OPTIMAL branch
- If the model uses parameters from the JSON (like supply/demand values), make sure these are properly defined in the script
- Do NOT initialize parameters to 0 - use the actual values from the original JSON
- Add error handling to check if the model was solved successfully
- Ensure ALL variables are defined before being used in constraints in the core code

Begin your response with the Python code immediately, without any text before or after."""


GUROBI_CBR_REFINE_PROMPT = """You are an optimization engineer reviewing a generated Gurobi solver.

The original structured problem is authoritative:
================ PROBLEM JSON ================
{json_input}
==============================================

Initial solver code:
================ INITIAL CODE ================
{initial_code}
==============================================

AST-retrieved solved cases are included only as structural references:
================ RETRIEVED CASES =============
{retrieved_cases}
==============================================

Return a corrected, complete Python solver script for the original problem.

Rules:
- Use retrieved cases only for modeling and code-structure patterns.
- Never copy their problem-specific coefficients, variables, or constraints.
- Preserve every numerical fact from the original problem JSON.
- Import only gurobipy as grb plus standard math/itertools/collections modules if needed.
- Set model.Params.Seed = 0, call optimize(), and print exactly one solver status line.
- Use Solver status: OPTIMAL, INFEASIBLE, UNBOUNDED, INF_OR_UNBD, or OTHER.
- Access ObjVal and variable X values only in the OPTIMAL branch.
- In the OPTIMAL branch print: Optimal objective value: <value>.
- Output only executable Python code, without Markdown fences or explanations.
"""


# =============================================================================
# Judge Prompt
# =============================================================================

JUDGE_PROMPT = """You are an evaluation judge for structured extraction modules.
Evaluate ONLY against the original question. Do NOT assume missing facts.
You will be given extracted text for a single module.
Focus on whether the content covers all key elements the module should include.
Be lenient on structure/format; similar descriptions are acceptable.
If important content seems missing, mention it briefly in reason and lower
semantic_score noticeably (e.g., 1-2 points).
Be conservative in scoring: default to 3-4 for decent but not perfect outputs.
Use score=5 sparingly: only when content is complete, precise, and unambiguous.
If anything is slightly missing, unclear, or questionable, use 4 or lower.
If there are clear errors or major omissions, use 1-2.
Rules above are guidance only, not strict enforcement.
Score each dimension from 1 to 5:
- structural_score: clarity/organization only
- semantic_score: content is correct, complete, faithful to the question
- conciseness_score: no redundancy or irrelevant content
Return JSON only with keys: structural_score, semantic_score, conciseness_score, error_type, reason.
reason must be <= 80 ASCII chars.

Module expectation: {expectation}
Extra rule: {extra_rule}

Original question:
{question}

Module name:
{module_name}

Module text:
{module_text}"""


# =============================================================================
# One-Shot Examples (used in extraction pipeline)
# =============================================================================

EXAMPLE_PROBLEM = """Summary of original problem: Transport at least 300 ducks to shore using boats and canoes, with trip limits and a canoe trip proportion requirement, to minimize total time.
Entities involved: Boats, canoes, ducks, trips.
Entity attributes: Array [transport, ducks per trip, minutes per trip]: [['Boat', 10, 20], ['Canoe', 8, 40]]
Quantities sought: Number of boat trips, number of canoe trips.
Constraints: Array [description, inequality]: [['Ducks transported', '>=300'], ['Boat trips', '<=12'], ['Canoe trips proportion', '>=0.6*(total trips)']]
Optimization objective: Minimize total time in minutes."""

EXAMPLE_ENTITIES = "boats, canoes, ducks, trips"

EXAMPLE_ATTRIBUTES = ("boats: ducks_per_trip (ducks transported per boat trip, 10 ducks), "
                      "minutes_per_trip (minutes per boat trip, 20 minutes), "
                      "trip_limit (maximum boat trips, 12); "
                      "canoes: ducks_per_trip (ducks transported per canoe trip, 8 ducks), "
                      "minutes_per_trip (minutes per canoe trip, 40 minutes), "
                      "proportion_requirement (minimum proportion of canoe trips, 0.6 or 60%); "
                      "ducks: total_required (minimum ducks to transport, 300 ducks); "
                      "trips: total (total number of trips)")

EXAMPLE_CONSTRAINTS = "Minimum_ducks_transported, Maximum_boat_trips, Minimum_canoe_trip_proportion"

EXAMPLE_CONSTRAINT_EXPLANATIONS = """
Explanation: Minimum_ducks_transported: The problem requires transporting at least 300 ducks
Calculation: Minimum_ducks_transported: (boats.trips * boats.ducks_per_trip) + (canoes.trips * canoes.ducks_per_trip) >= ducks.total_required
Explanation: Maximum_boat_trips: Environmental regulations limit boat trips to at most 12
Calculation: Maximum_boat_trips: boats.trips <= 12
Explanation: Minimum_canoe_trip_proportion: Environmental policy requires at least 60% of trips to be by canoe
Calculation: Minimum_canoe_trip_proportion: canoes.trips >= 0.6 * (boats.trips + canoes.trips)
"""

EXAMPLE_OBJECTIVE = "minimize_total_transport_time"

EXAMPLE_OBJECTIVE_CALCULATION = ("Total transport time = (number of boat trips * time per boat trip) + "
                                 "(number of canoe trips * time per canoe trip)")

EXAMPLE_DECISION_VARIABLES = "number_of_boat_trips (integer), number_of_canoe_trips (integer)"

EXAMPLE_DECISION_VARIABLES_CALCULATION = ("Number of boat trips = boats.trips; "
                                          "Number of canoe trips = canoes.trips")


# =============================================================================
# Prompt Registry
# =============================================================================

PROMPT_REGISTRY = {
    "normalizer": NORMALIZER_PROMPT,
    "entity_extractor": ENTITY_EXTRACTOR_PROMPT,
    "attribute_extractor": ATTRIBUTE_EXTRACTOR_PROMPT,
    "objective_extractor": OBJECTIVE_EXTRACTOR_PROMPT,
    "variable_extractor": VARIABLE_EXTRACTOR_PROMPT,
    "constraint_extractor": CONSTRAINT_EXTRACTOR_PROMPT,
    "gurobi_math_model": GUROBI_MATH_MODEL_PROMPT,
    "gurobi_core": GUROBI_CORE_PROMPT,
    "gurobi_full": GUROBI_FULL_PROMPT,
    "judge": JUDGE_PROMPT,
}


def load_prompt(name: str) -> str:
    """Return a prompt template by name."""
    if name not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt name: {name}. Available: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]
